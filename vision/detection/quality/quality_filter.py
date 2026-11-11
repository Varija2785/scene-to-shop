from pathlib import Path
import json
import numpy as np
from ultralytics import YOLOE
BASE_DIR = Path(__file__).resolve().parents[3]
IMAGE_DIR = BASE_DIR / "vision" / "detection" / "test_images"
OUTPUT_DIR = BASE_DIR / "vision" / "detection" / "quality" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "yoloe-11s-seg-pf.pt"
# Objects that describe the scene/background rather than a useful
# individual shopping object.
BACKGROUND_LABELS = {
    "home interior",
    "interior",
    "room",
    "floor",
    "wall",
    "ceiling",
    "background",
    "indoors",
    "outdoors",
    "house",
    "building",
    "architecture",
    "kitchen floor",
    "carpet floor",
}
CONF_THRESHOLD = 0.30
MIN_BOX_AREA_RATIO = 0.005
IOU_THRESHOLD = 0.70
def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
def clean_detections(result):
    names = result.names
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    image_h, image_w = result.orig_shape
    image_area = image_w * image_h
    candidates = []
    for i in range(len(boxes)):
        confidence = float(boxes.conf[i].item())
        class_id = int(boxes.cls[i].item())
        label = str(names[class_id]).lower().strip()
        coordinates = boxes.xyxy[i].cpu().numpy().tolist()
        area_ratio = box_area(coordinates) / image_area
        # Confidence filtering
        if confidence < CONF_THRESHOLD:
            continue
        # Background / scene filtering
        if label in BACKGROUND_LABELS:
            continue
        # Remove extremely tiny detections
        if area_ratio < MIN_BOX_AREA_RATIO:
            continue
        candidates.append({
            "label": label,
            "confidence": round(confidence, 4),
            "box": [round(float(v), 2) for v in coordinates],
            "area_ratio": round(float(area_ratio), 6),
        })
    # Highest confidence first
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    # Class-aware + overlap-aware deduplication.
    #
    # We keep two objects of the same class if they occupy clearly
    # different regions. Highly overlapping duplicates are collapsed.
    kept = []
    for candidate in candidates:
        duplicate = False
        for existing in kept:
            iou = calculate_iou(candidate["box"], existing["box"])
            if iou >= IOU_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept
def main():
    print("=" * 70)
    print("SCENE-TO-SHOP | OBJECT QUALITY + DEDUPLICATION")
    print("=" * 70)
    print(f"Loading model: {MODEL_NAME}")
    model = YOLOE(MODEL_NAME)
    image_paths = sorted(
        [
            p for p in IMAGE_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )
    if not image_paths:
        print(f"No test images found in: {IMAGE_DIR}")
        return
    manifest = {
        "model": MODEL_NAME,
        "confidence_threshold": CONF_THRESHOLD,
        "minimum_box_area_ratio": MIN_BOX_AREA_RATIO,
        "iou_threshold": IOU_THRESHOLD,
        "images": []
    }
    total_before = 0
    total_after = 0
    for image_path in image_paths:
        print()
        print(f"Processing: {image_path.name}")
        results = model.predict(
            source=str(image_path),
            conf=0.20,
            iou=0.45,
            imgsz=768,
            verbose=False
        )
        if not results:
            print("  No result returned.")
            continue
        result = results[0]
        before_count = len(result.boxes) if result.boxes is not None else 0
        cleaned = clean_detections(result)
        total_before += before_count
        total_after += len(cleaned)
        print(f"  Raw detections:     {before_count}")
        print(f"  Clean detections:   {len(cleaned)}")
        for idx, item in enumerate(cleaned, start=1):
            print(
                f"    {idx:02d}. "
                f"{item['label']} "
                f"| confidence={item['confidence']:.2f} "
                f"| area={item['area_ratio']:.3f}"
            )
        manifest["images"].append({
            "image": image_path.name,
            "raw_detection_count": before_count,
            "clean_detection_count": len(cleaned),
            "objects": cleaned
        })
    manifest["summary"] = {
        "total_raw_detections": total_before,
        "total_clean_detections": total_after,
        "detections_removed": total_before - total_after
    }
    output_file = OUTPUT_DIR / "quality_manifest.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print()
    print("=" * 70)
    print("QUALITY FILTER COMPLETE")
    print("=" * 70)
    print(f"Raw detections:     {total_before}")
    print(f"Clean detections:   {total_after}")
    print(f"Removed:            {total_before - total_after}")
    print(f"Manifest:           {output_file}")
    print("=" * 70)
if __name__ == "__main__":
    main()
