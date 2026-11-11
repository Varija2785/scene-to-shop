from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOE

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = BASE_DIR / "vision" / "detection" / "test_images"
OUTPUT_DIR = BASE_DIR / "vision" / "detection" / "outputs" / "objects_clean"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLOE("yoloe-11s-seg-pf.pt")

image_files = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
)

for image_path in image_files:

    print(f"\nProcessing: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        continue

    results = model.predict(
        source=str(image_path),
        conf=0.30,
        iou=0.45,
        imgsz=768,
        verbose=False,
    )

    result = results[0]

    if result.masks is None:
        print("No masks found.")
        continue

    image_name = image_path.stem
    output_dir = OUTPUT_DIR / image_name
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes

    saved = 0

    for i, mask in enumerate(masks):

        confidence = float(boxes.conf[i])
        class_id = int(boxes.cls[i])
        label = result.names[class_id]

        x1, y1, x2, y2 = (
            boxes.xyxy[i]
            .cpu()
            .numpy()
            .astype(int)
        )

        h, w = image.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        # Resize segmentation mask to original image size
        mask_resized = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # ------------------------------------------------
        # 1. NORMAL COLOR CROP
        # ------------------------------------------------

        crop = image[y1:y2, x1:x2]

        # ------------------------------------------------
        # 2. TRANSPARENT MASK CROP
        # ------------------------------------------------

        crop_mask = mask_binary[y1:y2, x1:x2]

        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)

        rgba[:, :, 3] = crop_mask * 255

        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        object_id = saved + 1

        base_name = (
            f"{object_id:02d}_"
            f"{safe_label}_"
            f"{confidence:.2f}"
        )

        color_path = output_dir / f"{base_name}_crop.jpg"
        mask_path = output_dir / f"{base_name}_mask.png"

        cv2.imwrite(
            str(color_path),
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        cv2.imwrite(
            str(mask_path),
            rgba
        )

        print(
            f"{object_id:02d}. "
            f"{label} "
            f"confidence={confidence:.3f}"
        )

        saved += 1

    print(f"Saved {saved} objects.")

print("\n" + "=" * 65)
print("CLEAN OBJECT EXTRACTION COMPLETE")
print("=" * 65)
print(f"Output: {OUTPUT_DIR}")