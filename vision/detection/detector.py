from pathlib import Path
from ultralytics import YOLOE

# --------------------------------------------------
# SCENE-TO-SHOP
# Prompt-Free Open-World Object Discovery
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = BASE_DIR / "vision" / "detection" / "test_images"
OUTPUT_DIR = BASE_DIR / "vision" / "detection" / "outputs" / "yoloe_prompt_free"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT:
# This is a PROMPT-FREE model.
# We do NOT provide classes.
# We do NOT call set_classes().
model = YOLOE("yoloe-11s-seg-pf.pt")

image_files = sorted(
    [
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
)

if not image_files:
    raise FileNotFoundError(f"No images found in: {IMAGE_DIR}")

print("=" * 70)
print("SCENE-TO-SHOP | PROMPT-FREE OBJECT DISCOVERY")
print("=" * 70)
print(f"Images: {len(image_files)}")
print("Classes supplied by us: NONE")
print("Mode: YOLOE Prompt-Free")
print("=" * 70)

for image_path in image_files:

    print(f"\n{'-' * 70}")
    print(f"IMAGE: {image_path.name}")
    print(f"{'-' * 70}")

    results = model.predict(
        source=str(image_path),
        conf=0.20,
        iou=0.45,
        imgsz=768,
        save=True,
        project=str(OUTPUT_DIR),
        name="results",
        exist_ok=True,
        verbose=True,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
        continue

    names = result.names

    print("\nDISCOVERED OBJECTS:")

    for i, box in enumerate(result.boxes):

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        label = names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(
            f"{i + 1:02d}. "
            f"{label:<25} "
            f"confidence={confidence:.3f} "
            f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

print("\n" + "=" * 70)
print("DONE")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)