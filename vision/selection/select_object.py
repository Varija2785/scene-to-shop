import cv2
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = BASE_DIR / "vision" / "detection" / "test_images"
OUTPUT_DIR = BASE_DIR / "vision" / "selection" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
image_path = IMAGE_DIR / "test.jpg"
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
print("Image loaded successfully.")
print("Draw a rectangle around the object you want to shop for.")
print("Press ENTER after selecting.")
print("Press C to cancel.")
x, y, w, h = cv2.selectROI(
    "SCENE TO SHOP - Select Object",
    image,
    showCrosshair=True,
    fromCenter=False
)
cv2.destroyAllWindows()
if w == 0 or h == 0:
    print("No object selected.")
    raise SystemExit
crop = image[y:y+h, x:x+w]
output_path = OUTPUT_DIR / "selected_object.jpg"
cv2.imwrite(str(output_path), crop)
print()
print("========================================")
print("OBJECT SELECTED SUCCESSFULLY")
print("========================================")
print(f"Coordinates : x={x}, y={y}, width={w}, height={h}")
print(f"Saved crop  : {output_path}")
print("========================================")
