from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[2]

file_path = (
    BASE_DIR
    / "vision"
    / "verification"
    / "outputs"
    / "verified_objects.json"
)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("SCENE-TO-SHOP | VERIFIED SHOPPING OBJECTS")
print("=" * 70)

count = 0

for item in data:
    if item.get("shopping_relevant") is True:
        count += 1

        print(
            f"{count}. "
            f"{item.get('object_name', 'Unknown')} "
            f"| {item.get('object_category', 'Unknown')} "
            f"| confidence={item.get('confidence', 0)}"
        )

print()
print(f"Total shopping-relevant objects: {count}")