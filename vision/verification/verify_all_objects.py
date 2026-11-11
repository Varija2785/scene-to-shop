from pathlib import Path
import json
import os
import re
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

OBJECTS_DIR = (
    BASE_DIR
    / "vision"
    / "detection"
    / "outputs"
    / "objects_clean"
)

OUTPUT_DIR = (
    BASE_DIR
    / "vision"
    / "verification"
    / "outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# GEMINI
# ---------------------------------------------------------

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in the project .env file."
    )

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.8-flash"

# ---------------------------------------------------------
# PROMPT
# ---------------------------------------------------------

PROMPT = """
You are the visual verification component of Scene-to-Shop.

The image is a crop produced by an object detector.
The detector label and filename may be WRONG.

Your job is to determine what physical object is actually visible.

Return ONLY valid JSON with exactly these fields:

{
  "is_physical_object": true,
  "object_name": "",
  "object_category": "",
  "description": "",
  "attributes": [],
  "shopping_relevant": true,
  "confidence": 0.0
}

Rules:

1. Identify the actual physical object visible.
2. Do NOT trust the filename or detector label.
3. Do not invent details that cannot be visually confirmed.
4. Use a specific object name when visually justified.
5. If the image is mostly background, wall, floor, ceiling,
   scene context, or another non-product region:
   set shopping_relevant to false.
6. confidence must be between 0 and 1.
7. Return JSON only.
"""

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def clean_json_response(text: str) -> str:
    """
    Remove Markdown JSON code fences if Gemini adds them.
    """

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def verify_image(image_path: Path):

    print(f"  Uploading: {image_path.name}")

    uploaded_file = client.files.upload(
        file=str(image_path)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[uploaded_file, PROMPT],
        config={
            "temperature": 0
        }
    )

    text = clean_json_response(response.text)

    try:
        result = json.loads(text)

    except json.JSONDecodeError:
        print("  WARNING: Gemini returned invalid JSON.")

        result = {
            "is_physical_object": None,
            "object_name": "",
            "object_category": "",
            "description": "",
            "attributes": [],
            "shopping_relevant": None,
            "confidence": 0.0,
            "raw_response": response.text
        }

    # Keep track of which detector crop produced this result.
    result["source_image"] = image_path.name

    return result


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

print("=" * 70)
print("SCENE-TO-SHOP | BATCH AI OBJECT VERIFICATION")
print("=" * 70)

print(f"Model: {MODEL}")
print(f"Objects directory: {OBJECTS_DIR}")
print()

if not OBJECTS_DIR.exists():
    raise FileNotFoundError(
        f"Object directory not found:\n{OBJECTS_DIR}"
    )

# Only verify color crops, not transparent masks.
image_paths = sorted(
    OBJECTS_DIR.rglob("*_crop.jpg")
)

print(f"Objects found: {len(image_paths)}")
print()

if not image_paths:
    raise RuntimeError("No *_crop.jpg files were found.")

all_results = []

for index, image_path in enumerate(image_paths, start=1):

    print(f"[{index}/{len(image_paths)}]")

    try:
        result = verify_image(image_path)

        all_results.append(result)

        print(
            f"  → {result.get('object_name', 'unknown')} "
            f"| confidence={result.get('confidence', 0)} "
            f"| shopping={result.get('shopping_relevant')}"
        )

    except Exception as e:

        print(f"  ERROR: {e}")

        all_results.append({
            "source_image": image_path.name,
            "is_physical_object": None,
            "object_name": "",
            "object_category": "",
            "description": "",
            "attributes": [],
            "shopping_relevant": None,
            "confidence": 0.0,
            "error": str(e)
        })

    print()

# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

output_file = OUTPUT_DIR / "verified_objects.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False
    )

print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

print(f"Total objects processed: {len(all_results)}")
print(f"Results saved to:")
print(output_file)
print()

shopping_objects = [
    x for x in all_results
    if x.get("shopping_relevant") is True
]

print(
    f"Shopping-relevant objects: "
    f"{len(shopping_objects)}"
)