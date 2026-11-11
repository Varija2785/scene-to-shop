from pathlib import Path
import json
import os

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# SCENE-TO-SHOP
# AI OBJECT VERIFICATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

# Load API key from project-root .env
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in the project .env file."
    )

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.8-flash"


# Test ONE object first
IMAGE_PATH = (
    BASE_DIR
    / "vision"
    / "detection"
    / "outputs"
    / "objects_clean"
    / "test3"
    / "02_lamp_0.87_crop.jpg"
)


PROMPT = """
You are the visual understanding component of Scene-to-Shop.

Look carefully at the provided object image.

The filename and any previous detector label may be WRONG.
Do not trust them.

Determine what physical object is actually shown.

Return ONLY valid JSON using exactly these fields:

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

1. Identify the actual object visible in the image.
2. Do not rely on the filename.
3. Do not invent details that cannot be seen.
4. Give a specific object name when visually justified.
5. If this is merely background, a scene, wall, floor,
   or another non-product region, set shopping_relevant to false.
6. confidence must be between 0 and 1.
"""


print("=" * 65)
print("SCENE-TO-SHOP | AI OBJECT VERIFICATION")
print("=" * 65)

print(f"Model: {MODEL}")
print(f"Image: {IMAGE_PATH.name}")
print()

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

print("Uploading object image...")

uploaded_file = client.files.upload(
    file=str(IMAGE_PATH)
)

print("Image uploaded.")
print("Asking Gemini to identify the object...")
print()

response = client.models.generate_content(
    model=MODEL,
    contents=[uploaded_file, PROMPT],
    config={
        "temperature": 0
    }
)

text = response.text.strip()

# Remove Markdown code fences if Gemini wraps the JSON in ```json ... ```
if text.startswith("```"):
    lines = text.splitlines()

    if lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    text = "\n".join(lines).strip()

print("RAW RESPONSE:")
print(text)
print()

try:
    result = json.loads(text)

    print("PARSED RESULT:")
    print(json.dumps(result, indent=2))

except json.JSONDecodeError:
    print("WARNING: Response was not valid JSON.")