from pathlib import Path
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)
API_KEY = __import__("os").getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY was not found in .env")
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.8-flash"
OBJECTS_DIR = (
    BASE_DIR
    / "vision"
    / "detection"
    / "outputs"
    / "objects_clean"
)
OUTPUT_DIR = BASE_DIR / "vision" / "verification" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "verified_objects_v2.json"
def extract_json(text):
    text = text.strip()
    # Remove markdown code fences if Gemini adds them.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text}")
    return json.loads(text[start:end + 1])
def verify_image(image_path):
    image_bytes = image_path.read_bytes()
    prompt = """
Look carefully at this image crop.
Identify ONLY what is visibly present in this crop.
Do not assume the object from the filename.
Do not use a predefined object/category vocabulary.
Do not invent details that cannot be seen.
Return ONLY valid JSON with exactly these fields:
{
  "is_real_object": true,
  "object_name": "what the image actually shows",
  "description": "short visual description",
  "shopping_relevant": true,
  "confidence": 0.0
}
Rules:
- is_real_object = false if the crop is mainly background, a scene, an
  architectural region, or an obvious false detection.
- shopping_relevant = true only when the visible object could reasonably
  correspond to something a person might search for or buy.
- object_name should be specific when the visual evidence supports it.
- If the object cannot be identified reliably, use the most accurate
  visually supported description rather than guessing.
- confidence must be between 0 and 1.
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            prompt
        ],
    )
    return extract_json(response.text)
def main():
    print("=" * 70)
    print("SCENE-TO-SHOP | GEMINI VISUAL OBJECT VERIFICATION")
    print("=" * 70)
    if not OBJECTS_DIR.exists():
        raise RuntimeError(
            f"Object directory not found: {OBJECTS_DIR}"
        )
    image_paths = sorted(OBJECTS_DIR.rglob("*_crop.jpg"))
    if not image_paths:
        raise RuntimeError(
            f"No *_crop.jpg files found in {OBJECTS_DIR}"
        )
    print(f"Found {len(image_paths)} object crops.")
    print()
    verified = []
    errors = 0
    for index, image_path in enumerate(image_paths, start=1):
        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.relative_to(OBJECTS_DIR)}"
        )
        try:
            result = verify_image(image_path)
            result["image"] = str(
                image_path.relative_to(BASE_DIR)
            )
            verified.append(result)
            print(
                f"  → {result.get('object_name', 'unknown')} "
                f"| shopping={result.get('shopping_relevant')} "
                f"| confidence={result.get('confidence')}"
            )
        except Exception as exc:
            errors += 1
            print(f"  → ERROR: {exc}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(verified, file, indent=2, ensure_ascii=False)
    shopping_count = sum(
        1
        for item in verified
        if item.get("shopping_relevant") is True
    )
    print()
    print("=" * 70)
    print("GEMINI VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"Processed:          {len(image_paths)}")
    print(f"Verified:           {len(verified)}")
    print(f"Errors:             {errors}")
    print(f"Shopping-relevant:  {shopping_count}")
    print(f"Output:             {OUTPUT_FILE}")
    print("=" * 70)
if __name__ == "__main__":
    main()
