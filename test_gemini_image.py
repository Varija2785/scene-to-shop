from pathlib import Path
from dotenv import load_dotenv
from google import genai
from PIL import Image
import os
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
image_path = BASE_DIR / "vision" / "selection" / "ui_outputs" / "selected_object.jpg"
print("Image exists:", image_path.exists())
print("Image:", image_path)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
try:
    image = Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            "Identify the main physical object in this image. Return only the object's name.",
            image
        ]
    )
    print("\nSUCCESS")
    print("Gemini identified:", response.text)
except Exception as e:
    print("\nFAILED")
    print(type(e).__name__)
    print(e)
