from pathlib import Path
from dotenv import load_dotenv
from google import genai
import os
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
key = os.getenv("GEMINI_API_KEY")
print("API key found:", bool(key))
client = genai.Client(api_key=key)
try:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Reply with exactly: GEMINI_TEST_OK"
    )
    print("\nSUCCESS")
    print(response.text)
except Exception as e:
    print("\nFAILED")
    print(type(e).__name__)
    print(e)
