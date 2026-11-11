from pathlib import Path
import os
import json
import requests

from dotenv import load_dotenv


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

IMAGE_PATH = (
    BASE_DIR
    / "vision"
    / "detection"
    / "outputs"
    / "objects_clean"
    / "test3"
    / "02_lamp_0.87_crop.jpg"
)

OUTPUT_DIR = (
    BASE_DIR
    / "vision"
    / "retrieval"
    / "outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# API KEY
# ---------------------------------------------------------

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("SERPAPI_API_KEY")

if not api_key:
    raise RuntimeError(
        "SERPAPI_API_KEY was not found in .env"
    )


# ---------------------------------------------------------
# IMAGE CHECK
# ---------------------------------------------------------

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

image_size = IMAGE_PATH.stat().st_size

print("=" * 70)
print("SCENE-TO-SHOP | GOOGLE LENS PRODUCT SEARCH")
print("=" * 70)

print()
print(f"Image: {IMAGE_PATH.name}")
print(f"Image size: {image_size / 1024:.1f} KB")
print()

if image_size > 500 * 1024:
    raise RuntimeError(
        "Image is larger than SerpApi's 500 KB upload limit."
    )


# ---------------------------------------------------------
# STEP 1 — UPLOAD IMAGE
# ---------------------------------------------------------

print("Step 1/2: Uploading image to SerpApi Image API...")

with open(IMAGE_PATH, "rb") as image_file:

    upload_response = requests.post(
        "https://serpapi.com/image",
        params={
            "api_key": api_key
        },
        files={
            "image": (
                IMAGE_PATH.name,
                image_file,
                "image/jpeg"
            )
        },
        timeout=60
    )


print(
    f"Upload HTTP status: "
    f"{upload_response.status_code}"
)

if not upload_response.ok:
    print(upload_response.text)
    raise RuntimeError(
        "SerpApi image upload failed."
    )

upload_result = upload_response.json()

if "error" in upload_result:
    raise RuntimeError(
        f"SerpApi upload error: "
        f"{upload_result['error']}"
    )

image_id = upload_result.get("image_id")

if not image_id:
    raise RuntimeError(
        f"SerpApi did not return image_id:\n"
        f"{upload_result}"
    )

print("Image uploaded successfully.")
print("Temporary image ID received.")
print()


# ---------------------------------------------------------
# STEP 2 — GOOGLE LENS
# ---------------------------------------------------------

print("Step 2/2: Searching Google Lens products...")

search_response = requests.get(
    "https://serpapi.com/search.json",
    params={
        "engine": "google_lens",
        "image_id": image_id,
        "type": "products",
        "hl": "en",
        "country": "in",
        "api_key": api_key
    },
    timeout=60
)

print(
    f"Search HTTP status: "
    f"{search_response.status_code}"
)

if not search_response.ok:
    print(search_response.text)
    raise RuntimeError(
        "Google Lens search failed."
    )

results = search_response.json()


# ---------------------------------------------------------
# SAVE RAW RESPONSE
# ---------------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "google_lens_test.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print()
print("=" * 70)
print("SEARCH COMPLETE")
print("=" * 70)

if "error" in results:

    print()
    print("Google Lens returned an error:")
    print(results["error"])

else:

    products = results.get(
        "visual_matches",
        []
    )

    print()
    print(
        f"Visual/product matches returned: "
        f"{len(products)}"
    )
    print()

    for index, product in enumerate(
        products[:10],
        start=1
    ):

        print(
            f"{index}. "
            f"{product.get('title', 'Unknown product')}"
        )

        print(
            f"   Source: "
            f"{product.get('source', 'N/A')}"
        )

        print(
            f"   Price: "
            f"{product.get('price', 'N/A')}"
        )

        print(
            f"   Link: "
            f"{product.get('link', 'N/A')}"
        )

        print()

print("Raw results saved to:")
print(output_file)