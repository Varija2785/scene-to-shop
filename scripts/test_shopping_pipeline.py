from pathlib import Path
import os
import json
import sys
from dotenv import load_dotenv
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SEARCH_DIR = ROOT / "vision" / "retrieval" / "search"
sys.path.insert(0, str(SEARCH_DIR))
from lens_search import search_google_lens
from visual_ranker import rank_products
image_path = ROOT / "vision" / "selection" / "ui_outputs" / "selected_object.jpg"
api_key = os.getenv("SERPAPI_API_KEY")
print("=" * 60)
print("SCENE TO SHOP - GEMINI-INDEPENDENT SHOPPING TEST")
print("=" * 60)
if not image_path.exists():
    raise SystemExit(f"Selected image not found: {image_path}")
if not api_key:
    raise SystemExit("SERPAPI_API_KEY is missing from the project .env file.")
print(f"Selected image: {image_path.name}")
cache_path = ROOT / "scripts" / "lens_results_cache.json"
use_cache = "--use-cache" in sys.argv
if use_cache:
    if not cache_path.exists():
        raise SystemExit(
            "No saved Lens results yet. Run a live test successfully first."
        )
    products = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(products)} cached Lens results.")
else:
    print("\\nStep 1: Searching Google Lens...")
    result = search_google_lens(image_path, api_key, max_results=12)
    if result.get("error"):
        raise SystemExit(f"Google Lens failed: {result['error']}")
    products = result.get("results", [])
    print(f"Google Lens returned {len(products)} products.")
    if products:
        cache_path.write_text(
            json.dumps(products, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved raw listing metadata to: {cache_path}")
if not products:
    raise SystemExit("No products returned. Check the Lens response.")
print("\\nAvailable fields in the first listing:")
print(", ".join(sorted(products[0].keys())))
print("\nStep 2: Ranking product images with CLIP...")
with Image.open(image_path) as img:
    selected_image = img.convert("RGB")
    ranked = rank_products(selected_image, products, max_products=12)
print("\nTOP RESULTS")
print("-" * 60)
for index, product in enumerate(ranked[:10], start=1):
    title = product.get("title") or "Untitled product"
    source_name = product.get("source") or "Unknown retailer"
    image_score = product.get("visual_similarity")
    title_score = product.get("title_similarity")
    relevance = product.get("relevance_score")
    price_data = product.get("price")
    currency = product.get("currency") or ""
    if isinstance(price_data, dict):
        price = (
            price_data.get("value")
            or price_data.get("extracted_value")
            or price_data.get("raw")
            or ""
        )
        currency = price_data.get("currency") or currency
    else:
        price = price_data
    def show_score(value):
        return f"{value:.4f}" if value is not None else "Unavailable"
    print(f"{index}. {title}")
    print(f"   Retailer: {source_name}")
    print(f"   Image similarity: {show_score(image_score)}")
    print(f"   Title relevance: {show_score(title_score)}")
    print(f"   Combined relevance: {show_score(relevance)}")
    if price:
        print(f"   Price: {currency} {price}".strip())
    print()
print("SHOPPING PIPELINE TEST FINISHED")
