from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from PIL import Image
ROOT = Path.cwd()
load_dotenv(ROOT / ".env")
SEARCH_DIR = ROOT / "vision" / "retrieval" / "search"
sys.path.insert(0, str(SEARCH_DIR))
from lens_search import search_google_lens
from visual_ranker import rank_products, _price_sort_info
image_path = ROOT / "vision" / "selection" / "ui_outputs" / "selected_object.jpg"
api_key = os.getenv("SERPAPI_API_KEY")
result = search_google_lens(image_path, api_key, max_results=12)
if result.get("error"):
    raise SystemExit(result["error"])
with Image.open(image_path) as image:
    ranked = rank_products(image.convert("RGB"), result["results"], max_products=12)
print("\nLIVE PRICE-SORT DIAGNOSTICS")
for item in ranked:
    print("-" * 50)
    print("Title:", item.get("title", "")[:85])
    print("Price field:", repr(item.get("price")))
    print("Currency field:", repr(item.get("currency")))
    print("Sort key:", _price_sort_info(item))
print("\nCURRENT RESULT ORDER")
for index, item in enumerate(ranked, 1):
    print(index, item.get("title", "")[:65], repr(item.get("price")), repr(item.get("currency")))
