import io
import json
import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR / "vision" / "retrieval" / "search"))
from lens_search import search_google_lens, classify_product_link  # noqa: E402
from visual_ranker import rank_products  # noqa: E402
app = FastAPI(title="Scene-to-Shop API")
@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/lens")
async def lens(image: UploadFile = File(...), max_results: int = Form(12)):
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise HTTPException(500, "SERPAPI_API_KEY is not set on the backend")
    data = await image.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        return search_google_lens(tmp, api_key, max_results=max_results)
    finally:
        os.remove(tmp)
@app.post("/rank")
async def rank(
    image: UploadFile = File(...),
    products_json: str = Form(...),
    max_products: int = Form(16),
):
    pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
    return rank_products(pil, json.loads(products_json), max_products=max_products)
@app.post("/classify")
def classify(url: str = Form(...), title: str = Form(""), source: str = Form("")):
    return {"result": classify_product_link(url, title, source)}
