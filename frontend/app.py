import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
import os
import json
import time
import sys
import html
import re
import math
import hashlib
from urllib.parse import urlparse

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Scene-to-Shop",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = BASE_DIR / "vision" / "selection" / "ui_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

RETRIEVAL_DIR = BASE_DIR / "vision" / "retrieval" / "search"
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from lens_search import search_google_lens, classify_product_link
from visual_ranker import rank_products

# Primary model can be changed in .env with GEMINI_MODEL=...
# If it times out or is not found, the next models are tried automatically.
DEFAULT_MODEL = "gemini-3.8-flash"
MODEL = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
# Optional: GEMINI_FALLBACK_MODELS=model-a,model-b in .env (empty by default)
_fallbacks = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
# If .env points to a retired model, the current default is tried automatically.
MODEL_CHAIN = list(dict.fromkeys([MODEL] + _fallbacks + [DEFAULT_MODEL]))

# ==================================================
# SESSION STATE
# ==================================================
DEFAULTS = {
    "wishlist": [],
    "compare_products": [],
    "comparison_folders": {},
    "active_page": "Shop",
    "canvas_reset_counter": 0,
    "scene_search_results": None,
    "scene_search_error": None,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def clear_analysis():
    for key in (
        "object_name",
        "description",
        "confidence",
        "shopping_relevant",
        "analysis_error",
    ):
        st.session_state.pop(key, None)


def reset_selection():
    for key in (
        "selected_object",
        "object_name",
        "description",
        "confidence",
        "shopping_relevant",
        "analysis_error",
        "scene_search_results",
        "scene_search_error",
    ):
        st.session_state.pop(key, None)
    st.session_state.pop("compare_message", None)
    # Changing the canvas key clears any old rectangle without touching the uploaded file.
    st.session_state["canvas_reset_counter"] = st.session_state.get("canvas_reset_counter", 0) + 1


def product_id(product):
    identity = (
        product.get("link")
        or product.get("title")
        or repr(product)
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def short_token(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:10]


# ==================================================
# GEMINI OBJECT ANALYSIS
# ==================================================
def prepare_for_ai(image, max_side=1024):
    """Large phone photos make Gemini very slow. Send a smaller copy."""
    small = image.copy()
    small.thumbnail((max_side, max_side))
    return small


def analyze_object(image):
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY is not configured in .env"}
    try:
        image = prepare_for_ai(image)
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=60000),  # 60 seconds per call
        )
        prompt = """
Look carefully at the selected image.
Identify what is actually visible.
Do not use a predefined object/category list.
Do not assume the object from the filename.
Do not invent details that cannot be seen.
Return ONLY valid JSON:
{
  "object_name": "specific natural name",
  "description": "short visible description",
  "shopping_relevant": true,
  "confidence": 0.0
}
Rules:
\\- Describe only what can actually be seen.
\\- Use a specific natural name when supported.
\\- If unclear, use "Unclear object".
\\- Confidence must be between 0 and 1.
\\- shopping_relevant should be true if the object
  can reasonably be searched for or purchased.
"""
        response = None
        last_exc = None
        first_exc = None
        for model_name in MODEL_CHAIN:
            for attempt in range(4):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                        ),
                    )
                    print(f"[Scene-to-Shop] Gemini model used: {model_name}")
                    break
                except Exception as exc:
                    first_exc = first_exc or exc
                    last_exc = exc
                    msg = str(exc).lower()
                    print(f"[Scene-to-Shop] {model_name} attempt {attempt + 1} failed: {exc}")
                    busy = any(
                        k in msg
                        for k in ("503", "unavailable", "overloaded", "high demand")
                    )
                    next_model = any(
                        k in msg
                        for k in ("404", "not_found", "not found",
                                  "timed out", "timeout", "deadline")
                    )
                    if busy and attempt < 3:
                        time.sleep(2 ** (attempt + 1))  # wait 2s, 4s, 8s then retry
                        continue
                    if busy or next_model:
                        break  # give up on this model, try the next one
                    raise
            if response is not None:
                break
        if response is None and first_exc is not None:
            raise first_exc  # show the PRIMARY model's error, not a fallback's
        if response is None or not response.text:
            raise RuntimeError("Gemini did not return a response.")
        text = response.text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("Gemini returned an unexpected response.")
        return result
    except Exception as exc:
        error_text = str(exc)
        if "503" in error_text or "UNAVAILABLE" in error_text:
            return {"error": "Gemini is busy (high demand). It was retried 4 times. Wait a minute and click Retry."}
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            return {"error": "Gemini API quota is exhausted. Please try again later."}
        if "timed out" in error_text.lower() or "timeout" in error_text.lower() or "deadline" in error_text.lower():
            return {"error": "Gemini did not respond within 90 seconds. Check internet / API key and click Retry."}
        return {"error": error_text}


def save_analysis(result):
    st.session_state["object_name"] = result.get("object_name", "Unclear object")
    st.session_state["description"] = result.get("description", "")
    st.session_state["confidence"] = float(result.get("confidence", 0))
    st.session_state["shopping_relevant"] = bool(result.get("shopping_relevant", False))
    st.session_state.pop("analysis_error", None)


# ==================================================
# WISHLIST AND COMPARE ACTIONS
# ==================================================
def toggle_wishlist(product):
    current = list(st.session_state["wishlist"])
    pid = product_id(product)
    existing_ids = [product_id(item) for item in current]
    if pid in existing_ids:
        st.session_state["wishlist"] = [
            item for item in current if product_id(item) != pid
        ]
    else:
        st.session_state["wishlist"] = current + [product]


def comparison_folder_name():
    """Automatically group products under the currently analysed object."""
    name = str(st.session_state.get("object_name") or "Products").strip()
    name = name.title() if name else "Products"
    return f"{name} Comparison"


def sync_compare_list():
    folders = st.session_state.get("comparison_folders", {})
    st.session_state["compare_products"] = [
        item for group in folders.values() for item in group
    ]


def toggle_compare(product):
    """Add/remove a product in its automatically created comparison folder."""
    folders = dict(st.session_state.get("comparison_folders", {}))
    folder_name = comparison_folder_name()
    items = list(folders.get(folder_name, []))
    pid = product_id(product)
    existing_ids = [product_id(item) for item in items]

    if pid in existing_ids:
        folders[folder_name] = [item for item in items if product_id(item) != pid]
        if not folders[folder_name]:
            folders.pop(folder_name, None)
    else:
        if len(items) >= 4:
            st.session_state["compare_message"] = (
                "You can compare up to four products in each folder."
            )
            return
        folders[folder_name] = items + [product]
        st.session_state.pop("compare_message", None)

    st.session_state["comparison_folders"] = folders
    sync_compare_list()


def remove_from_comparison(folder_name, product):
    folders = dict(st.session_state.get("comparison_folders", {}))
    remaining = [
        item for item in folders.get(folder_name, [])
        if product_id(item) != product_id(product)
    ]
    if remaining:
        folders[folder_name] = remaining
    else:
        folders.pop(folder_name, None)
    st.session_state["comparison_folders"] = folders
    sync_compare_list()


def source_app_name(product):
    """Return the supplied source name, or infer a shopping site from its URL."""
    source = str(product.get("source") or "").strip()
    link = str(product.get("link") or "").strip()
    host = (urlparse(link).hostname or "").lower()
    host = host.removeprefix("www.")

    generic_sources = {"", "google", "google lens", "shopping", "web result", "unknown"}
    if source and source.casefold() not in generic_sources:
        return source

    known_sites = (
        ("amazon.", "Amazon"),
        ("flipkart.com", "Flipkart"),
        ("myntra.com", "Myntra"),
        ("meesho.com", "Meesho"),
        ("ajio.com", "AJIO"),
        ("tatacliq.com", "Tata CLiQ"),
        ("nykaa.com", "Nykaa"),
        ("ebay.", "eBay"),
        ("etsy.com", "Etsy"),
        ("walmart.com", "Walmart"),
        ("target.com", "Target"),
        ("aliexpress.com", "AliExpress"),
        ("temu.com", "Temu"),
    )
    for domain, app_name in known_sites:
        if domain in host:
            return app_name

    if host:
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host
    return source or "Source not available"


def display_price(product):
    price = product.get("price", "")
    currency = product.get("currency", "")
    if isinstance(price, dict):
        currency = price.get("currency") or currency
        value = (
            price.get("value")
            or price.get("extracted_value")
            or price.get("raw")
            or ""
        )
    else:
        value = str(price or "")
    if not value:
        return "Price not available"
    return f"{currency} {value}".strip()


def has_price(product):
    return display_price(product) != "Price not available"


def search_google_shopping(query, api_key, max_results=12):
    """Google Shopping results carry real prices (Lens visual matches often do not)."""
    query = (query or "").strip()
    if not query or query.casefold() == "unclear object":
        return []
    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_shopping",
                "q": query,
                "gl": "in",
                "hl": "en",
                "api_key": api_key,
            },
            timeout=25,
        )
        data = response.json()
    except Exception:
        return []
    items = []
    for r in (data.get("shopping_results") or [])[:max_results]:
        items.append(
            {
                "title": r.get("title"),
                "link": r.get("link") or r.get("product_link"),
                "source": r.get("source"),
                "price": r.get("price") or r.get("extracted_price") or "",
                "thumbnail": r.get("thumbnail"),
            }
        )
    return items


def merge_products(*lists):
    seen, merged = set(), []
    for items in lists:
        for item in items or []:
            ident = (item.get("link") or item.get("title") or "").strip().casefold()
            if not ident or ident in seen:
                continue
            seen.add(ident)
            merged.append(item)
    return merged


# ==================================================
# THEME: PREMIUM DARK GREEN + CREAM
# Responsive product grid: 4 columns on desktop,
# 3 on tablet, 2 on mobile.
# ==================================================
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --bg: #0b1f18;
        --bg-2: #0f2a21;
        --panel: #14352a;
        --panel-2: #194135;
        --line: rgba(245,239,224,.16);
        --cream: #f5efe0;
        --cream-soft: #d9d2bf;
        --muted: #a9b8ab;
        --accent: #c9b88a;
        --green: #0f2a21;
    }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }

    /* Menu button that opens / closes the left panel */
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] button {
        background: var(--cream) !important; color: var(--green) !important;
        border-radius: 999px !important; padding: 6px 14px !important;
        font-weight: 700 !important; gap: 6px;
    }
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="collapsedControl"] button * { color: var(--green) !important; }
    [data-testid="stExpandSidebarButton"]::after,
    [data-testid="collapsedControl"] button::after {
        content: "Menu"; font-size: .85rem; color: var(--green);
    }
    [data-testid="stSidebarCollapseButton"] button {
        background: rgba(245,239,224,.12) !important; border-radius: 999px !important;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(201,184,138,.10), transparent 35%),
            radial-gradient(ellipse at 92% 10%, rgba(74,140,108,.18), transparent 32%),
            var(--bg);
        color: var(--cream);
    }
    [data-testid="stHeader"] { background: rgba(11,31,24,.88); }
    [data-testid="stSidebar"] {
        background: var(--bg-2);
        border-right: 1px solid var(--line);
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4 { color: var(--cream) !important; letter-spacing: -.02em; }
    .stMarkdown p, label, [data-testid="stCaptionContainer"] { color: var(--cream-soft); }
    hr { border-color: var(--line); }

    /* ---------- Hero ---------- */
    .hero {
        padding: 28px 28px 24px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(25,65,53,.85), rgba(15,42,33,.85));
        margin-bottom: 1.2rem;
    }
    .hero .eyebrow {
        letter-spacing: .18em; font-size: .72rem; font-weight: 700;
        color: var(--accent); text-transform: uppercase;
    }
    .hero h1 {
        margin: .35rem 0 .3rem; font-size: clamp(1.7rem, 4vw, 2.8rem);
        line-height: 1.1; color: var(--cream) !important;
    }
    .hero p { margin: 0; color: var(--cream-soft); font-size: clamp(.95rem, 2vw, 1.1rem); }

    .step-head { display: flex; align-items: center; gap: 12px; margin: 1.4rem 0 .6rem; }
    .step-num {
        width: 34px; height: 34px; border-radius: 50%;
        background: var(--cream); color: var(--green);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; flex: 0 0 34px;
    }
    .step-title { font-size: 1.25rem; font-weight: 700; color: var(--cream); line-height: 1.2; }
    .step-sub { font-size: .88rem; color: var(--muted); }

    /* ---------- Uploader / alerts ---------- */
    [data-testid="stFileUploader"] {
        background: rgba(20,53,42,.7);
        border: 1px dashed rgba(201,184,138,.6);
        border-radius: 16px; padding: 12px;
    }
    [data-testid="stAlert"] {
        background: rgba(20,53,42,.75) !important;
        border: 1px solid var(--line); border-radius: 12px;
    }
    [data-testid="stAlert"] p { color: var(--cream) !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 16px !important;
        background: rgba(20,53,42,.55);
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--line) !important; border-radius: 14px !important;
        background: rgba(20,53,42,.35);
    }

    /* ---------- Buttons ---------- */
    .stButton > button, [data-testid="stLinkButton"] a {
        border-radius: 10px !important; font-weight: 600 !important; min-height: 40px;
        border: 1px solid rgba(245,239,224,.35) !important;
        background: transparent !important; color: var(--cream) !important;
    }
    .stButton > button p, [data-testid="stLinkButton"] a p { color: inherit !important; }
    .stButton > button:hover, [data-testid="stLinkButton"] a:hover {
        border-color: var(--cream) !important; background: rgba(245,239,224,.08) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--cream) !important; color: var(--green) !important; border: 0 !important;
    }
    .stButton > button[kind="primary"] p { color: var(--green) !important; }
    .stButton > button[kind="primary"]:hover { background: #fff8e8 !important; }
    [data-testid="stLinkButton"] a {
        background: var(--cream) !important; color: var(--green) !important; border: 0 !important;
    }
    [data-testid="stLinkButton"] a:hover { background: #fff8e8 !important; }

    /* ---------- Product cards ---------- */
    .p-img-empty {
        width: 100%; aspect-ratio: 1/1; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        background: var(--panel-2); color: var(--muted); font-size: .85rem;
    }
    [class*="st-key-grid_"] [data-testid="stImage"] img {
        width: 100% !important; aspect-ratio: 1 / 1; object-fit: cover;
        border-radius: 12px; background: #fff;
    }
    .p-title {
        font-weight: 700; color: var(--cream); font-size: .95rem; line-height: 1.3;
        height: 2.6em; overflow: hidden;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    }
    .p-source {
        display: inline-block; margin-top: 6px; padding: 2px 10px; border-radius: 999px;
        border: 1px solid var(--line); color: var(--accent);
        font-size: .75rem; font-weight: 600;
    }
    .p-price { margin-top: 6px; font-size: 1.15rem; font-weight: 800; color: var(--cream); }
    .p-match {
        display: inline-block; margin: 4px 0 8px; padding: 2px 10px; border-radius: 8px;
        background: rgba(201,184,138,.16); color: var(--accent);
        font-size: .78rem; font-weight: 700;
    }
    .p-note { font-size: .72rem; color: var(--muted); margin-top: 4px; }

    .p-badge {
        display: inline-block; margin: 0 6px 6px 0; padding: 2px 10px; border-radius: 8px;
        background: var(--cream); color: var(--green); font-size: .72rem; font-weight: 800;
    }
    .cmp-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; margin: .4rem 0 .6rem; }
    .cmp-table { border-collapse: collapse; width: 100%; min-width: 560px; }
    .cmp-table th, .cmp-table td {
        padding: 10px 12px; border-bottom: 1px solid var(--line);
        vertical-align: top; text-align: left; color: var(--cream); font-size: .9rem; min-width: 150px;
    }
    .cmp-table th.rowhead {
        color: var(--accent); font-size: .74rem; text-transform: uppercase; letter-spacing: .08em;
        min-width: 90px; position: sticky; left: 0; background: #0f2a21;
    }
    .cmp-table img {
        width: 130px; height: 130px; object-fit: cover; border-radius: 10px;
        display: block; margin-bottom: 8px; background: #fff;
    }
    .cmp-table td.best { background: rgba(201,184,138,.18); font-weight: 800; }
    .cmp-table a { color: var(--accent); font-weight: 700; }

    /* ---------- Responsive grid (one wrapping flex row) ---------- */
    [class*="st-key-grid_"] [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important; row-gap: 1rem !important; column-gap: 1rem !important;
    }
    [class*="st-key-grid_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 0 0 calc(25% - 0.75rem) !important;
        width: calc(25% - 0.75rem) !important;
        min-width: 0 !important;
    }
    @media (max-width: 1099px) {
        [class*="st-key-grid_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 0 0 calc(33.333% - 0.667rem) !important;
            width: calc(33.333% - 0.667rem) !important;
        }
    }
    @media (max-width: 767px) {
        [data-testid="stAppViewContainer"] .main .block-container { padding: 1rem .7rem 2rem; }
        .hero { padding: 20px 16px; }
        [class*="st-key-grid_"] [data-testid="stHorizontalBlock"] {
            column-gap: .6rem !important; row-gap: .6rem !important;
        }
        [class*="st-key-grid_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 0 0 calc(50% - 0.3rem) !important;
            width: calc(50% - 0.3rem) !important;
        }
        .p-title { font-size: .85rem; }
        .p-price { font-size: 1rem; }
    }

    /* Small round icons on the image corners (like Amazon / AJIO).
       Each icon is its own absolutely positioned container (no columns). */
    [class*="st-key-media_"] { position: relative; gap: 0 !important; }
    [class*="st-key-media_"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
    [class*="st-key-actl_"], [class*="st-key-actr_"] {
        position: absolute !important; bottom: 10px; z-index: 5;
        width: auto !important; height: auto !important;
    }
    [class*="st-key-actl_"] { left: 10px; }
    [class*="st-key-actr_"] { right: 10px; }
    [class*="st-key-actl_"] .stButton, [class*="st-key-actr_"] .stButton { width: auto !important; }

    [class*="st-key-cmp_"] button,
    [class*="st-key-wish_"] button,
    [class*="st-key-rmw_"] button,
    [class*="st-key-rmc_"] button {
        width: 38px !important; min-width: 38px !important;
        height: 38px !important; min-height: 38px !important;
        padding: 0 !important; border-radius: 50% !important;
        background: #ffffff !important; border: 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,.45) !important;
        display: flex !important; align-items: center; justify-content: center;
    }
    [class*="st-key-cmp_"] button:hover,
    [class*="st-key-wish_"] button:hover,
    [class*="st-key-rmw_"] button:hover,
    [class*="st-key-rmc_"] button:hover { background: #f5efe0 !important; }
    [class*="st-key-cmp_"] button, [class*="st-key-cmp_"] button *,
    [class*="st-key-wish_"] button, [class*="st-key-wish_"] button *,
    [class*="st-key-rmw_"] button, [class*="st-key-rmw_"] button *,
    [class*="st-key-rmc_"] button, [class*="st-key-rmc_"] button * {
        color: #0f2a21 !important; -webkit-text-fill-color: #0f2a21 !important;
        font-size: 1.2rem !important; font-weight: 700 !important; line-height: 1 !important;
        margin: 0 !important;
    }
    [class*="st-key-wish_on_"] button, [class*="st-key-wish_on_"] button * {
        color: #e0245e !important; -webkit-text-fill-color: #e0245e !important;
    }
    [class*="st-key-cmp_on_"] button, [class*="st-key-cmp_on_"] button * {
        color: #16a34a !important; -webkit-text-fill-color: #16a34a !important;
    }
    .p-badges { min-height: 32px; margin-top: 4px; }

    /* Generic columns (AI understanding etc.) stack on mobile */
    @media (max-width: 767px) {
        .stApp [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# UI HELPERS
# ==================================================
def step_header(number, title, subtitle=""):
    sub = f'<div class="step-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="step-head">
            <div class="step-num">{number}</div>
            <div><div class="step-title">{html.escape(title)}</div>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def visual_match_percent(product):
    value = product.get("visual_similarity")
    if value is None:
        return None
    try:
        return int(round(max(0.0, min(1.0, float(value))) * 100))
    except (TypeError, ValueError):
        return None


def render_product_card(product, ctx, index, mode, folder_name=None, badges=None):
    """
    mode: "shop" (compare + heart icons on the image), "wishlist" / "compare" (remove icon)
    """
    pid = product_id(product)
    title = product.get("title") or "Untitled product"
    source = source_app_name(product)
    thumbnail = product.get("thumbnail") or ""
    link = product.get("link") or ""
    key_base = f"{short_token(ctx)}_{index}_{pid}"

    with st.container(border=True):
        # 1) Image with small round icons on its bottom corners
        with st.container(key=f"media_{key_base}"):
            if thumbnail:
                try:
                    st.image(thumbnail, width="stretch")
                except Exception:
                    st.markdown('<div class="p-img-empty">Image unavailable</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="p-img-empty">No image</div>', unsafe_allow_html=True)

            if mode == "shop":
                wishlist_ids = [product_id(i) for i in st.session_state["wishlist"]]
                is_wishlisted = pid in wishlist_ids
                folder_items = st.session_state.get("comparison_folders", {}).get(
                    comparison_folder_name(), []
                )
                is_compared = pid in [product_id(i) for i in folder_items]
                with st.container(key=f"actl_{key_base}"):
                    if st.button(
                        "⇄",
                        key=f"cmp_{'on' if is_compared else 'off'}_{key_base}",
                        help=(
                            f"Remove from {comparison_folder_name()}"
                            if is_compared
                            else f"Add to {comparison_folder_name()}"
                        ),
                    ):
                        toggle_compare(product)
                        st.rerun()
                with st.container(key=f"actr_{key_base}"):
                    if st.button(
                        "♥" if is_wishlisted else "♡",
                        key=f"wish_{'on' if is_wishlisted else 'off'}_{key_base}",
                        help="Remove from Wishlist" if is_wishlisted else "Add to Wishlist",
                    ):
                        toggle_wishlist(product)
                        st.rerun()
            elif mode == "wishlist":
                with st.container(key=f"actr_{key_base}"):
                    if st.button("✕", key=f"rmw_{key_base}", help="Remove from Wishlist"):
                        toggle_wishlist(product)
                        st.rerun()
            elif mode == "compare":
                with st.container(key=f"actr_{key_base}"):
                    if st.button("✕", key=f"rmc_{key_base}", help="Remove from comparison"):
                        remove_from_comparison(folder_name, product)
                        st.rerun()

        # 2) Info
        match = visual_match_percent(product)
        match_html = f'<div class="p-match">Visual match: {match}%</div>' if match is not None else ""
        badges_html = "".join(
            f'<span class="p-badge">{html.escape(b)}</span>' for b in (badges or [])
        )
        st.markdown(
            f"""
            <div class="p-title" title="{html.escape(title)}">{html.escape(title)}</div>
            <span class="p-source">{html.escape(source)}</span>
            <div class="p-price">{html.escape(display_price(product))}</div>
            <div class="p-badges">{badges_html}</div>{match_html}
            """,
            unsafe_allow_html=True,
        )

        # 3) View product
        if link:
            link_info = classify_product_link(link, title=title, source=source)
            if not link_info.get("valid_url"):
                st.caption("No valid destination link available.")
            elif link_info.get("likely_product_page"):
                st.link_button("View Product", link, width="stretch")
            else:
                st.link_button("View Result", link, width="stretch")
                st.markdown(
                    '<div class="p-note">May be a general page, not the exact product.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No link available.")


def render_product_grid(items, ctx, mode, folder_name=None, badges=None):
    """One wrapping flex row: 4 cols desktop, 3 tablet, 2 mobile (see CSS)."""
    if not items:
        return
    with st.container(key=f"grid_{short_token(ctx)}"):
        cols = st.columns(len(items), gap="medium")
        for index, product in enumerate(items):
            with cols[index]:
                render_product_card(
                    product, ctx, index, mode, folder_name,
                    badges=(badges or {}).get(product_id(product)),
                )


# ==================================================
# PRICE TOOLS: sort, badges, summary, comparison table
# ==================================================
SORT_BEST = "Best visual match"
SORT_ASC = "Price: Low to High"
SORT_DESC = "Price: High to Low"


def price_value(product):
    """Numeric price, or None when the product has no usable price."""
    price = product.get("price")
    if isinstance(price, dict):
        value = price.get("extracted_value")
        if isinstance(value, (int, float)):
            return float(value)
        price = price.get("value") or price.get("raw") or ""
    if isinstance(price, (int, float)) and not isinstance(price, bool):
        return float(price)
    match = re.search(r"\d[\d,]*\.?\d*", str(price or ""))
    if match:
        try:
            return float(match.group().replace(",", ""))
        except ValueError:
            return None
    return None


def sort_products(items, choice):
    items = list(items)
    if choice not in (SORT_ASC, SORT_DESC):
        return items
    priced = [p for p in items if price_value(p) is not None]
    unpriced = [p for p in items if price_value(p) is None]
    priced.sort(key=price_value, reverse=(choice == SORT_DESC))
    return priced + unpriced


def compute_badges(items):
    """pid -> list of badge texts. Only shown when there is something to compare."""
    badges = {}
    priced = [(price_value(p), p) for p in items if price_value(p) is not None]
    if len(priced) >= 2:
        low = min(v for v, _ in priced)
        high = max(v for v, _ in priced)
        if low < high:
            for v, p in priced:
                if v == low:
                    badges.setdefault(product_id(p), []).append("★ Lowest price")
    matches = [(visual_match_percent(p), p) for p in items if visual_match_percent(p) is not None]
    if len(matches) >= 2:
        best = max(m for m, _ in matches)
        if best > min(m for m, _ in matches):
            for m, p in matches:
                if m == best:
                    badges.setdefault(product_id(p), []).append("◎ Best match")
    return badges


def currency_symbol(product):
    return re.match(r"^\D*", display_price(product)).group().strip()


def show_price_summary(items):
    priced = [p for p in items if price_value(p) is not None]
    if len(priced) >= 2:
        low = min(priced, key=price_value)
        high = max(priced, key=price_value)
        st.caption(
            f"Lowest: {display_price(low)} ({source_app_name(low)}) · "
            f"Highest: {display_price(high)} ({source_app_name(high)})"
        )


def nav_label(page):
    if page == "Wishlist":
        return f"Wishlist ({len(st.session_state['wishlist'])})"
    if page == "Comparison":
        return f"Comparison ({len(st.session_state['comparison_folders'])})"
    return page


def go_to(page):
    st.session_state["active_page"] = page


def render_comparison_table(items):
    prices = [price_value(p) for p in items if price_value(p) is not None]
    lowest = min(prices) if len(prices) >= 2 and min(prices) < max(prices) else None
    matches = [visual_match_percent(p) for p in items if visual_match_percent(p) is not None]
    best_match = max(matches) if len(matches) >= 2 and max(matches) > min(matches) else None

    product_cells, price_cells, source_cells, match_cells, link_cells = [], [], [], [], []
    for p in items:
        title = html.escape(p.get("title") or "Untitled product")
        thumb = p.get("thumbnail") or ""
        img = f'<img src="{html.escape(thumb, quote=True)}" alt="">' if thumb.startswith("http") else ""
        product_cells.append(f"<td>{img}{title}</td>")

        value = price_value(p)
        is_low = lowest is not None and value == lowest
        label = html.escape(display_price(p)) + (" ★ Lowest" if is_low else "")
        price_cells.append(f'<td class="best">{label}</td>' if is_low else f"<td>{label}</td>")

        source_cells.append(f"<td>{html.escape(source_app_name(p))}</td>")

        m = visual_match_percent(p)
        if m is None:
            match_cells.append("<td>—</td>")
        elif best_match is not None and m == best_match:
            match_cells.append(f'<td class="best">{m}% ◎ Best</td>')
        else:
            match_cells.append(f"<td>{m}%</td>")

        link = p.get("link") or ""
        if link.startswith(("http://", "https://")):
            link_cells.append(f'<td><a href="{html.escape(link, quote=True)}" target="_blank" rel="noopener">View</a></td>')
        else:
            link_cells.append("<td>—</td>")

    rows = [
        ("Product", product_cells),
        ("Price", price_cells),
        ("Source", source_cells),
        ("Visual match", match_cells),
        ("Link", link_cells),
    ]
    body = "".join(
        f'<tr><th class="rowhead">{label}</th>{"".join(cells)}</tr>' for label, cells in rows
    )
    st.markdown(
        f'<div class="cmp-wrap"><table class="cmp-table"><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    if lowest is not None:
        highest = max(prices)
        cheapest = next(p for p in items if price_value(p) == lowest)
        diff = highest - lowest
        st.caption(
            f"Cheapest option saves {currency_symbol(cheapest)}{diff:,.0f} "
            f"({diff / highest * 100:.0f}%) compared with the most expensive."
        )


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.markdown("## 🛍️ Scene-to-Shop")
    st.caption("AI-powered visual shopping")
    nav_choice = st.radio(
        "Navigate",
        ["Shop", "Wishlist", "Comparison"],
        key="active_page",
        label_visibility="collapsed",
        format_func=nav_label,
    )
    st.divider()
    st.caption("**Shop** · Upload, select and search")
    st.caption("**Wishlist** · Saved products")
    st.caption("**Comparison** · Folders, up to 4 products each")

# ==================================================
# WISHLIST PAGE
# ==================================================
if nav_choice == "Wishlist":
    st.button("← Back to Shop", on_click=go_to, args=("Shop",), key="back_from_wishlist")
    st.title(f"♥ Wishlist ({len(st.session_state['wishlist'])})")
    if not st.session_state["wishlist"]:
        st.info("Your Wishlist is empty. Add products using the heart button on a product card.")
    else:
        sort_choice = st.selectbox(
            "Sort by", ["Recently added", SORT_ASC, SORT_DESC], key="wishlist_sort"
        )
        wish_items = sort_products(st.session_state["wishlist"], sort_choice)
        show_price_summary(wish_items)
        render_product_grid(wish_items, "wishlist_page", "wishlist", badges=compute_badges(wish_items))
        if st.button("Clear wishlist", key="clear_wishlist"):
            st.session_state["wishlist"] = []
            st.rerun()
    st.stop()

# ==================================================
# COMPARISON PAGE
# ==================================================
if nav_choice == "Comparison":
    st.button("← Back to Shop", on_click=go_to, args=("Shop",), key="back_from_comparison")
    folders = st.session_state.get("comparison_folders", {})
    st.title(f"⇄ Comparison Folders ({len(folders)})")
    if not folders:
        st.info("No comparison folders yet. Click the compare button on products to create a folder automatically.")
    for folder_name, items in list(folders.items()):
        with st.expander(f"📁 {folder_name} · {len(items)} products", expanded=True):
            render_product_grid(
                list(items), f"folder_{folder_name}", "compare", folder_name,
                badges=compute_badges(items),
            )
            if len(items) >= 2:
                st.markdown("#### Side-by-side comparison")
                render_comparison_table(items)

            folder_token = short_token(folder_name)
            if st.button(f"Delete folder: {folder_name}", key=f"delete_folder_{folder_token}"):
                st.session_state["confirm_delete_folder"] = folder_name
                st.rerun()

            if st.session_state.get("confirm_delete_folder") == folder_name:
                st.warning(f"Delete '{folder_name}' and all products inside it?")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Yes, delete folder", key=f"confirm_delete_{folder_token}"):
                        all_folders = dict(st.session_state["comparison_folders"])
                        all_folders.pop(folder_name, None)
                        st.session_state["comparison_folders"] = all_folders
                        sync_compare_list()
                        st.session_state.pop("confirm_delete_folder", None)
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"cancel_delete_{folder_token}"):
                        st.session_state.pop("confirm_delete_folder", None)
                        st.rerun()
    st.stop()

# ==================================================
# SHOP PAGE: HERO
# ==================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered visual shopping</div>
        <h1>Find products from your photo</h1>
        <p>Upload a scene, select an object, discover similar products.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

quick1, quick2, _spacer = st.columns([1, 1, 2], gap="small")
with quick1:
    st.button(
        f"♥ Wishlist ({len(st.session_state['wishlist'])})",
        on_click=go_to, args=("Wishlist",), key="top_wishlist", width="stretch",
    )
with quick2:
    st.button(
        f"⇄ Comparison ({len(st.session_state['comparison_folders'])})",
        on_click=go_to, args=("Comparison",), key="top_comparison", width="stretch",
    )

# ==================================================
# STEP 1: UPLOAD
# ==================================================
step_header(1, "Upload your room or product image", "JPG · PNG · WEBP")
uploaded_file = st.file_uploader(
    "Choose image",
    type=["jpg", "jpeg", "png", "webp"],
    key="scene_upload",
    label_visibility="collapsed",
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((1600, 1600))  # keeps the app fast with big phone photos

    # ==============================================
    # STEP 2: SELECT OBJECT
    # ==============================================
    step_header(2, "Select an object", "Draw a rectangle around the object")

    size_choice = st.radio(
        "Selection area size (choose Small on phones)",
        ["Small", "Medium", "Large"],
        index=2,
        horizontal=True,
        key="canvas_size_choice",
    )
    max_width = {"Small": 340, "Medium": 560, "Large": 900}[size_choice]
    canvas_width = min(max_width, image.width)
    scale = canvas_width / image.width
    canvas_height = int(image.height * scale)
    display_image = image.resize((canvas_width, canvas_height))

    canvas_result = st_canvas(
        fill_color="rgba(201, 184, 138, 0.25)",
        stroke_width=4,
        stroke_color="#f5efe0",
        background_image=display_image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key=f"object_selector_{st.session_state['canvas_reset_counter']}_{size_choice}",
    )

    button_col1, button_col2 = st.columns(2, gap="medium")
    with button_col1:
        understand_clicked = st.button("Analyze with AI →", type="primary", width="stretch")
    with button_col2:
        retry_selection_clicked = st.button("↩ Retry Selection", width="stretch")
    st.caption("AI identifies the object before searching for visually similar products.")

    if retry_selection_clicked:
        reset_selection()
        st.rerun()

    if understand_clicked:
        objects = (
            canvas_result.json_data.get("objects", [])
            if canvas_result.json_data
            else []
        )
        if not objects:
            st.warning("Please draw a rectangle around an object first.")
        else:
            rect = objects[-1]
            left = rect.get("left", 0)
            top = rect.get("top", 0)
            rect_width = rect.get("width", 0)
            rect_height = rect.get("height", 0)
            if rect_width < 15 or rect_height < 15:
                st.warning("The selected area is too small.")
            else:
                scale_x = image.width / canvas_width
                scale_y = image.height / canvas_height
                x1 = max(0, int(left * scale_x))
                y1 = max(0, int(top * scale_y))
                x2 = min(image.width, int((left + rect_width) * scale_x))
                y2 = min(image.height, int((top + rect_height) * scale_y))
                padding_x = int((x2 - x1) * 0.15)
                padding_y = int((y2 - y1) * 0.15)
                crop_x1 = max(0, x1 - padding_x)
                crop_y1 = max(0, y1 - padding_y)
                crop_x2 = min(image.width, x2 + padding_x)
                crop_y2 = min(image.height, y2 + padding_y)
                selected = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                output_path = OUTPUT_DIR / "selected_object.jpg"
                selected.save(output_path, quality=95)
                st.session_state["selected_object"] = selected
                clear_analysis()
                st.session_state["scene_search_results"] = None
                st.session_state["scene_search_error"] = None
                with st.spinner("AI is understanding the selected object..."):
                    result = analyze_object(selected)
                if "error" in result:
                    st.session_state["analysis_error"] = result["error"]
                else:
                    save_analysis(result)

# ==================================================
# STEP 3: AI UNDERSTANDING
# ==================================================
if "selected_object" in st.session_state:
    step_header(3, "AI Understanding", "What the AI sees in your selection")
    selected_object = st.session_state["selected_object"]
    image_col, details_col = st.columns([1, 1], gap="large")

    with image_col:
        st.image(selected_object, caption="Selected object", width="stretch")

    with details_col:
        if "analysis_error" in st.session_state:
            st.warning("AI understanding is temporarily unavailable.")
            st.write(st.session_state["analysis_error"])
            st.caption(f"Model configured: {MODEL} (set by GEMINI_MODEL in .env, if present)")
            if st.button("🔄 Retry AI Analysis", type="primary", width="stretch"):
                clear_analysis()
                with st.spinner("Trying AI analysis again..."):
                    result = analyze_object(selected_object)
                if "error" in result:
                    st.session_state["analysis_error"] = result["error"]
                else:
                    save_analysis(result)
                st.rerun()
        elif "object_name" in st.session_state:
            object_name = st.session_state["object_name"]
            description = st.session_state.get("description", "")
            confidence = st.session_state.get("confidence", 0.0)
            shopping_relevant = st.session_state.get("shopping_relevant", False)

            st.success(f"**{object_name}**")
            if description:
                st.write(description)
            st.write(f"AI confidence: **{confidence * 100:.1f}%**")

            if shopping_relevant:
                st.success("Suitable for visual shopping")
                if st.button("🛍️ Find Similar Products", type="primary", width="stretch"):
                    if not SERPAPI_API_KEY:
                        st.session_state["scene_search_error"] = (
                            "SERPAPI_API_KEY is not configured in .env"
                        )
                        st.session_state["scene_search_results"] = None
                    else:
                        with st.spinner("Searching for visually similar products..."):
                            search_result = search_google_lens(
                                OUTPUT_DIR / "selected_object.jpg",
                                SERPAPI_API_KEY,
                                max_results=12,
                            )
                            if not search_result.get("error"):
                                lens_products = search_result.get("results", [])
                                shop_products = search_google_shopping(
                                    st.session_state.get("object_name", ""),
                                    SERPAPI_API_KEY,
                                )
                                products = merge_products(shop_products, lens_products)
                                if products:
                                    products = rank_products(
                                        selected_object,
                                        products,
                                        max_products=16,
                                    )
                                    search_result["results"] = products
                        if search_result.get("error"):
                            st.session_state["scene_search_error"] = search_result["error"]
                            st.session_state["scene_search_results"] = None
                        else:
                            st.session_state["scene_search_results"] = search_result.get("results", [])
                            st.session_state["scene_search_error"] = None
                    st.rerun()
            else:
                st.warning("This does not appear to be shopping-relevant.")
        else:
            st.info("Click 'Analyze with AI →' to analyze your selection.")

# ==================================================
# STEP 4: SIMILAR PRODUCTS (full width, responsive grid)
# ==================================================
if "selected_object" in st.session_state:
    search_error = st.session_state.get("scene_search_error")
    if search_error:
        st.error(search_error)

    products = st.session_state.get("scene_search_results")
    if products:
        header_slot = st.container()

        f1, f2 = st.columns(2, gap="medium")
        with f1:
            only_priced = st.toggle(
                "Only show products with a price",
                value=True,
                key="only_priced",
            )
        with f2:
            sort_choice = st.selectbox(
                "Sort by", [SORT_BEST, SORT_ASC, SORT_DESC], key="shop_sort"
            )

        shown = [p for p in products if has_price(p)] if only_priced else list(products)

        price_values = sorted({price_value(p) for p in shown if price_value(p) is not None})
        if len(price_values) >= 2:
            low_bound, high_bound = int(price_values[0]), math.ceil(price_values[-1])
            price_range = st.slider(
                "Price range",
                min_value=low_bound,
                max_value=high_bound,
                value=(low_bound, high_bound),
                key=f"price_range_{low_bound}_{high_bound}",
            )
            shown = [
                p for p in shown
                if price_value(p) is None or price_range[0] <= price_value(p) <= price_range[1]
            ]

        shown = sort_products(shown, sort_choice)

        with header_slot:
            step_header(4, "Similar Products", f"{len(shown)} of {len(products)} results")

        compare_message = st.session_state.get("compare_message")
        if compare_message:
            st.warning(compare_message)

        if shown:
            show_price_summary(shown)
            render_product_grid(shown, "shop_results", "shop", badges=compute_badges(shown))
        else:
            st.info("No results match these filters. Turn off the price switch or widen the price range.")
    elif products == []:
        st.info("No visual product matches were found.")

    st.markdown("---")
    if st.button("🆕 Select Another Object", width="stretch"):
        reset_selection()
        st.rerun()
else:
    st.info("Upload an image above to start.")