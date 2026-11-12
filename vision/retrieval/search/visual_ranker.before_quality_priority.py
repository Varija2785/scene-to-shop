"""PyTorch-free visual product ranking fallback for Scene-to-Shop.

This module intentionally does not import torch or transformers, so it can run
on Windows systems where PyTorch DLLs are blocked by Application Control.

Ranking note:
- Image similarity is an approximate low-level visual comparison (color + resized pixels),
  not CLIP semantic similarity.
- Title similarity is left as None because meaningful image-to-text semantic scoring
  requires a multimodal model such as CLIP.
- Listing-quality and price sorting heuristics are retained.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
import re
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image


def _image_features(image):
    """Return a normalized descriptor using color histograms and resized RGB pixels."""
    if image is None:
        return None

    rgb = image.convert("RGB")
    # A compact spatial descriptor retains rough layout and dominant colors.
    small = rgb.resize((16, 16), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0

    # Per-channel histograms are less sensitive to small shifts than raw pixels.
    hsv = np.asarray(rgb.convert("HSV").resize((64, 64)), dtype=np.uint8)
    hist_parts = [
        np.histogram(hsv[:, :, channel], bins=16, range=(0, 256), density=True)[0]
        for channel in range(3)
    ]
    histogram = np.concatenate(hist_parts).astype(np.float32)

    # Weight histogram and spatial color descriptor, then normalize.
    descriptor = np.concatenate((histogram * 2.0, pixels))
    norm = float(np.linalg.norm(descriptor))
    if norm > 1e-12:
        descriptor /= norm
    return descriptor


def image_embedding(image):
    """Compatibility helper: return a normalized, PyTorch-free image descriptor."""
    return _image_features(image)


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def download_product_image(url):
    if not url:
        return None
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/154.0.0.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        print(f"Could not download product image: {exc}")
        return None
def _price_sort_info(product):
    """Sort INR prices first, foreign currencies next, and unknown prices last."""
    import math

    price = product.get("price", "")
    currency = str(product.get("currency") or "").strip().upper()
    if isinstance(price, dict):
        currency = str(price.get("currency") or currency).strip().upper()
        value = price.get("extracted_value")
        if value is None:
            value = price.get("value") or price.get("raw") or ""
        raw = str(price.get("raw") or price.get("value") or value or "").strip()
    else:
        value = price
        raw = str(price or "").strip()

    text = f"{currency} {raw}".upper()
    if "₹" in text or any(x in text for x in ("INR", "RS.", "RS ", "RUPEE")):
        currency = "INR"
    elif "€" in text or "EUR" in text:
        currency = "EUR"
    elif "£" in text or "GBP" in text:
        currency = "GBP"
    elif "$" in text or "USD" in text:
        currency = "USD"

    if value in ("", None) and not raw:
        return (9, "UNKNOWN", float("inf"))
    try:
        if isinstance(value, (int, float)):
            amount = float(value)
        else:
            cleaned = re.sub(r"[^0-9,.]", "", str(value or raw))
            if not cleaned:
                return (9, "UNKNOWN", float("inf"))
            if "," in cleaned and "." in cleaned:
                cleaned = cleaned.replace(",", "")
            elif "," in cleaned:
                parts = cleaned.split(",")
                cleaned = "".join(parts) if len(parts[-1]) == 3 else ".".join(parts)
            amount = float(cleaned)
        if not math.isfinite(amount) or amount < 0:
            return (9, "UNKNOWN", float("inf"))
    except (TypeError, ValueError):
        return (9, "UNKNOWN", float("inf"))

    if currency == "INR":
        return (0, "INR", amount)
    if currency in ("EUR", "USD", "GBP"):
        return (1, currency, amount)
    if currency:
        return (2, currency, amount)
    return (8, "UNKNOWN", amount)


def listing_quality_info(product):
    """Heuristic estimate of whether a result looks like a purchasable listing."""
    score = 0.50
    reasons = []
    title = str(product.get("title") or product.get("name") or "").lower()
    link = str(product.get("link") or product.get("url") or "").lower()
    source = str(product.get("source") or "").lower()
    url_path = urlparse(link).path.lower()
    price = product.get("price")

    if isinstance(price, dict):
        has_price = any(
            price.get(key) not in ("", None)
            for key in ("value", "extracted_value", "raw")
        )
    else:
        has_price = price not in ("", None)

    if has_price:
        score += 0.20
        reasons.append("price present")

    metadata_keys = (
        "product_id", "offer", "offers", "availability",
        "brand", "rating", "reviews", "merchant",
    )
    present_metadata = [
        key for key in metadata_keys
        if product.get(key) not in ("", None, [], {})
    ]
    if present_metadata:
        score += min(0.20, 0.05 * len(present_metadata))
        reasons.append("product metadata present")

    if re.search(r"/(product|products|item|p)/", url_path):
        score += 0.10
        reasons.append("product-like URL path")

    non_product_patterns = (
        r"\bhow to\b", r"\btutorial\b", r"\bnews\b",
        r"\bblog\b", r"\barticle\b", r"\bfree download\b",
        r"\b3d models?\b", r"\breview\b", r"\bguide\b",
    )
    page_text = f"{title} {url_path}"
    matched = [pattern for pattern in non_product_patterns if re.search(pattern, page_text)]
    if matched:
        score -= min(0.35, 0.15 * len(matched))
        reasons.append("article, informational, or download-style signal")

    if any(token in source for token in ("pinterest", "instagram", "facebook")):
        score -= 0.10
        reasons.append("social or inspiration source")

    return (max(0.0, min(1.0, score)), reasons)


def _product_sort_key(product):
    """Sort by listing quality, approximate relevance, then price."""
    quality = product.get("listing_quality")
    if quality is None:
        quality, _ = listing_quality_info(product)
    quality_group = 0 if quality >= 0.60 else 1
    score = product.get("relevance_score")
    if score is None:
        return (quality_group, 1, 0, *_price_sort_info(product), 0)
    band = int(score / 0.05)
    return (quality_group, 0, -band, *_price_sort_info(product), -score)


def rank_products(selected_image, products, max_products=12):
    """Rank products by approximate image appearance, listing quality, and price."""
    if not products:
        return []

    candidates = [dict(item) for item in list(products)[:max_products]]
    print(f"Ranking {len(candidates)} product listings (PyTorch-free mode)...")

    query_features = image_embedding(selected_image)
    urls = [item.get("thumbnail", "") for item in candidates]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(urls)))) as executor:
        images = list(executor.map(download_product_image, urls))

    for index, product in enumerate(candidates):
        product_features = image_embedding(images[index]) if images[index] is not None else None
        image_score = (
            cosine_similarity(query_features, product_features)
            if query_features is not None and product_features is not None
            else None
        )
        product["visual_similarity"] = image_score
        # A color/pixel descriptor cannot provide CLIP-style semantic title scoring.
        product["title_similarity"] = None
        product["relevance_score"] = image_score
        quality, reasons = listing_quality_info(product)
        product["listing_quality"] = quality
        product["listing_quality_reasons"] = reasons

    candidates.sort(key=_product_sort_key)
    return candidates
