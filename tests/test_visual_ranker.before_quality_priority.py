import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIR = ROOT / "vision" / "retrieval" / "search"
sys.path.insert(0, str(SEARCH_DIR))
from visual_ranker import (
    _price_sort_info,
    _product_sort_key,
    listing_quality_info,
)
class VisualRankerTests(unittest.TestCase):
    def test_inr_price_sorting(self):
        products = [
            {"title": "Foreign", "price": {
                "value": "29", "extracted_value": 29, "currency": "EUR"
            }},
            {"title": "INR 290", "price": {
                "value": "290", "extracted_value": 290, "currency": "INR"
            }},
            {"title": "INR 120", "price": {
                "value": "120", "extracted_value": 120, "currency": "INR"
            }},
            {"title": "No price", "price": ""},
        ]
        products.sort(key=_price_sort_info)
        self.assertEqual(
            [p["title"] for p in products],
            ["INR 120", "INR 290", "Foreign", "No price"],
        )
    def test_article_scores_below_commercial_listing(self):
        product = {
            "title": "Everyday item",
            "link": "https://shop.example/products/123",
            "price": {"value": "499", "currency": "INR"},
            "brand": "Example",
        }
        article = {
            "title": "Review: tools for ecommerce photos",
            "link": "https://example.com/blog/review",
            "price": "",
        }
        product_score, _ = listing_quality_info(product)
        article_score, _ = listing_quality_info(article)
        self.assertGreater(product_score, article_score)
    def test_uncertain_listing_is_not_deleted(self):
        products = [
            {
                "title": "Uncertain listing",
                "relevance_score": 0.79,
                "link": "https://example.com/page",
                "price": "",
            },
            {
                "title": "Credible product",
                "relevance_score": 0.80,
                "link": "https://shop.example/products/123",
                "price": {"value": "250", "currency": "INR"},
                "brand": "Example",
            },
        ]
        for product in products:
            product["listing_quality"], product["listing_quality_reasons"] = (
                listing_quality_info(product)
            )
        products.sort(key=_product_sort_key)
        self.assertEqual(len(products), 2)
        self.assertIn("Uncertain listing", [p["title"] for p in products])
    def test_cheaper_price_breaks_close_relevance_tie(self):
        products = [
            {
                "title": "More expensive",
                "relevance_score": 0.82,
                "listing_quality": 0.85,
                "price": {"value": "450", "currency": "INR"},
            },
            {
                "title": "Cheaper",
                "relevance_score": 0.80,
                "listing_quality": 0.85,
                "price": {"value": "250", "currency": "INR"},
            },
        ]
        products.sort(key=_product_sort_key)
        self.assertEqual(products[0]["title"], "Cheaper")
if __name__ == "__main__":
    unittest.main(verbosity=2)
