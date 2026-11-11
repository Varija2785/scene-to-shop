import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(ROOT / "vision" / "retrieval" / "search")
)
from lens_search import classify_product_link
class ProductLinkTests(unittest.TestCase):
    def test_product_like_path_is_only_a_candidate(self):
        result = classify_product_link(
            "https://shop.example/products/123", "Everyday item"
        )
        self.assertTrue(result["valid_url"])
        self.assertTrue(result["likely_product_page"])
        self.assertIn("unverified", result["reason"])
    def test_article_is_not_classified_as_product(self):
        result = classify_product_link(
            "https://example.com/blog/review", "Review of a product"
        )
        self.assertTrue(result["valid_url"])
        self.assertFalse(result["likely_product_page"])
    def test_invalid_url_is_rejected(self):
        result = classify_product_link("not-a-url", "Item")
        self.assertFalse(result["valid_url"])
        self.assertFalse(result["likely_product_page"])
    def test_general_category_url_is_not_confirmed(self):
        result = classify_product_link(
            "https://shop.example/category/toys", "Toy"
        )
        self.assertTrue(result["valid_url"])
        self.assertFalse(result["likely_product_page"])
if __name__ == "__main__":
    unittest.main(verbosity=2)
