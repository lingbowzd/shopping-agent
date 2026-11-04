import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.data import load_wands, parse_features, prepare  # noqa: E402
from shopping_agent.retrieval import ProductSearchService  # noqa: E402


class Stage1ATests(unittest.TestCase):
    def test_feature_parser_preserves_duplicate_keys_and_malformed_segments(self):
        raw = "color:blue|color:red|broken|:bad|size: 12: inches|empty: "
        features, malformed = parse_features(raw)
        self.assertEqual(features, {"color": ["blue", "red"], "size": ["12: inches"]})
        self.assertEqual(malformed, 3)
        self.assertEqual(parse_features(None), ({}, 0))

    def test_real_data_reproducibility_and_search(self):
        dataset = ROOT / "data/raw/WANDS/dataset"
        if not dataset.exists():
            self.skipTest("Clone official WANDS into data/raw/WANDS first")
        products, queries, labels = load_wands(dataset)
        self.assertIn("category hierarchy", products.columns)
        self.assertEqual(len(queries), 480)
        self.assertEqual(len(labels), 233448)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_manifest = prepare(dataset, Path(first))
            second_manifest = prepare(dataset, Path(second))
            self.assertEqual(first_manifest["selected_product_ids_sha256"], second_manifest["selected_product_ids_sha256"])
            self.assertEqual(first_manifest["sampled_count"], 3000)
            self.assertEqual((Path(first) / "products.jsonl").read_bytes(), (Path(second) / "products.jsonl").read_bytes())
            service = ProductSearchService(Path(first) / "products.jsonl")
            stages = {}
            hits = service.search("modern coffee table", top_k=3, stage_trace=stages)
            self.assertEqual(len(hits), 3)
            self.assertEqual(stages["bm25_ids"][:3], [hit["product_id"] for hit in hits])
            self.assertEqual(stages["eligible_count"], 3000)
            ids = {json.loads(line)["product_id"] for line in (Path(first) / "products.jsonl").read_text().splitlines()}
            self.assertTrue(all(hit["product_id"] in ids for hit in hits))
            self.assertTrue(all(item["text"] for hit in hits for item in hit["evidence"]))

    def test_active_full_catalog_contains_every_source_product_id(self):
        dataset = ROOT / "data/raw/WANDS/dataset"
        output = ROOT / "data/processed"
        if not dataset.exists() or not (output / "products.jsonl").exists():
            self.skipTest("Prepare the official full WANDS catalog first")
        products, _, _ = load_wands(dataset)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["selection_mode"], "full")
        with (output / "products.jsonl").open() as stream:
            actual = {int(json.loads(line)["product_id"]) for line in stream}
        self.assertEqual(actual, set(products.product_id))
        self.assertEqual(len(actual), 42994)


if __name__ == "__main__":
    unittest.main()
