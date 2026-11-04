import json
import math
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.evaluation import consolidate_labels, query_metrics  # noqa: E402
from shopping_agent.retrieval import ProductSearchService, rrf_fuse  # noqa: E402
from shopping_agent.rerank import cohere_rerank  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_metric_definitions(self):
        ndcg, rr = query_metrics(
            ["Partial", "Exact", "Irrelevant"],
            ["Partial", "Exact", "Irrelevant"], 3,
        )
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg, expected)
        self.assertEqual(rr, 0.5)
        self.assertEqual(query_metrics(["Partial"], ["Partial"], 10), (1.0, None))
        self.assertEqual(query_metrics(["Irrelevant"], ["Irrelevant"], 10), (None, None))

    def test_conflicts_are_excluded_and_identical_duplicates_collapsed(self):
        labels = pd.DataFrame([
            {"query_id": 1, "product_id": 10, "label": "Exact"},
            {"query_id": 1, "product_id": 10, "label": "Exact"},
            {"query_id": 1, "product_id": 11, "label": "Exact"},
            {"query_id": 1, "product_id": 11, "label": "Partial"},
        ])
        qrels, stats = consolidate_labels(labels)
        self.assertEqual(qrels, {1: {"10": "Exact"}})
        self.assertEqual(stats["conflicting_pairs_excluded"], 1)
        self.assertEqual(stats["duplicate_label_rows"], 2)

    def test_candidate_ranking_keeps_zero_score_products(self):
        rows = [
            {"product_id": "1", "search_text": "coffee table"},
            {"product_id": "2", "search_text": "blue sofa"},
            {"product_id": "3", "search_text": "coffee desk"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            service = ProductSearchService(path)
            ranked = service.rank_candidates("coffee", ["1", "2", "3"])
            self.assertEqual([product_id for product_id, _ in ranked], ["1", "3", "2"])
            self.assertEqual(ranked[-1][1], 0.0)

    def test_rrf_rewards_a_product_found_by_both_retrievers(self):
        scores = rrf_fuse([[1, 2], [2, 3]], k=60)
        self.assertAlmostEqual(scores[2], 1 / 62 + 1 / 61)
        self.assertGreater(scores[2], scores[1])

    def test_cohere_rerank_maps_response_indexes_to_input_order(self):
        response = io.BytesIO(json.dumps({"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.3},
        ]}).encode())
        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}), \
             patch("shopping_agent.rerank.urlopen", return_value=response) as mocked:
            ranked = cohere_rerank("chair", ["sofa", "chair"], 2)
        self.assertEqual(ranked, [(1, 0.9), (0, 0.3)])
        self.assertEqual(mocked.call_args.args[0].get_header("Authorization"), "Bearer test-key")


if __name__ == "__main__":
    unittest.main()
