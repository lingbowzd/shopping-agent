import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_retrievers import evaluate_cohere_with_checkpoint  # noqa: E402


class FakeSearchService:
    def __init__(self):
        self.calls = []

    def rank_candidates(self, query, product_ids, mode):
        self.calls.append((query, mode))
        return [(product_ids[0], 0.9), (product_ids[1], 0.3)]


class CohereCheckpointTests(unittest.TestCase):
    def test_resume_skips_recorded_queries(self):
        queries = pd.DataFrame([
            {"query_id": 1, "query": "coffee table"},
            {"query_id": 2, "query": "desk"},
        ])
        labels = pd.DataFrame([
            {"query_id": query_id, "product_id": product_id, "label": label}
            for query_id in (1, 2)
            for product_id, label in ((11, "Exact"), (12, "Partial"))
        ])
        service = FakeSearchService()
        expected = {"model": "test", "k": 2, "query_ids": [1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            first = evaluate_cohere_with_checkpoint(service, queries, labels, 2, checkpoint, expected, {})
            self.assertEqual(len(service.calls), 2)
            self.assertEqual(first["mrr_at_k"], 1.0)
            self.assertEqual(len(json.loads(checkpoint.read_text())["per_query"]), 2)
            second = evaluate_cohere_with_checkpoint(service, queries, labels, 2, checkpoint, expected, {})
            self.assertEqual(len(service.calls), 2)
            self.assertEqual(second["per_query"], first["per_query"])


if __name__ == "__main__":
    unittest.main()
