import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.llm import ConfiguredLLM, LLMResponse  # noqa: E402
from shopping_agent.modes import run_pipeline  # noqa: E402
from shopping_agent.query_understanding import understand_query  # noqa: E402
from shopping_agent.rag import MAX_CONTEXT_CHARS, validate_answer  # noqa: E402
from shopping_agent.retrieval import ProductSearchService  # noqa: E402
from shopping_agent.tracing import summarize_usage  # noqa: E402


class FakeSearchService:
    def __init__(self):
        self.calls = []

    def search(self, query, mode, top_k, filters=None):
        self.calls.append((query, mode, top_k, filters))
        return [{
            "rank": 1, "product_id": "33698", "title": "smart coffee table with storage",
            "product_class": "Coffee & Cocktail Tables", "score": 1.0, "score_type": "BM25",
            "evidence": [{"field": "title", "text": "smart coffee table with storage"}],
            "features": {"Storage": ["drawer"]}, "average_rating": 4.2,
            "rating_count": 17, "review_count": 12,
        }]


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return LLMResponse(self.content, "fake-model", {"total_tokens": 10})


class SequenceLLM:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return LLMResponse(next(self.contents), "fake-model", {"total_tokens": 10})


class PipelineModeTests(unittest.TestCase):
    def test_llm_query_understanding_accepts_grounded_paraphrase(self):
        service = FakeSearchService()
        draft = {
            "search_query": "glass coffee table",
            "product_class": "Coffee & Cocktail Tables",
            "product_class_source": "table for coffee",
            "material": "glass", "material_source": "made of glass",
            "min_historical_rating": 4.0, "rating_source": "at least 4 stars",
            "unsupported_constraints": [],
        }
        llm = FakeLLM(json.dumps(draft))
        result = run_pipeline("a table for coffee made of glass, at least 4 stars",
                              "search_only", "bm25", 5, search_service=service,
                              query_llm=llm, query_understanding_mode="llm", log_path=None)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(service.calls[0][3].product_class, "Coffee & Cocktail Tables")
        self.assertEqual(service.calls[0][3].material, "glass")
        self.assertEqual(result["trace"]["query_understanding_validation"], "accepted")
        self.assertEqual(len(result["trace"]["llm_calls"]), 1)

    def test_invalid_llm_intent_falls_back_to_rules_without_retry(self):
        service = FakeSearchService()
        draft = {
            "search_query": "coffee table",
            "product_class": "Beds", "product_class_source": "coffee table",
            "material": None, "material_source": None,
            "min_historical_rating": 4.0, "rating_source": "at least 4 stars",
            "unsupported_constraints": [],
        }
        llm = FakeLLM(json.dumps(draft))
        result = run_pipeline("coffee table at least 4 stars under $300", "search_only", "bm25", 5,
                              search_service=service, query_llm=llm,
                              query_understanding_mode="llm", log_path=None)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(service.calls[0][3].product_class, "Coffee & Cocktail Tables")
        self.assertEqual(result["trace"]["query_understanding_validation"], "fallback_to_rules")
        self.assertIn("live price or budget", result["query_plan"]["unsupported_constraints"])

    def test_pydantic_rejects_extra_price_filter_and_invalid_rating(self):
        service = FakeSearchService()
        draft = {
            "search_query": "coffee table", "product_class": "Coffee & Cocktail Tables",
            "product_class_source": "coffee table", "material": None,
            "material_source": None, "min_historical_rating": 6.0,
            "rating_source": "at least 4 stars", "unsupported_constraints": [],
            "max_price": 300,
        }
        llm = FakeLLM(json.dumps(draft))
        result = run_pipeline("coffee table at least 4 stars under $300", "search_only", "bm25", 5,
                              search_service=service, query_llm=llm,
                              query_understanding_mode="llm", log_path=None)
        self.assertEqual(result["trace"]["query_understanding_validation"], "fallback_to_rules")
        self.assertEqual(service.calls[0][3].min_historical_rating, 4.0)
        self.assertEqual(result["query_plan"]["unsupported_constraints"], ["live price or budget"])
        self.assertEqual(len(llm.calls), 1)

    def test_llm_search_rewrite_cannot_add_unrequested_attribute(self):
        service = FakeSearchService()
        draft = {
            "search_query": "coffee table waterproof",
            "product_class": "Coffee & Cocktail Tables", "product_class_source": "coffee table",
            "material": None, "material_source": None,
            "min_historical_rating": None, "rating_source": None,
            "unsupported_constraints": [],
        }
        result = run_pipeline("coffee table", "search_only", "bm25", 5,
                              search_service=service, query_llm=FakeLLM(json.dumps(draft)),
                              query_understanding_mode="llm", log_path=None)
        self.assertEqual(result["trace"]["query_understanding_validation"], "fallback_to_rules")
        self.assertEqual(service.calls[0][0], "coffee table")

    def test_llm_understanding_standard_rag_has_two_distinct_llm_calls(self):
        service = FakeSearchService()
        draft = {
            "search_query": "smart coffee table",
            "product_class": "Coffee & Cocktail Tables", "product_class_source": "coffee table",
            "material": None, "material_source": None,
            "min_historical_rating": None, "rating_source": None,
            "unsupported_constraints": [],
        }
        llm = SequenceLLM([json.dumps(draft), json.dumps({"recommendations": [],
                                                          "unknown_requirements": []})])
        result = run_pipeline("smart coffee table", "standard_rag", "bm25", 5,
                              search_service=service, llm=llm,
                              query_understanding_mode="llm", log_path=None)
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual([item["purpose"] for item in result["trace"]["llm_calls"]],
                         ["query_understanding", "answer_generation"])
        self.assertEqual(result["trace"]["agent_tool_calls"], [])

    def test_query_understanding_extracts_only_supported_metadata(self):
        plan = understand_query("solid wood coffee table at least 4 stars under $300 in stock")
        self.assertEqual(plan.search_query, "solid wood coffee table")
        self.assertEqual(plan.filters.product_class, "Coffee & Cocktail Tables")
        self.assertEqual(plan.filters.material, "solid wood")
        self.assertEqual(plan.filters.min_historical_rating, 4.0)
        self.assertEqual(plan.unsupported_constraints, ("live price or budget", "live inventory"))
        self.assertEqual(understand_query("glass coffee table").filters.material, "glass")

    def test_metadata_filters_are_applied_before_bm25_top_k(self):
        def product(product_id, product_class, material, rating, rating_count):
            return {"product_id": product_id, "title": "solid wood coffee table",
                    "product_class": product_class, "category_hierarchy": "Furniture / Tables",
                    "description": "coffee table", "raw_features": "",
                    "features": {"topmaterial": [material]} if material else {},
                    "average_rating": rating, "rating_count": rating_count,
                    "review_count": 1, "search_text": "coffee table " * 3}
        records = [
            product("1", "Coffee & Cocktail Tables", "solid wood", 4.5, 10),
            product("2", "Coffee & Cocktail Tables", "glass", 4.8, 20),
            product("3", "Dining Tables", "solid wood", 4.7, 20),
            product("4", "Coffee & Cocktail Tables", "solid wood", 3.5, 10),
            product("5", "Coffee & Cocktail Tables", "solid wood", 4.9, 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "products.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in records))
            service = ProductSearchService(catalog)
            plan = understand_query("solid wood coffee table at least 4 stars")
            hits = service.search(plan.search_query, mode="bm25", top_k=1, filters=plan.filters)
            with patch.object(service, "_dense_ranking", return_value=[(0, 0.9)]), \
                 patch("shopping_agent.retrieval.cohere_rerank") as reranker:
                hybrid_hits = service.search(plan.search_query, mode="hybrid", top_k=2,
                                             filters=plan.filters, rerank=False)
                reranker.assert_not_called()
                self.assertEqual([hit["product_id"] for hit in hybrid_hits], ["1"])
                service.search("coffee table", mode="bm25", top_k=1, rerank=True,
                               candidate_k=5)
                reranker.assert_called_once()
        self.assertEqual([hit["product_id"] for hit in hits], ["1"])

    def test_parsed_filter_toggle_does_not_enable_reranking(self):
        service = FakeSearchService()
        result = run_pipeline("glass coffee table at least 4 stars", "search_only", "bm25", 5,
                              search_service=service, use_metadata_filter=False,
                              use_reranker=False, log_path=None)
        self.assertTrue(service.calls[0][3].is_empty())
        self.assertFalse(result["trace"]["retrieval_config"]["use_reranker"])
        self.assertEqual(result["trace"]["parsed_filters"]["material"], "glass")

    def test_explicit_id_comparison_uses_one_catalog_batch_read(self):
        records = [
            {"product_id": "1", "title": "first accent chair", "product_class": "Accent Chairs",
             "category_hierarchy": "Furniture", "description": "wood frame",
             "raw_features": "", "features": {"framematerial": ["wood"]},
             "average_rating": 4.5, "rating_count": 2, "review_count": 1,
             "search_text": "first accent chair"},
            {"product_id": "2", "title": "second accent chair", "product_class": "Accent Chairs",
             "category_hierarchy": "Furniture", "description": "metal frame",
             "raw_features": "", "features": {"framematerial": ["metal"]},
             "average_rating": 4.2, "rating_count": 2, "review_count": 1,
             "search_text": "second accent chair"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            service = ProductSearchService(path)
            llm = FakeLLM(json.dumps({"recommendations": [
                {"product_id": "2", "evidence_field": "title",
                 "evidence_quote": "second accent chair"}], "unknown_requirements": []}))
            result = run_pipeline("Compare product ID 1 and ID 2", "standard_rag", "bm25", 5,
                                  search_service=service, llm=llm, log_path=None)
        self.assertEqual(result["trace"]["retrieval_calls"][0]["lookup_type"], "explicit_product_ids")
        self.assertEqual(result["trace"]["retrieval_calls"][0]["retrieved_ids"], ["1", "2"])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result["final_answer"]["recommendations"][0]["product_id"], "2")

    def test_configured_llm_makes_one_tool_free_request(self):
        response = io.BytesIO(json.dumps({
            "model": "test-model",
            "choices": [{"message": {"content": "{}", "tool_calls": None}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 5},
        }).encode())
        with patch("shopping_agent.llm._config", return_value=("test-model", "test-key", "https://api.example.test/v1")), \
             patch("shopping_agent.llm.urlopen", return_value=response) as mocked:
            result = ConfiguredLLM().generate([{"role": "user", "content": "Return JSON"}])
        self.assertEqual(mocked.call_count, 1)
        payload = json.loads(mocked.call_args.args[0].data)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(result.content, "{}")

    def test_search_only_one_retrieval_and_no_llm(self):
        service = FakeSearchService()
        llm = FakeLLM("{}")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.jsonl"
            result = run_pipeline("smart coffee table", "search_only", "bm25", 5,
                                  search_service=service, llm=llm, log_path=path)
            logged = json.loads(path.read_text().strip())
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(result["cards"][0]["product_id"], "33698")
        self.assertEqual(logged["retrieval_calls"][0]["retrieved_ids"], ["33698"])
        self.assertEqual(logged["llm_calls"], [])
        self.assertEqual(logged["agent_tool_calls"], [])

    def test_filtered_standard_rag_still_calls_search_and_llm_once(self):
        service = FakeSearchService()
        llm = FakeLLM(json.dumps({"recommendations": [], "unknown_requirements": []}))
        result = run_pipeline("coffee table at least 4 stars under $300", "standard_rag", "bm25", 5,
                              search_service=service, llm=llm, log_path=None)
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(service.calls[0][0], "coffee table")
        self.assertEqual(service.calls[0][3].min_historical_rating, 4.0)
        self.assertEqual(service.calls[0][3].product_class, "Coffee & Cocktail Tables")
        self.assertIn("live price or budget is unavailable in WANDS",
                      result["final_answer"]["unknown_requirements"])
        self.assertEqual(result["trace"]["agent_tool_calls"], [])

    def test_standard_rag_one_retrieval_one_generation_no_agent_tools(self):
        service = FakeSearchService()
        llm = FakeLLM(json.dumps({
            "recommendations": [
                {"product_id": "99999", "evidence_field": "title", "evidence_quote": "invented"},
                {"product_id": "33698", "evidence_field": "title", "evidence_quote": "smart coffee table with storage"},
            ],
            "unknown_requirements": ["live price"],
        }))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.jsonl"
            result = run_pipeline("smart coffee table", "standard_rag", "bm25", 5,
                                  search_service=service, llm=llm, log_path=path)
            logged = json.loads(path.read_text().strip())
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result["trace"]["agent_tool_calls"], [])
        self.assertEqual([item["product_id"] for item in result["final_answer"]["recommendations"]], ["33698"])
        self.assertLessEqual(result["trace"]["context_chars"], MAX_CONTEXT_CHARS)
        self.assertIn("smart coffee table with storage", logged["prompt_context"])
        self.assertEqual(len(logged["llm_calls"]), 1)
        self.assertIn("99999", " ".join(result["final_answer"]["rejected_model_items"]))

    def test_unsupported_quote_is_not_released(self):
        service = FakeSearchService()
        llm = FakeLLM(json.dumps({
            "recommendations": [{"product_id": "33698", "evidence_field": "title",
                                 "evidence_quote": "delivers tomorrow for $100"}],
            "unknown_requirements": [],
        }))
        result = run_pipeline("smart coffee table", "standard_rag", "bm25", 5,
                              search_service=service, llm=llm, log_path=None)
        self.assertEqual(result["final_answer"]["recommendations"], [])
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(len(llm.calls), 1)

    def test_exact_quote_with_wrong_field_is_corrected_to_observed_field(self):
        records = [{"product_id": "1", "title": "acacia end table",
                    "product_class": "End Tables",
                    "evidence": [{"field": "description", "text": "solid acacia wood top"}],
                    "features": {"topmaterial": ["solid wood"]}}]
        answer = validate_answer(json.dumps({"recommendations": [
            {"product_id": "1", "evidence_field": "features.topmaterialdetails",
             "evidence_quote": "solid acacia wood top"}]}), records)
        self.assertEqual(answer["recommendations"][0]["evidence_field"], "description")
        self.assertEqual(len(answer["source_field_corrections"]), 1)

    def test_quote_in_two_real_feature_fields_remains_verifiable(self):
        records = [{"product_id": "1", "title": "coffee table",
                    "product_class": "Coffee & Cocktail Tables", "evidence": [],
                    "features": {"basematerial": ["solid + manufactured wood"],
                                 "topmaterial": ["solid + manufactured wood"]}}]
        answer = validate_answer(json.dumps({"recommendations": [
            {"product_id": "1", "evidence_field": "basematerial",
             "evidence_quote": "solid + manufactured wood"}]}), records)
        self.assertEqual(answer["recommendations"][0]["evidence_field"], "features.basematerial")

    def test_dimension_request_is_reported_unverified(self):
        plan = understand_query("coffee table under 40 inches")
        self.assertIn("unverified dimensions or units", plan.unsupported_constraints)
        self.assertEqual(plan.search_query, "coffee table")
        answer = validate_answer(json.dumps({
            "recommendations": [{"product_id": "1", "evidence_field": "title",
                                 "evidence_quote": "coffee table"}],
            "unknown_requirements": ["width is unverified"],
        }), [{"product_id": "1", "title": "coffee table", "product_class": "Tables",
              "evidence": [], "features": {}}])
        self.assertEqual(answer["status"], "insufficient_evidence")

    def test_shared_style_uses_observed_feature_fields_across_products(self):
        records = [
            {"product_id": "1", "title": "dining table", "product_class": "Dining Tables",
             "evidence": [], "features": {"dsprimaryproductstyle": ["traditional"]}},
            {"product_id": "2", "title": "dining chair", "product_class": "Dining Chairs",
             "evidence": [], "features": {"style": ["traditional"]}},
            {"product_id": "3", "title": "sofa", "product_class": "Sofas",
             "evidence": [], "features": {"style": ["traditional"]}},
        ]
        answer = validate_answer(json.dumps({"recommendations": [
            {"product_id": record["product_id"], "evidence_field": "title",
             "evidence_quote": record["title"]} for record in records
        ]}), records)
        self.assertEqual(answer["shared_style"]["value"], "traditional")
        self.assertEqual(len(answer["shared_style"]["sources"]), 3)

    def test_partial_llm_usage_preserves_reported_tokens_without_cost(self):
        usage = summarize_usage({"llm_calls": [
            {"usage": {"input_tokens": 100, "output_tokens": 20}},
            {"usage": {}},
        ]}, 50)
        self.assertEqual((usage["input_tokens"], usage["output_tokens"]), (100, 20))
        self.assertFalse(usage["token_usage_complete"])
        self.assertIsNone(usage["estimated_llm_api_cost_usd"])

    def test_final_answer_caps_verified_recommendations_at_three(self):
        records = [
            {"product_id": str(i), "title": f"table {i}", "product_class": "Tables",
             "evidence": [], "features": {}}
            for i in range(1, 5)
        ]
        content = json.dumps({"recommendations": [
            {"product_id": str(i), "evidence_field": "title", "evidence_quote": f"table {i}"}
            for i in range(1, 5)
        ]})
        answer = validate_answer(content, records)
        self.assertEqual([item["product_id"] for item in answer["recommendations"]], ["1", "2", "3"])


if __name__ == "__main__":
    unittest.main()
