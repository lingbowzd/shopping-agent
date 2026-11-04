import json
import sys
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.agent import AgentBudgets, run_agentic_rag  # noqa: E402
from shopping_agent.llm import LLMResponse  # noqa: E402
from shopping_agent.modes import run_pipeline  # noqa: E402
from shopping_agent.query_understanding import understand_query  # noqa: E402
from shopping_agent.router import route_query  # noqa: E402
from shopping_agent.schemas import RetrievalConfig  # noqa: E402


ROUTE_CASES = [
    ("Show me turquoise throw pillows.", "search_only"),
    ("Find a wooden coffee table under 40 inches.", "search_only"),
    ("I need a 24-inch bar stool with a backrest.", "search_only"),
    ("Show me desks and office chairs.", "search_only"),
    ("Which sofa ships tomorrow for under $300?", "search_only"),
    ("Find a glass coffee table rated at least 4 stars.", "search_only"),
    ("What is a good coffee table for a small apartment?", "standard_rag"),
    ("Which sofa suits a small room?", "standard_rag"),
    ("Which of these three accent chairs suits a small room?", "standard_rag"),
    ("Compare two end tables by material and rating.", "standard_rag"),
    ("Recommend a reading chair for my bedroom.", "standard_rag"),
    ("Which one is better for a narrow entryway?", "standard_rag"),
    ("I'd like a coffee table with matching end tables.", "agentic_rag"),
    ("I'm furnishing an apartment with a coffee table and end tables.", "agentic_rag"),
    ("Find a desk, chair and storage that work together.", "agentic_rag"),
    ("Help me set up a desk, chair and storage for my bedroom.", "agentic_rag"),
    ("Can you coordinate a sofa and rug for my living room?", "agentic_rag"),
    ("Help me find a set of dining table, chairs and sofa and the style matches", "agentic_rag"),
]


class FixtureSearch:
    def __init__(self, color):
        self.color = color
        self.queries = []
        self.filters = []

    def search(self, query, mode, top_k, filters=None):
        self.queries.append(query)
        self.filters.append(filters)
        if "coffee table" in query:
            return [{"product_id": "1", "title": "coffee table", "product_class": "Coffee & Cocktail Tables",
                     "features": {"basecolor": [self.color]}, "evidence": [
                         {"field": "title", "text": "coffee table"}],
                     "average_rating": None, "rating_count": None}]
        return [{"product_id": "2", "title": "end table", "product_class": "End Tables",
                 "features": {"basecolor": [self.color]}, "evidence": [
                     {"field": "title", "text": "end table"}],
                 "average_rating": None, "rating_count": None}]


class ConditionalModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        tool_messages = [msg for msg in messages if msg.type == "tool"]
        if not tool_messages:
            return AIMessage(content="", tool_calls=[{
                "name": "search_products", "args": {"query": "coffee table", "top_k": 2},
                "id": "call-1",
            }])
        if len(tool_messages) == 1:
            first = json.loads(tool_messages[-1].content)["products"][0]
            color = first["features"]["basecolor"][0]
            return AIMessage(content="", tool_calls=[{
                "name": "search_products", "args": {"query": f"{color} end table", "top_k": 2},
                "id": "call-2",
            }])
        return AIMessage(content=json.dumps({"recommendations": [
            {"product_id": "1", "evidence_field": "title", "evidence_quote": "coffee table"},
            {"product_id": "2", "evidence_field": "title", "evidence_quote": "end table"},
        ], "unknown_requirements": []}))


class TitleFinalLLM:
    def __init__(self, product_ids):
        self.product_ids = product_ids

    def generate(self, messages):
        content = messages[-1]["content"]
        context = json.loads(content.split("Source context JSON: ", 1)[1].split("\n\n", 1)[0])
        by_id = {hit["product_id"]: hit for hit in context["products"]}
        recommendations = [
            {"product_id": product_id, "evidence_field": "title",
             "evidence_quote": by_id[product_id]["title"]}
            for product_id in self.product_ids
        ]
        return LLMResponse(json.dumps({"recommendations": recommendations,
                                       "unknown_requirements": []}),
                           "fake-final", {"prompt_tokens": 1, "completion_tokens": 1})


class RouterAgentTests(unittest.TestCase):
    def test_human_curated_routes(self):
        for query, expected in ROUTE_CASES:
            with self.subTest(query=query):
                self.assertEqual(route_query(query).route, expected)

    def test_multiple_categories_do_not_become_one_hard_category_filter(self):
        for query in (
            "Help me find a set of dining table, chairs and sofa and the style matches",
            "Show me desks and office chairs.",
        ):
            with self.subTest(query=query):
                self.assertIsNone(understand_query(query).filters.product_class)

    def test_counterfactual_first_result_changes_second_search(self):
        followups = []
        for color in ("walnut", "white metal"):
            service = FixtureSearch(color)
            result = run_agentic_rag(
                "Help me find a coffee table and matching end tables",
                service, RetrievalConfig(backend="bm25", top_k=2),
                model=ConditionalModel(), final_llm=TitleFinalLLM(["1", "2"]),
            )
            self.assertEqual([item["product_id"] for item in result["final_answer"]["recommendations"]],
                             ["1", "2"])
            self.assertEqual(result["trace"]["budget_usage"]["search_calls"], 2)
            self.assertEqual(result["trace"]["agent_tool_calls"][1]["observed_cue_from_first_search"]["value"],
                             color)
            followups.append(service.queries[1])
        self.assertEqual(followups, ["walnut end table", "white metal end table"])

    def test_one_search_budget_stops_without_second_tool(self):
        service = FixtureSearch("walnut")

        class FinalLLM:
            def generate(self, messages):
                from shopping_agent.llm import LLMResponse
                return LLMResponse(json.dumps({"recommendations": [], "unknown_requirements": []}),
                                   "fake", {"prompt_tokens": 1, "completion_tokens": 1})

        result = run_agentic_rag(
            "Help me find a coffee table and matching end tables",
            service, RetrievalConfig(backend="bm25", top_k=2), model=ConditionalModel(),
            final_llm=FinalLLM(), budgets=AgentBudgets(max_llm_calls=3, max_tool_calls=2,
                                                       max_search_calls=1),
        )
        self.assertEqual(result["trace"]["budget_usage"]["search_calls"], 1)
        self.assertEqual(len(service.queries), 1)

    def test_forced_agent_preserves_explicit_hard_filters(self):
        class OneSearchModel:
            def bind_tools(self, tools, **kwargs):
                return self

            def invoke(self, messages):
                if not any(message.type == "tool" for message in messages):
                    return AIMessage(content="", tool_calls=[{
                        "name": "search_products", "args": {"query": "coffee table", "top_k": 2},
                        "id": "call-1",
                    }])
                return AIMessage(content=json.dumps({"recommendations": [],
                                                     "unknown_requirements": []}))

        service = FixtureSearch("walnut")
        result = run_pipeline(
            "solid wood coffee table at least 4 stars", "agentic_rag", "bm25", 2,
            search_service=service, agent_model=OneSearchModel(), log_path=None,
        )
        self.assertEqual(service.filters[0].product_class, "Coffee & Cocktail Tables")
        self.assertEqual(service.filters[0].material, "solid wood")
        self.assertEqual(service.filters[0].min_historical_rating, 4.0)
        self.assertEqual(result["trace"]["parsed_filters"]["material"], "solid wood")

    def test_coordinated_three_category_goal_continues_after_premature_answer(self):
        class ThreeCategorySearch:
            def __init__(self):
                self.queries = []

            def search(self, query, mode, top_k, filters=None):
                self.queries.append(query)
                if "sofa" in query:
                    product_id, title, product_class = "3", "walnut sofa", "Sofas"
                elif "chair" in query:
                    product_id, title, product_class = "2", "walnut dining chair", "Dining Chairs"
                else:
                    product_id, title, product_class = "1", "walnut dining table", "Dining Tables"
                return [{"product_id": product_id, "title": title,
                         "product_class": product_class,
                         "features": {"basecolor": ["walnut"]},
                         "evidence": [{"field": "title", "text": title}],
                         "average_rating": None, "rating_count": None}]

        class PrematureModel:
            def bind_tools(self, tools, **kwargs):
                return self

            def invoke(self, messages):
                tool_count = sum(message.type == "tool" for message in messages)
                if tool_count == 0:
                    search_query = "dining table"
                elif tool_count == 1 and messages[-1].type == "tool":
                    return AIMessage(content=json.dumps({"recommendations": [
                        {"product_id": "1", "evidence_field": "title",
                         "evidence_quote": "walnut dining table"}]}))
                elif tool_count == 1:
                    search_query = "walnut dining chair"
                elif tool_count == 2:
                    search_query = "walnut sofa"
                else:
                    return AIMessage(content=json.dumps({"recommendations": [
                        {"product_id": product_id, "evidence_field": "title",
                         "evidence_quote": title}
                        for product_id, title in (
                            ("1", "walnut dining table"), ("2", "walnut dining chair"),
                            ("3", "walnut sofa"))]}))
                return AIMessage(content="", tool_calls=[{
                    "name": "search_products", "args": {"query": search_query, "top_k": 3},
                    "id": f"call-{tool_count}-{search_query}",
                }])

        service = ThreeCategorySearch()
        result = run_pipeline(
            "Help me find a set of dining table, chairs and sofa and the style matches",
            "auto", "bm25", 3, search_service=service,
            agent_model=PrematureModel(), llm=TitleFinalLLM(["1", "2", "3"]),
            log_path=None,
        )
        self.assertEqual(result["mode"], "agentic_rag")
        self.assertEqual(result["query_plan"]["filters"]["product_class"], None)
        self.assertEqual(service.queries, ["dining table", "walnut dining chair", "walnut sofa"])
        self.assertEqual([item["product_id"] for item in result["final_answer"]["recommendations"]],
                         ["1", "2", "3"])
        self.assertEqual(result["trace"]["budget_usage"]["search_calls"], 3)

        class EmptyFinalLLM:
            model = "fake-empty"

            def generate(self, messages):
                raise ValueError("LLM response has no text content (finish_reason=length)")

        fallback = run_pipeline(
            "Help me find a set of dining table, chairs and sofa and the style matches",
            "auto", "bm25", 3, search_service=ThreeCategorySearch(),
            agent_model=PrematureModel(), llm=EmptyFinalLLM(), log_path=None,
        )
        self.assertEqual([item["product_id"] for item in fallback["final_answer"]["recommendations"]],
                         ["1", "2", "3"])
        self.assertEqual(fallback["final_answer"]["status"], "insufficient_evidence")
        self.assertIn("no text content", fallback["trace"]["llm_calls"][-1]["fallback_error"])


if __name__ == "__main__":
    unittest.main()
