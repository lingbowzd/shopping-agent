"""Run Auto, Search Only, Standard RAG or Agentic RAG over WANDS."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.modes import DEFAULT_LOG, DEFAULT_PRODUCTS, run_pipeline  # noqa: E402
from shopping_agent.retrieval import ProductSearchService  # noqa: E402
from shopping_agent.query_understanding import ProductFilters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--pipeline", choices=("auto", "search_only", "standard_rag", "agentic_rag"), default="search_only")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid", "hybrid_rerank"), default="hybrid")
    parser.add_argument("--query-understanding", choices=("none", "rules", "llm"), default="rules")
    parser.add_argument("--no-filter", action="store_true", help="Disable parsed metadata filters")
    parser.add_argument("--rerank", action="store_true", help="Use Cohere reranking after the chosen retriever")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--category", help="Explicit WANDS product_class filter")
    parser.add_argument("--min-rating", type=float, help="Minimum historical average rating")
    parser.add_argument("--material", help="Explicit parsed material filter")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    service = ProductSearchService(args.products)
    result = run_pipeline(
        args.query, args.pipeline, args.retriever, args.top_k,
        search_service=service, log_path=args.log,
        query_understanding_mode=args.query_understanding,
        use_metadata_filter=not args.no_filter,
        use_reranker=args.rerank or args.retriever == "hybrid_rerank",
        candidate_k=args.candidate_k,
        manual_filters=ProductFilters(args.category, args.min_rating, args.material),
    )
    trace = result["trace"]
    route = result["route_decision"]
    print(f"Pipeline: {result['mode']} | retriever: {trace['retrieval_config']['backend']} | rerank: {trace['retrieval_config']['use_reranker']}")
    print(f"Route: {route['route']} ({route['router_type']}) — {route['reason']}")
    if trace["query_understanding_mode"] == "llm":
        print(f"Query understanding: LLM ({trace['query_understanding_validation']})")
    else:
        print(f"Query understanding: {trace['query_understanding_mode']}")
    plan = result["query_plan"]
    print(f"Search query: {plan['search_query']}")
    applied = {key: value for key, value in plan["filters"].items() if value is not None}
    print(f"Metadata filters: {applied or '(none)'}")
    if plan["unsupported_constraints"]:
        print("Cannot filter with WANDS: " + "; ".join(plan["unsupported_constraints"]))
    retrieved_ids = [product_id for call in trace["retrieval_calls"]
                     for product_id in call.get("retrieved_ids", [])]
    print(f"Retrieved IDs: {', '.join(retrieved_ids) or '(none)'}")
    if result["mode"] == "search_only":
        for hit in result["cards"]:
            print(f"{hit['rank']}. {hit['product_id']} | {hit['title']} | {hit['product_class']} | {hit['score_type']}={hit['score']}")
            for item in hit["evidence"]:
                print(f"   {item['field']}: {item['text']}")
            for item in hit.get("filter_evidence", []):
                print(f"   filter {item['field']}: {item['value']}")
    else:
        final = result["final_answer"]
        print(f"LLM calls: {len(trace['llm_calls'])} | agent tool calls: {len(trace['agent_tool_calls'])}")
        print(f"Answer status: {final['status']}")
        if result["mode"] == "agentic_rag":
            for index, call in enumerate(trace["agent_tool_calls"], 1):
                print(f"  Tool {index}: {call['tool']}({call['query']!r}, filters={call.get('effective_filters')}) -> {call.get('retrieved_ids', [])}")
                if call.get("observed_cue_from_first_search"):
                    print(f"    Observed cue: {call['observed_cue_from_first_search']}")
        print(final["answer"])
        if final.get("shared_style"):
            print(f"Shared style evidence: {final['shared_style']}")
        for item in final["recommendations"]:
            print(f"  ID {item['product_id']} | {item['title']} | {item['evidence_field']}: {item['evidence_quote']}")
        if final["unknown_requirements"]:
            print("Unverified requirements: " + "; ".join(final["unknown_requirements"]))
        if final["rejected_model_items"]:
            print("Validation: " + "; ".join(final["rejected_model_items"]))
        if final.get("source_field_corrections"):
            print("Verified source field corrections: " + "; ".join(final["source_field_corrections"]))
    usage = trace["usage_summary"]
    print(f"Usage: {usage['input_tokens']} input tokens, {usage['output_tokens']} output tokens; "
          f"estimated LLM API cost: {usage['estimated_llm_api_cost_usd']}; "
          f"latency: {usage['total_latency_ms']} ms")
    if not usage["token_usage_complete"]:
        print("Usage note: " + usage["cost_note"])
    print(f"Pipeline log: {args.log}")


if __name__ == "__main__":
    main()
