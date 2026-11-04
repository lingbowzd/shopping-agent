"""Local WANDS shopping search comparison UI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.modes import DEFAULT_LOG, DEFAULT_PRODUCTS, run_pipeline  # noqa: E402
from shopping_agent.query_understanding import ProductFilters  # noqa: E402
from shopping_agent.retrieval import ProductSearchService  # noqa: E402
from shopping_agent.schemas import RetrievalConfig  # noqa: E402

st.set_page_config(page_title="WANDS Shopping Search", layout="wide")
st.title("WANDS Shopping Search")
st.caption("Real catalog records · historical ratings · no live price, stock, shipping, photos or purchase links")


@st.cache_resource(show_spinner="Loading WANDS catalog and BM25 index…")
def catalog() -> ProductSearchService:
    return ProductSearchService(DEFAULT_PRODUCTS)


def display_result(result: dict, heading: str) -> None:
    st.subheader(heading)
    trace = result["trace"]
    route = result["route_decision"]
    st.info(f"Route: {route['route']} ({route['router_type']}) — {route['reason']}")
    plan = result["query_plan"]
    st.write("Semantic query:", plan["search_query"])
    st.write("Applied filters:", {key: value for key, value in plan["filters"].items() if value is not None}
             or "none")
    if trace.get("parsed_filters") is not None:
        st.caption(f"Parsed filters: {trace['parsed_filters']}")
    if plan["unsupported_constraints"]:
        st.warning("WANDS cannot verify: " + ", ".join(plan["unsupported_constraints"]))
    st.caption(f"Catalog candidates: {trace.get('catalog_count', 'N/A')} · "
               f"metadata eligible: {trace.get('eligible_count', 'N/A')}")
    if result["mode"] != "search_only":
        final = result["final_answer"]
        st.caption(f"Answer status: {final['status']}")
        st.write(final["answer"])
        if final.get("shared_style"):
            st.caption(f"Shared style evidence: {final['shared_style']}")
        if final.get("unknown_requirements"):
            st.warning("Unverified requirements: " + "; ".join(final["unknown_requirements"]))
        if final.get("source_field_corrections"):
            st.caption("Verified field corrections: " + "; ".join(final["source_field_corrections"]))
    seen = set()
    for hit in result["cards"]:
        if hit["product_id"] in seen:
            continue
        seen.add(hit["product_id"])
        with st.container(border=True):
            st.markdown(f"**{hit['product_id']} · {hit['title']}**")
            st.caption(f"{hit.get('product_class') or 'Unclassified'} · "
                       f"historical rating: {hit.get('average_rating', hit.get('historical_average_rating')) or 'unknown'} "
                       f"({hit.get('rating_count', hit.get('historical_rating_count')) or 'unknown'} ratings)")
            if hit.get("score") is not None:
                st.caption(f"Rank {hit.get('rank')} · {hit.get('score_type')} {hit['score']}")
            for item in hit.get("evidence", [])[:2]:
                st.write(f"{item['field']}: {item['text']}")
            for item in hit.get("filter_evidence", [])[:4]:
                st.write(f"Filter evidence — {item['field']}: {item['value']}")
            features = hit.get("features") or {}
            if features:
                with st.expander("Observed feature fields"):
                    st.json(features)
    usage = trace["usage_summary"]
    model_names = sorted({str(call.get("model")) for call in trace["llm_calls"] if call.get("model")})
    st.caption(f"LLM calls: {usage['llm_call_count']} · input/output tokens: "
               f"{usage['input_tokens']}/{usage['output_tokens']} · "
               f"LLM API cost estimate: {usage['estimated_llm_api_cost_usd'] if usage['estimated_llm_api_cost_usd'] is not None else 'N/A'} "
               f"· total latency: {usage['total_latency_ms']} ms")
    if not usage["token_usage_complete"]:
        st.caption(usage["cost_note"])
    if model_names:
        st.caption("Models: " + ", ".join(model_names))
    if trace.get("budget_usage"):
        st.caption(f"Agent budget use: {trace['budget_usage']} / {trace['budgets']}")
    with st.expander("Retrieval stages"):
        for index, call in enumerate(trace["retrieval_calls"], 1):
            st.write(f"Search {index}: {call['query']} → {call['retrieved_ids']}")
            if call.get("stages"):
                st.json(call["stages"])
    if result["mode"] == "agentic_rag":
        st.markdown("**Actual agent tool calls**")
        for index, call in enumerate(trace["agent_tool_calls"], 1):
            st.write(f"{index}. search_products({call['query']!r}) → {call.get('retrieved_ids', [])}")
            if call.get("observed_cue_from_first_search"):
                st.caption(f"New observed cue: {call['observed_cue_from_first_search']}")
            if call.get("error"):
                st.warning(call["error"])
    with st.expander("Full execution trace"):
        st.json(trace)


service = catalog()
with st.sidebar:
    st.header("Run controls")
    pipeline = st.selectbox("Pipeline", ["auto", "search_only", "standard_rag", "agentic_rag"])
    backend = st.selectbox("Retrieval backend", ["hybrid", "bm25", "dense"])
    use_filter = st.checkbox("Use parsed metadata filters", value=True)
    rerank = st.checkbox("Cohere reranker", value=False)
    parser = st.selectbox("Query parser", ["rules", "none", "llm"])
    top_k = st.slider("Top K", 1, 10, 5)
    candidate_k = st.slider("Rerank candidate pool", 5, 100, 30)
    categories = ["(none)"] + sorted(service.class_postings)
    category = st.selectbox("Manual WANDS category", categories)
    rating_on = st.checkbox("Manual minimum historical rating")
    rating = st.slider("Minimum rating", 0.0, 5.0, 4.0, 0.5, disabled=not rating_on)
    material = st.selectbox("Manual material", ["(none)"] + sorted(service.material_postings))
    st.caption("Manual hard filters remain active even when parsed metadata filtering is off.")

query = st.text_area(
    "Shopping request",
    value="I'm furnishing my first apartment. I like mid-century modern furniture. "
          "Can you help me pick a coffee table and matching end tables?",
    height=100,
)
col_run, col_compare = st.columns(2)
run_clicked = col_run.button("Run selected mode", type="primary", use_container_width=True)
compare_clicked = col_compare.button("Compare all three manual modes", use_container_width=True)

config = RetrievalConfig(
    backend=backend, use_metadata_filter=use_filter, use_reranker=rerank,
    candidate_k=max(candidate_k, top_k), top_k=top_k,
)
manual = ProductFilters(
    None if category == "(none)" else category,
    rating if rating_on else None,
    None if material == "(none)" else material,
)

if run_clicked or compare_clicked:
    modes = ["search_only", "standard_rag", "agentic_rag"] if compare_clicked else [pipeline]
    results = []
    for mode in modes:
        with st.spinner(f"Running {mode}…"):
            try:
                results.append(run_pipeline(
                    query, mode, backend, top_k, search_service=service,
                    retrieval_config=config, manual_filters=manual,
                    query_understanding_mode=parser, log_path=DEFAULT_LOG,
                ))
            except Exception as error:
                st.error(f"{mode} failed: {error}")
    st.session_state["shopping_results"] = results
    st.session_state["shopping_query"] = query

if st.session_state.get("shopping_results"):
    st.caption(f"Results for: {st.session_state.get('shopping_query')}")
    results = st.session_state["shopping_results"]
    if len(results) > 1:
        for column, result in zip(st.columns(len(results)), results):
            with column:
                display_result(result, result["mode"].replace("_", " ").title())
    else:
        result = results[0]
        display_result(result, result["mode"].replace("_", " ").title())
    st.download_button(
        "Download traces as JSON",
        data=json.dumps([result["trace"] for result in results],
                        indent=2, ensure_ascii=False),
        file_name="wands_traces.json",
        mime="application/json",
    )
