"""Dispatch manual or routed shopping pipelines over one WANDS catalog."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import ConfiguredLLM
from .llm_query_understanding import understand_query_with_llm
from .query_understanding import ProductFilters, QueryPlan, understand_query
from .rag import run_standard_rag
from .retrieval import ProductSearchService
from .router import manual_route, route_query
from .schemas import RetrievalConfig
from .tracing import summarize_usage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCTS = ROOT / "data/processed/products.jsonl"
DEFAULT_LOG = ROOT / "data/processed/logs/pipeline.jsonl"
PIPELINE_MODES = {"auto", "search_only", "standard_rag", "agentic_rag"}
RETRIEVAL_BACKENDS = {"bm25", "dense", "hybrid", "hybrid_rerank"}


class ConfiguredSearch:
    """Add independent filtering/reranking options to the shared service."""

    def __init__(self, service: Any, config: RetrievalConfig):
        self.service = service
        self.config = config
        self.last_stages: dict[str, Any] = {}

    def search(self, query: str, mode: str, top_k: int,
               filters: ProductFilters | None = None) -> list[dict[str, Any]]:
        if isinstance(self.service, ProductSearchService):
            return self.service.search(
                query, mode=self.config.backend, top_k=top_k, filters=filters,
                rerank=self.config.use_reranker,
                candidate_k=max(self.config.candidate_k, top_k),
                stage_trace=self.last_stages,
            )
        return self.service.search(query, mode=self.config.backend, top_k=top_k, filters=filters)


class ExplicitIDLookup:
    """Match the one-search interface with a single direct batch catalog read."""

    def __init__(self, service: ProductSearchService, product_ids: list[str]):
        self.service = service
        self.product_ids = product_ids

    def search(self, query: str, mode: str, top_k: int,
               filters: ProductFilters | None = None) -> list[dict[str, Any]]:
        return self.service.get_products(self.product_ids[:top_k], query, filters)


def _merge_filters(parsed: ProductFilters, manual: ProductFilters) -> ProductFilters:
    ratings = [value for value in (parsed.min_historical_rating, manual.min_historical_rating)
               if value is not None]
    return ProductFilters(
        manual.product_class or parsed.product_class,
        max(ratings) if ratings else None,
        manual.material or parsed.material,
    )


def run_search_only(query: str, retrieval_backend: str, top_k: int, search_service: Any,
                    plan: QueryPlan | None = None) -> dict[str, Any]:
    plan = plan or understand_query(query)
    started = time.perf_counter()
    hits = search_service.search(plan.search_query, mode=retrieval_backend, top_k=top_k,
                                 filters=plan.filters)
    retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
    trace = {
        "pipeline_mode": "search_only",
        "query": query,
        "query_plan": plan.to_dict(),
        "retrieval_backend": retrieval_backend,
        "top_k": top_k,
        "retrieval_calls": [{
            "query": plan.search_query, "backend": retrieval_backend, "top_k": top_k,
            "filters": plan.to_dict()["filters"],
            "retrieved_ids": [hit["product_id"] for hit in hits],
            "latency_ms": retrieval_ms,
            "stages": getattr(search_service, "last_stages", None),
        }],
        "prompt_context": None,
        "context_chars": 0,
        "llm_calls": [],
        "agent_tool_calls": [],
        "grounded_final_answer": None,
    }
    return {"mode": "search_only", "query": query, "backend": retrieval_backend,
            "query_plan": plan.to_dict(), "cards": hits, "final_answer": None, "trace": trace}


def run_pipeline(
    query: str, pipeline_mode: str = "search_only", retrieval_backend: str = "hybrid",
    top_k: int = 5, *, search_service: Any | None = None, llm: Any | None = None,
    log_path: Path | None = DEFAULT_LOG, query_understanding_mode: str = "rules",
    query_llm: Any | None = None, retrieval_config: RetrievalConfig | None = None,
    use_metadata_filter: bool = True, use_reranker: bool | None = None,
    candidate_k: int = 30, manual_filters: ProductFilters | None = None,
    agent_model: Any | None = None, agent_budgets: Any | None = None,
) -> dict[str, Any]:
    """Run one mode and append its actual execution log as one JSONL line."""
    if pipeline_mode not in PIPELINE_MODES:
        raise ValueError(f"Unsupported pipeline mode: {pipeline_mode}")
    if retrieval_backend not in RETRIEVAL_BACKENDS:
        raise ValueError(f"Unsupported retrieval backend: {retrieval_backend}")
    if query_understanding_mode not in {"none", "rules", "llm"}:
        raise ValueError(f"Unsupported query understanding mode: {query_understanding_mode}")
    if not query.strip() or len(query) > 1_000:
        raise ValueError("Query must contain 1-1000 characters")
    if top_k < 1 or top_k > 50:
        raise ValueError("top_k must be between 1 and 50")
    started_total = time.perf_counter()
    decision = route_query(query) if pipeline_mode == "auto" else manual_route(pipeline_mode)
    selected_mode = decision.route
    backend = "hybrid" if retrieval_backend == "hybrid_rerank" else retrieval_backend
    config = retrieval_config or RetrievalConfig(
        backend=backend, use_metadata_filter=use_metadata_filter,
        use_reranker=(retrieval_backend == "hybrid_rerank") if use_reranker is None else use_reranker,
        candidate_k=max(candidate_k, top_k), top_k=top_k,
    )
    service = search_service if search_service is not None else ProductSearchService(DEFAULT_PRODUCTS)
    manual = manual_filters or ProductFilters()
    if hasattr(service, "class_postings") and manual.product_class and manual.product_class not in service.class_postings:
        raise ValueError(f"Unknown WANDS product_class: {manual.product_class}")
    if hasattr(service, "material_postings") and manual.material and manual.material not in service.material_postings:
        raise ValueError(f"Unsupported material filter: {manual.material}")
    understanding_call = None
    if selected_mode == "agentic_rag":
        # The tool-calling model supplies validated search arguments itself.
        baseline = understand_query(query)
        parsed_hard = baseline.filters
        hard_filters = _merge_filters(
            parsed_hard if config.use_metadata_filter else ProductFilters(), manual
        )
        plan = QueryPlan(baseline.original_query, baseline.original_query, hard_filters,
                         baseline.unsupported_constraints)
        from .agent import AgentBudgets, run_agentic_rag

        result = run_agentic_rag(
            query, service, config, manual_filters=hard_filters, model=agent_model,
            final_llm=llm,
            budgets=agent_budgets or AgentBudgets(),
        )
        result["query_plan"] = plan.to_dict()
        for requirement in plan.unsupported_constraints:
            notice = (f"{requirement} cannot be verified reliably from WANDS"
                      if requirement == "unverified dimensions or units"
                      else f"{requirement} is unavailable in WANDS")
            if notice not in result["final_answer"]["unknown_requirements"]:
                result["final_answer"]["unknown_requirements"].append(notice)
        if result["final_answer"]["recommendations"] and result["final_answer"]["unknown_requirements"]:
            result["final_answer"]["status"] = "insufficient_evidence"
        effective_parser = "agent_tool_arguments"
        result["trace"]["parsed_filters"] = asdict(parsed_hard)
    elif query_understanding_mode == "llm":
        client = query_llm if query_llm is not None else llm
        if client is None:
            client = ConfiguredLLM(max_tokens=1500)
        plan, understanding_call = understand_query_with_llm(query, client)
        if selected_mode == "standard_rag" and llm is None and query_llm is None:
            llm = client
        effective_parser = "llm"
    elif query_understanding_mode == "none":
        baseline = understand_query(query)
        plan = QueryPlan(baseline.original_query, query, ProductFilters(),
                         baseline.unsupported_constraints)
        effective_parser = "none"
    else:
        plan = understand_query(query)
        effective_parser = "rules"
    if selected_mode != "agentic_rag":
        parsed_filters = plan.filters
        applied_filters = _merge_filters(
            parsed_filters if config.use_metadata_filter else ProductFilters(), manual
        )
        plan = QueryPlan(plan.original_query, plan.search_query, applied_filters,
                         plan.unsupported_constraints)
        configured_service = ConfiguredSearch(service, config)
        explicit_ids = re.findall(r"\b(?:product[_ ]?id|id)\s*#?\s*(\d+)\b", query, re.I)
        if selected_mode == "standard_rag" and explicit_ids and isinstance(service, ProductSearchService):
            configured_service = ExplicitIDLookup(service, explicit_ids)
        if selected_mode == "search_only":
            result = run_search_only(query, config.backend, config.top_k, configured_service, plan)
        else:
            result = run_standard_rag(query, config.backend, config.top_k, configured_service, llm, plan)
        result["trace"]["parsed_filters"] = asdict(parsed_filters)
        if isinstance(configured_service, ExplicitIDLookup):
            result["trace"]["retrieval_calls"][0]["lookup_type"] = "explicit_product_ids"
    result["trace"]["query_understanding_mode"] = effective_parser
    result["trace"]["route_decision"] = decision.model_dump()
    result["trace"]["retrieval_config"] = config.model_dump()
    if hasattr(service, "ids"):
        result["trace"]["catalog_count"] = len(service.ids)
        result["trace"]["eligible_count"] = service.eligible_count(plan.filters)
    if understanding_call is not None:
        result["trace"]["llm_calls"].insert(0, understanding_call)
        result["trace"]["query_understanding_validation"] = understanding_call["validation"]
    result["trace"]["usage_summary"] = summarize_usage(
        result["trace"], (time.perf_counter() - started_total) * 1000
    )
    result["route_decision"] = decision.model_dump()
    result["trace"]["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result["trace"], ensure_ascii=False) + "\n")
    return result
