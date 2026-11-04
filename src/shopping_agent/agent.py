"""Bounded LangGraph tool loop over the shared WANDS search service."""

from __future__ import annotations

import json
import os
import re
import ssl
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
import truststore
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field

from .llm import ConfiguredLLM, _config
from .query_understanding import ProductFilters
from .rag import MAX_CONTEXT_CHARS, SYSTEM_PROMPT as FINAL_SYSTEM_PROMPT, validate_answer
from .router import COORDINATION
from .schemas import RetrievalConfig

FEATURE_CUES = ("color", "material", "style", "finish", "shape")
MAX_TOOL_TEXT = 6000
REQUESTED_CATEGORY_GOALS = (
    ("dining tables", r"\bdining\s+tables?\b", r"\bdining\s+tables?\b"),
    ("coffee tables", r"\bcoffee\s+tables?\b", r"\bcoffee\s*&\s*cocktail\s+tables\b"),
    ("end tables", r"\bend\s+tables?\b", r"\bend\s+tables\b"),
    ("chairs", r"\bchairs?\b", r"\bchairs\b"),
    ("sofas", r"\bsofas?\b", r"\bsofas\b"),
    ("desks", r"\bdesks?\b", r"\bdesks\b"),
    ("rugs", r"\brugs?\b", r"\brugs\b"),
    ("lamps", r"\blamps?\b", r"\blamps\b"),
    ("beds", r"\bbeds?\b", r"\bbeds\b"),
)

SYSTEM_PROMPT = """You are a WANDS catalog shopping assistant. Search with search_products when catalog
evidence is needed. The tool returns compact source-backed product fields, not live
price, stock, delivery, images or links. Never invent those unavailable facts.
For coordinated products, use a second search only if useful: choose its terms
from a concrete field/value you actually saw in an earlier search result AND
that was not already in the user's request. Put a color, finish, material or
shape value literally in the next search query, such as "acacia end table".
For a coordinated set, cover every requested product category within the call
budget. Do not present only one category as a complete set. If a category
cannot be found, name it as an unknown requirement in the final JSON.
The filters object accepts only product_class, min_historical_rating, material;
put color/style/shape words in query, never as new filter keys. If the tool
rejects a follow-up, read its available_cues and use one in a revised query
or stop. A preplanned second category alone is not evidence-conditioned.
Do not search again for missing live data. A single search is fine.
Do not call tools when enough evidence is available. Finish with exactly one JSON
object: {"recommendations":[{"product_id":"string","evidence_field":"title or
product_class or features.KEY or snippet field","evidence_quote":"exact source
substring"}],"unknown_requirements":["short string"]}. At most three products.
Use only IDs and exact source quotes from tool results. No markdown."""


class ToolFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_class: str | None = None
    min_historical_rating: float | None = Field(default=None, ge=0, le=5)
    material: str | None = None


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=300)
    filters: ToolFilters | None = None
    top_k: int = Field(default=5, ge=1, le=20)


@dataclass(frozen=True)
class AgentBudgets:
    max_llm_calls: int = 6
    max_tool_calls: int = 5
    max_search_calls: int = 3

    def __post_init__(self) -> None:
        if min(self.max_llm_calls, self.max_tool_calls, self.max_search_calls) < 1:
            raise ValueError("Agent budgets must be positive")


def configured_chat_model() -> ChatOpenAI:
    model, key, base_url = _config()
    http_client = httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), timeout=90)
    return ChatOpenAI(
        model=model, api_key=key, base_url=base_url, temperature=0,
        max_tokens=2500, max_retries=0, http_client=http_client,
    )


def _compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    for key, values in (hit.get("features") or {}).items():
        if any(cue in key.casefold() for cue in FEATURE_CUES) and isinstance(values, list):
            selected[key] = [str(value)[:100] for value in values[:2]]
        if len(selected) >= 8:
            break
    return {
        "product_id": hit["product_id"],
        "title": str(hit.get("title") or "")[:220],
        "product_class": str(hit.get("product_class") or "")[:120],
        "historical_average_rating": hit.get("average_rating"),
        "historical_rating_count": hit.get("rating_count"),
        "features": selected,
        "evidence": (hit.get("evidence") or [])[:2],
    }


def _grounding_record(compact: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": compact["product_id"], "title": compact["title"],
        "product_class": compact["product_class"],
        "features": compact["features"], "evidence": compact["evidence"],
        "historical_average_rating": compact["historical_average_rating"],
        "historical_rating_count": compact["historical_rating_count"],
    }


def _conditional_cue(first_hits: list[dict[str, Any]], later_query: str,
                     original_query: str) -> dict[str, str] | None:
    query = later_query.casefold()
    original = original_query.casefold()
    for hit in first_hits:
        for key, values in hit.get("features", {}).items():
            for value in values:
                value = value.strip().casefold()
                if len(value) >= 4 and value not in original and re.search(
                    r"\b" + re.escape(value) + r"\b", query
                ):
                    return {"product_id": hit["product_id"], "field": f"features.{key}", "value": value}
    return None


def _available_cues(first_hits: list[dict[str, Any]], original_query: str) -> list[dict[str, str]]:
    cues = []
    for hit in first_hits[:3]:
        for key, values in hit.get("features", {}).items():
            if not any(cue in key.casefold() for cue in ("color", "material", "shape", "finish")):
                continue
            for value in values:
                value = value.strip().casefold()
                if len(value) >= 4 and value not in original_query.casefold():
                    cues.append({"product_id": hit["product_id"], "field": f"features.{key}",
                                 "value": value})
            if len(cues) >= 6:
                return cues
    return cues


def _category_goals(query: str) -> list[tuple[str, str]]:
    return [(label, class_pattern) for label, query_pattern, class_pattern in REQUESTED_CATEGORY_GOALS
            if re.search(query_pattern, query, re.I)]


def _missing_categories(goals: list[tuple[str, str]],
                        hits: list[dict[str, Any]]) -> list[str]:
    return [label for label, class_pattern in goals
            if not any(re.search(class_pattern, hit.get("product_class") or "", re.I)
                       for hit in hits)]


def _source_backed_fallback(goals: list[tuple[str, str]],
                            hits: list[dict[str, Any]]) -> str:
    """Select observed titles, favoring a shared literal style value."""
    selected = []
    anchor_styles: set[str] | None = None
    for _, class_pattern in goals:
        candidates = [hit for hit in hits
                      if re.search(class_pattern, hit.get("product_class") or "", re.I)]
        if not candidates:
            continue
        def styles(hit: dict[str, Any]) -> set[str]:
            return {str(value).strip().casefold()
                    for key, values in hit.get("features", {}).items()
                    if "style" in key.casefold() for value in values if str(value).strip()}

        chosen = next((hit for hit in candidates
                       if anchor_styles and styles(hit) & anchor_styles), candidates[0])
        selected.append({
            "product_id": chosen["product_id"],
            "evidence_field": "title",
            "evidence_quote": chosen["title"],
        })
        current_styles = styles(chosen)
        anchor_styles = (anchor_styles & current_styles
                         if anchor_styles is not None else current_styles)
    return json.dumps({
        "recommendations": selected[:3],
        "unknown_requirements": ["Final LLM returned no text; catalog-evidence fallback was used"],
    }, ensure_ascii=False)


def run_agentic_rag(
    query: str, search_service: Any, config: RetrievalConfig, *,
    manual_filters: ProductFilters | None = None, model: Any | None = None,
    final_llm: Any | None = None,
    budgets: AgentBudgets = AgentBudgets(),
) -> dict[str, Any]:
    """Execute actual StateGraph agent/tool transitions and validate final IDs."""
    chat_model = model if model is not None else configured_chat_model()
    search_tool_calls: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    manual = manual_filters or ProductFilters()
    run_query = query
    coordinated = bool(COORDINATION.search(query))
    category_goals = _category_goals(query) if coordinated else []

    @tool("search_products", args_schema=SearchArgs)
    def search_products(query: str, filters: ToolFilters | None = None, top_k: int = 5) -> str:
        """Search real WANDS products; optional filters are class, historical rating, material."""
        started = time.perf_counter()
        with lock:
            executed = sum("retrieved_ids" in item for item in search_tool_calls)
            if len(search_tool_calls) >= budgets.max_tool_calls or executed >= budgets.max_search_calls:
                return json.dumps({"error": "search budget exhausted"})
            event: dict[str, Any] = {"tool": "search_products", "query": query, "requested_filters": None,
                                     "top_k": top_k}
            search_tool_calls.append(event)
        if coordinated and len(search_tool_calls) > 1 and search_tool_calls[0].get("hits"):
            cue = _conditional_cue(search_tool_calls[0]["hits"], query, original_query=run_query)
            if cue is None:
                event["error"] = "Follow-up search did not use a new observed product attribute"
                event["available_cues"] = _available_cues(search_tool_calls[0]["hits"], run_query)
                event["example_queries"] = [
                    {"query": f"{item['value']} {query}", "source": item}
                    for item in event["available_cues"][:3]
                ]
                event["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return json.dumps({"error": event["error"],
                                   "available_cues": event["available_cues"],
                                   "example_queries": event["example_queries"]}, ensure_ascii=False)
            event["observed_cue_from_first_search"] = cue
        supplied = filters if isinstance(filters, ToolFilters) else ToolFilters.model_validate(filters or {})
        event["requested_filters"] = supplied.model_dump(exclude_none=True)
        parsed = supplied if config.use_metadata_filter else ToolFilters()
        chosen_class = manual.product_class or parsed.product_class
        chosen_material = manual.material or parsed.material
        thresholds = [v for v in (manual.min_historical_rating, parsed.min_historical_rating)
                      if v is not None]
        effective = ProductFilters(chosen_class, max(thresholds) if thresholds else None, chosen_material)
        if manual.product_class and parsed.product_class and manual.product_class != parsed.product_class:
            event["error"] = "Tool category conflicts with manual category"
            return json.dumps({"error": event["error"]})
        if manual.material and parsed.material and manual.material != parsed.material:
            event["error"] = "Tool material conflicts with manual material"
            return json.dumps({"error": event["error"]})
        if chosen_class and hasattr(search_service, "class_postings") and chosen_class not in search_service.class_postings:
            event["error"] = f"Unknown WANDS product_class: {chosen_class}"
            return json.dumps({"error": event["error"]})
        event["effective_filters"] = asdict(effective)
        event["catalog_count"] = len(search_service.ids) if hasattr(search_service, "ids") else None
        event["eligible_count"] = search_service.eligible_count(effective) if hasattr(search_service, "eligible_count") else None
        kwargs = {"mode": config.backend, "top_k": min(top_k, config.top_k), "filters": effective}
        if hasattr(search_service, "eligible_count"):
            event["stages"] = {}
            kwargs.update(rerank=config.use_reranker, candidate_k=max(config.candidate_k, top_k),
                          stage_trace=event["stages"])
        hits = search_service.search(query, **kwargs)
        compact = [_compact_hit(hit) for hit in hits]
        for hit in compact:
            seen[hit["product_id"]] = hit
        event["retrieved_ids"] = [hit["product_id"] for hit in compact]
        event["hits"] = compact
        event["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return json.dumps({"products": compact}, ensure_ascii=False)

    tools = [search_products]
    bound_model = chat_model.bind_tools(tools, parallel_tool_calls=False)
    graph = StateGraph(MessagesState)

    def finalize() -> AIMessage:
        started = time.perf_counter()
        records = []
        prioritized = []
        for _, class_pattern in category_goals:
            prioritized.extend(
                [hit for hit in seen.values()
                 if re.search(class_pattern, hit.get("product_class") or "", re.I)][:2]
            )
        prioritized.extend(seen.values())
        included = set()
        for hit in prioritized:
            if hit["product_id"] in included:
                continue
            candidate = json.dumps({"products": records + [hit]}, ensure_ascii=False)
            if len(candidate) > MAX_CONTEXT_CHARS:
                continue
            records.append(hit)
            included.add(hit["product_id"])
        context = json.dumps({"products": records}, ensure_ascii=False)
        category_instruction = (
            "Cover each requested category with at most one product: "
            + ", ".join(label for label, _ in category_goals)
            + ". " if category_goals else ""
        )
        messages = [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"User request: {run_query}\n\nSource context JSON: {context}\n\n"
                + category_instruction
                + "For each recommendation, use evidence_field title and copy the "
                  "exact title as evidence_quote. Return JSON only."
            )},
        ]
        client = final_llm if final_llm is not None else ConfiguredLLM(max_tokens=3000)
        fallback_error = None
        try:
            response = client.generate(messages)
        except ValueError as error:
            if "no text content" not in str(error):
                raise
            fallback_error = str(error)
            response = None
        if response is None or not response.content.strip():
            fallback_error = fallback_error or "Final LLM response was empty"
            content = _source_backed_fallback(category_goals, list(seen.values()))
            model_name = getattr(client, "model", type(client).__name__)
            usage, finish_reason = {}, "empty_response"
        else:
            content = response.content
            model_name, usage, finish_reason = response.model, response.usage, response.finish_reason
        model_calls.append({
            "purpose": "agent_final_generation", "model": model_name,
            "response_text": content, "tool_calls": [],
            "usage": usage, "finish_reason": finish_reason,
            "fallback_error": fallback_error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "context_chars": len(context),
        })
        return AIMessage(content=content)

    def call_model(state: MessagesState) -> dict[str, Any]:
        executed = sum("retrieved_ids" in item for item in search_tool_calls)
        use_tools = (len(model_calls) < budgets.max_llm_calls - 1
                     and len(search_tool_calls) < budgets.max_tool_calls
                     and executed < budgets.max_search_calls)
        if not use_tools:
            return {"messages": [finalize()]}
        started = time.perf_counter()
        response = bound_model.invoke([SystemMessage(content=SYSTEM_PROMPT)]
                                      + state["messages"])
        model_calls.append({
            "purpose": "agent_decision",
            "model": getattr(chat_model, "model_name", type(chat_model).__name__),
            "response_text": str(response.content or ""),
            "tool_calls": response.tool_calls,
            "usage": response.usage_metadata or {},
            "finish_reason": (response.response_metadata or {}).get("finish_reason"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        missing = _missing_categories(category_goals, list(seen.values()))
        can_continue = (
            missing and len(model_calls) < budgets.max_llm_calls - 1
            and len(search_tool_calls) < budgets.max_tool_calls
            and executed < budgets.max_search_calls
        )
        if not response.tool_calls and not can_continue and (
            not str(response.content or "").strip() or category_goals
        ):
            if len(model_calls) < budgets.max_llm_calls:
                response = finalize()
        return {"messages": [response]}

    def next_node(state: MessagesState) -> str:
        last = state["messages"][-1]
        if last.tool_calls and len(model_calls) < budgets.max_llm_calls:
            return "tools"
        executed = sum("retrieved_ids" in item for item in search_tool_calls)
        if (_missing_categories(category_goals, list(seen.values()))
                and len(model_calls) < budgets.max_llm_calls - 1
                and len(search_tool_calls) < budgets.max_tool_calls
                and executed < budgets.max_search_calls):
            return "continue"
        return "end"

    def prompt_continue(state: MessagesState) -> dict[str, Any]:
        missing = _missing_categories(category_goals, list(seen.values()))
        return {"messages": [HumanMessage(content=(
            "The requested coordinated set still lacks retrieved evidence for: "
            + ", ".join(missing)
            + ". Decide whether to search a missing category using an exact new "
              "attribute observed in earlier product results. If reliable evidence "
              "cannot be found within budget, report the missing categories."
        ))]}

    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("continue", prompt_continue)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", next_node,
                                {"tools": "tools", "continue": "continue", "end": END})
    graph.add_edge("tools", "agent")
    graph.add_edge("continue", "agent")
    result = graph.compile().invoke({"messages": [HumanMessage(content=query)]},
                                    config={"recursion_limit": 2 * budgets.max_llm_calls + 4})
    raw_final = str(result["messages"][-1].content or "")
    final = validate_answer(raw_final, [_grounding_record(hit) for hit in seen.values()])
    for missing in _missing_categories(category_goals, [
        seen[item["product_id"]] for item in final["recommendations"]
    ]):
        notice = f"No verified {missing} recommendation in this set"
        if notice not in final["unknown_requirements"]:
            final["unknown_requirements"].append(notice)
    if len(category_goals) >= 2 and final["recommendations"]:
        notice = "Visual compatibility cannot be verified from WANDS text alone"
        if notice not in final["unknown_requirements"]:
            final["unknown_requirements"].append(notice)
    if final["recommendations"] and final["unknown_requirements"]:
        final["status"] = "insufficient_evidence"
    trace = {
        "pipeline_mode": "agentic_rag", "query": query,
        "retrieval_backend": config.backend, "retrieval_config": config.model_dump(),
        "llm_calls": model_calls,
        "retrieval_calls": [item for item in search_tool_calls if "retrieved_ids" in item],
        "agent_tool_calls": search_tool_calls,
        "tool_messages": [
            {"tool_call_id": msg.tool_call_id, "content": str(msg.content)[:MAX_TOOL_TEXT]}
            for msg in result["messages"] if msg.type == "tool"
        ],
        "budgets": asdict(budgets),
        "budget_usage": {"llm_calls": len(model_calls), "tool_calls": len(search_tool_calls),
                         "search_calls": sum("retrieved_ids" in item for item in search_tool_calls)},
        "grounded_final_answer": final,
    }
    return {
        "mode": "agentic_rag", "query": query, "backend": config.backend,
        "cards": [hit for event in search_tool_calls for hit in event.get("hits", [])],
        "final_answer": final, "trace": trace,
    }
