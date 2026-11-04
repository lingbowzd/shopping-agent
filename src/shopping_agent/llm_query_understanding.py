"""One LLM intent proposal, checked against WANDS-supported metadata."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from .query_understanding import CLASS_ALIASES, ProductFilters, QueryPlan, understand_query

SUPPORTED_CLASSES = frozenset(name for _, name in CLASS_ALIASES)


class UnsupportedCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Literal["live price or budget", "live inventory", "shipping or delivery",
                  "unverified dimensions or units"]
    source_text: str = Field(min_length=1, max_length=120)


class IntentDraft(BaseModel):
    """The LLM proposes values; validation checks structure and source grounding."""

    model_config = ConfigDict(extra="forbid", strict=True)

    search_query: str = Field(min_length=1, max_length=300)
    product_class: str | None
    product_class_source: str | None
    material: Literal["wood", "solid wood", "metal", "glass", "leather", "velvet"] | None
    material_source: str | None
    min_historical_rating: float | None = Field(ge=0, le=5)
    rating_source: str | None
    unsupported_constraints: list[UnsupportedCondition] = Field(max_length=4)

    @model_validator(mode="after")
    def check_query_grounding(self, info: ValidationInfo) -> "IntentDraft":
        query = info.context["query"]
        baseline = info.context["baseline"]
        normalized_query = " ".join(query.casefold().split())
        for value, source, field_name in (
            (self.product_class, self.product_class_source, "product_class"),
            (self.material, self.material_source, "material"),
            (self.min_historical_rating, self.rating_source, "min_historical_rating"),
        ):
            if (value is None) != (source is None):
                raise ValueError(f"{field_name} requires a source span exactly when set")
            if source is not None and not source.strip():
                raise ValueError(f"{field_name} source span is blank")
            if source is not None and " ".join(source.casefold().split()) not in normalized_query:
                raise ValueError(f"{field_name} source span is absent from the user query")
        if self.product_class is not None and self.product_class not in SUPPORTED_CLASSES:
            raise ValueError("product_class is not a supported WANDS category")
        for condition in self.unsupported_constraints:
            if not condition.source_text.strip():
                raise ValueError("unsupported condition source span is blank")
            if " ".join(condition.source_text.casefold().split()) not in normalized_query:
                raise ValueError("unsupported condition source span is absent from the user query")
        known = baseline.filters
        if known.product_class is not None and self.product_class != known.product_class:
            raise ValueError("product_class conflicts with a directly recognized category")
        if known.material is not None and self.material != known.material:
            raise ValueError("material conflicts with a directly recognized material")
        if known.min_historical_rating is not None and self.min_historical_rating != known.min_historical_rating:
            raise ValueError("rating conflicts with a directly recognized threshold")
        if not self.search_query.strip():
            raise ValueError("search_query is blank")
        stopwords = {"the", "and", "for", "with", "from", "this", "that", "have", "want", "need"}
        query_terms = {term for term in re.findall(r"\w+", normalized_query)
                       if len(term) >= 3 and term not in stopwords}
        search_terms = {term for term in re.findall(r"\w+", self.search_query.casefold())
                        if len(term) >= 3 and term not in stopwords}
        if not query_terms & search_terms:
            raise ValueError("search_query has no terms from the user query")
        allowed_terms = set(re.findall(r"\w+", normalized_query))
        allowed_terms.update(re.findall(r"\w+", (self.product_class or "").casefold()))
        allowed_terms.update(re.findall(r"\w+", self.material or ""))
        if search_terms - allowed_terms:
            raise ValueError("search_query adds terms not grounded in the query or selected metadata")
        if re.search(r"\$\s*\d|\b(?:under|below)\s+\d+\s*(?:dollars|usd)\b|\b(?:in\s+stock|available\s+now|next[- ]day|shipping|delivery)\b",
                     self.search_query, re.I):
            raise ValueError("search_query still contains an unsupported live-data condition")
        return self


SYSTEM_PROMPT = """Extract a shopping search plan from the user's query. Return exactly one JSON object with these fields:
search_query (short English search text), product_class, product_class_source,
material, material_source, min_historical_rating, rating_source, unsupported_constraints.
Use null for absent filters and their source fields. Every non-null filter must cite a
short exact substring of the user query in its matching *_source field. Do not invent
requirements. product_class must be one of the allowed WANDS classes supplied below.
material must be wood, solid wood, metal, glass, leather, or velvet. Rating is a
minimum historical average on a 0-5 scale, only when the user requests one.
unsupported_constraints is an array of {"name": ..., "source_text": ...}; names
can only be live price or budget, live inventory, shipping or delivery, or
unverified dimensions or units.
These have no WANDS metadata and must never become filters. Preserve useful
shopping terms in search_query; remove unsupported numeric constraints.
Return JSON only, with no markdown or tool calls."""


def understand_query_with_llm(query: str, llm: Any) -> tuple[QueryPlan, dict[str, Any]]:
    """Return a validated LLM plan or the deterministic plan after validation failure."""
    baseline = understand_query(query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\nAllowed product_class values: "
         + json.dumps(sorted(SUPPORTED_CLASSES))},
        {"role": "user", "content": query},
    ]
    started = time.perf_counter()
    response = llm.generate(messages)
    trace = {
        "purpose": "query_understanding", "model": response.model,
        "request_messages": messages, "response_text": response.content,
        "usage": response.usage, "finish_reason": response.finish_reason,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    try:
        draft = IntentDraft.model_validate_json(
            response.content, context={"query": query, "baseline": baseline}
        )
    except ValueError as error:
        trace["validation"] = "fallback_to_rules"
        trace["validation_error"] = str(error)[:500]
        return baseline, trace
    unsupported = list(baseline.unsupported_constraints)
    for condition in draft.unsupported_constraints:
        if condition.name not in unsupported:
            unsupported.append(condition.name)
    plan = QueryPlan(
        original_query=baseline.original_query,
        search_query=" ".join(draft.search_query.split()),
        filters=ProductFilters(draft.product_class, draft.min_historical_rating, draft.material),
        unsupported_constraints=tuple(unsupported),
    )
    trace["validation"] = "accepted"
    return plan, trace
