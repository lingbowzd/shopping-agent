"""Small transparent policy for choosing the cheapest sufficient pipeline."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["search_only", "standard_rag", "agentic_rag"]
    reason: str
    router_type: Literal["manual", "rules"] = "rules"
    confidence: float | None = None


CATEGORY_PATTERNS = (
    r"\bcoffee\s+tables?\b", r"\bend\s+tables?\b", r"\bdining\s+tables?\b",
    r"\bdesks?\b",
    r"\bchairs?\b", r"\bsofas?\b", r"\brugs?\b", r"\bstorage\b",
    r"\bbookcases?\b", r"\blamps?\b", r"\bbeds?\b",
)
COORDINATION = re.compile(
    r"\b(?:match(?:es|ing)?|same\s+style|cohesive|complement\w*|"
    r"go\s+together|work\s+(?:well\s+)?together|"
    r"coordinate|coordinated|furnish(?:ing)?|set(?:ting)?\s+up|set\s+of)\b", re.I
)
RECOMMENDATION = re.compile(
    r"\b(?:recommend|suggest|best|good|better|compare|comparison|"
    r"which\s+of\s+these|which\s+one|would\s+suit|suits?|would\s+work\s+best|"
    r"trade[- ]offs?)\b", re.I
)
UNSUPPORTED_LIVE = re.compile(
    r"\$\s*\d|\b(?:ships?\s+tomorrow|arriv\w*\s+tomorrow|in\s+stock|"
    r"under\s+\$\s*\d|delivery\s+by)\b", re.I
)


def route_query(query: str) -> RouteDecision:
    """Route on task structure; never escalate merely for more filter terms."""
    categories = sum(bool(re.search(pattern, query, re.I)) for pattern in CATEGORY_PATTERNS)
    if COORDINATION.search(query) and categories >= 2:
        return RouteDecision(
            route="agentic_rag",
            reason="Coordinated products may need a later search based on an earlier catalog result.",
        )
    if UNSUPPORTED_LIVE.search(query):
        return RouteDecision(
            route="search_only",
            reason="Requested live commerce data is unavailable in WANDS; extra LLM steps cannot verify it.",
        )
    if RECOMMENDATION.search(query):
        return RouteDecision(
            route="standard_rag",
            reason="A recommendation or comparison can use one fixed retrieved candidate set.",
        )
    return RouteDecision(
        route="search_only",
        reason="A direct product lookup or one-product constraint search needs ranked catalog cards.",
    )


def manual_route(route: str) -> RouteDecision:
    if route not in {"search_only", "standard_rag", "agentic_rag"}:
        raise ValueError(f"Unsupported manual route: {route}")
    return RouteDecision(route=route, reason="Selected manually.", router_type="manual")
