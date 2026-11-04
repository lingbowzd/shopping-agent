"""Conservative, deterministic interpretation of WANDS shopping queries."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProductFilters:
    product_class: str | None = None
    min_historical_rating: float | None = None
    material: str | None = None

    def is_empty(self) -> bool:
        return all(value is None for value in asdict(self).values())


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    search_query: str
    filters: ProductFilters
    unsupported_constraints: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "search_query": self.search_query,
            "filters": asdict(self.filters),
            "unsupported_constraints": list(self.unsupported_constraints),
        }


CLASS_ALIASES = (
    (r"\bcoffee\s+(?:and\s+cocktail\s+)?tables?\b", "Coffee & Cocktail Tables"),
    (r"\baccent\s+chairs?\b", "Accent Chairs"),
    (r"\bdining\s+chairs?\b", "Dining Chairs"),
    (r"\bbar\s+stools?\b", "Bar Stools"),
    (r"\bend\s+tables?\b", "End Tables"),
    (r"\bdining\s+tables?\b", "Dining Tables"),
    (r"\btable\s+lamps?\b", "Table Lamps"),
    (r"\bfloor\s+lamps?\b", "Floor Lamps"),
    (r"\barea\s+rugs?\b", "Area Rugs"),
    (r"\b(?:bookcases?|bookshelves|bookshelf)\b", "Bookcases"),
    (r"\b(?:sofas?|couches?)\b", "Sofas"),
    (r"\bdesks?\b", "Desks"),
    (r"\bnightstands?\b", "Nightstands"),
    (r"\bbeds?\b", "Beds"),
    (r"\bchandeliers?\b", "Chandeliers"),
)

RATING_PATTERNS = (
    re.compile(r"\b(?:at\s+least|minimum|rated\s+at\s+least|>=)\s*([0-5](?:\.\d)?)\s*(?:stars?|/\s*5)\b", re.I),
    re.compile(r"\b([0-5](?:\.\d)?)\s*\+\s*stars?\b", re.I),
    re.compile(r"\brated\s+([0-5](?:\.\d)?)\s*stars?\s+or\s+(?:higher|better)\b", re.I),
)
MATERIAL_RE = re.compile(r"\b(?:made\s+of|constructed\s+from|material\s*[:=]?)\s+(solid\s+wood|wood|metal|glass|leather|velvet)\b", re.I)
SOLID_WOOD_RE = re.compile(r"\bsolid\s+wood\b(?!\s+finish)", re.I)


def understand_query(query: str) -> QueryPlan:
    """Extract only supported hard filters; keep other intent in the search text."""
    normalized = " ".join(query.split())
    class_match = None
    for pattern, name in CLASS_ALIASES:
        match = re.search(pattern, normalized, re.I)
        if match:
            class_match = (match, name)
            break
    # A single class filter would silently discard other requested categories.
    class_mentions = [(match.start(), match.end(), name)
                      for pattern, name in CLASS_ALIASES
                      for match in re.finditer(pattern, normalized, re.I)]
    matched_classes = {name for _, _, name in class_mentions}
    other_chair_mention = any(
        not any(start <= match.start() and match.end() <= end
                for start, end, _ in class_mentions)
        for match in re.finditer(r"\bchairs?\b", normalized, re.I)
    )
    category = (class_match[1] if class_match and len(matched_classes) == 1
                and not other_chair_mention else None)
    rating = None
    search_query = normalized
    for pattern in RATING_PATTERNS:
        match = pattern.search(normalized)
        if match:
            rating = float(match.group(1))
            search_query = search_query.replace(match.group(), " ")
            break
    material_match = MATERIAL_RE.search(normalized) or SOLID_WOOD_RE.search(normalized)
    material = material_match.group(1).lower() if material_match and material_match.lastindex else (
        "solid wood" if material_match else None
    )
    if material is None and class_match:
        prefix = normalized[:class_match[0].start()]
        adjective = re.search(r"\b(wooden|wood|metal|glass|leather|velvet)\s+$", prefix, re.I)
        if adjective:
            material = "wood" if adjective.group(1).lower() == "wooden" else adjective.group(1).lower()
    unsupported = []
    if re.search(r"\$\s*\d|\b(?:price|budget|cost)\b", normalized, re.I):
        unsupported.append("live price or budget")
        search_query = re.sub(r"\b(?:under|below|less\s+than)\s*\$\s*\d+(?:\.\d+)?\b", " ", search_query, flags=re.I)
    if re.search(r"\b(?:in\s+stock|available\s+now|inventory)\b", normalized, re.I):
        unsupported.append("live inventory")
        search_query = re.sub(r"\b(?:in\s+stock|available\s+now|inventory)\b", " ", search_query, flags=re.I)
    if re.search(r"\b(?:shipping|delivery|delivered|next[- ]day|ship\s+by)\b", normalized, re.I):
        unsupported.append("shipping or delivery")
        search_query = re.sub(r"\b(?:shipping|delivery|delivered|next[- ]day|ship\s+by)\b", " ", search_query, flags=re.I)
    dimension = re.search(
        r"\b(?:(?:under|below|less\s+than|at\s+most|up\s+to)\s*)?"
        r"\d+(?:\.\d+)?\s*(?:inches|inch|in\.)(?!\w)", normalized, re.I
    )
    if dimension:
        unsupported.append("unverified dimensions or units")
        search_query = search_query.replace(dimension.group(), " ")
    search_query = " ".join(search_query.split()) or normalized
    return QueryPlan(
        original_query=normalized,
        search_query=search_query,
        filters=ProductFilters(category, rating, material),
        unsupported_constraints=tuple(unsupported),
    )
