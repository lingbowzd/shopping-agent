"""Explicit judged-candidate ranking metrics for WANDS labels."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any

import pandas as pd

from .retrieval import ProductSearchService

RELEVANCE = {"Irrelevant": 0, "Partial": 1, "Exact": 2}


def consolidate_labels(labels: pd.DataFrame) -> tuple[dict[int, dict[str, str]], dict[str, int]]:
    """Collapse identical duplicates and exclude pairs with conflicting labels."""
    labels_by_pair: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in labels.itertuples(index=False):
        if row.label not in RELEVANCE:
            raise ValueError(f"Unexpected WANDS label: {row.label}")
        labels_by_pair[(int(row.query_id), str(row.product_id))].add(row.label)
    qrels: dict[int, dict[str, str]] = defaultdict(dict)
    conflicts = 0
    for (query_id, product_id), values in labels_by_pair.items():
        if len(values) != 1:
            conflicts += 1
            continue
        qrels[query_id][product_id] = next(iter(values))
    return dict(qrels), {
        "source_label_rows": len(labels),
        "unique_query_product_pairs": len(labels_by_pair),
        "duplicate_label_rows": len(labels) - len(labels_by_pair),
        "conflicting_pairs_excluded": conflicts,
        "scored_pairs": len(labels_by_pair) - conflicts,
    }


def dcg_at_k(relevance: list[int], k: int) -> float:
    return sum((2 ** level - 1) / math.log2(rank + 1) for rank, level in enumerate(relevance[:k], 1))


def query_metrics(ranked_labels: list[str], all_labels: list[str], k: int) -> tuple[float | None, float | None]:
    """nDCG uses graded relevance; MRR treats Exact as the sole hit."""
    if k < 1:
        raise ValueError("k must be positive")
    ordered = [RELEVANCE[label] for label in ranked_labels]
    ideal = sorted((RELEVANCE[label] for label in all_labels), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    ndcg = dcg_at_k(ordered, k) / ideal_dcg if ideal_dcg else None
    if "Exact" not in all_labels:
        rr = None
    else:
        rr = next((1 / rank for rank, label in enumerate(ranked_labels[:k], 1) if label == "Exact"), 0.0)
    return ndcg, rr


def evaluate_judged_candidates(
    service: ProductSearchService, queries: pd.DataFrame, labels: pd.DataFrame, k: int = 10,
    mode: str = "bm25",
) -> dict[str, Any]:
    """Evaluate one retriever within each query's explicitly judged products."""
    if k < 1:
        raise ValueError("k must be positive")
    qrels, label_stats = consolidate_labels(labels)
    per_query = []
    for row in queries.itertuples(index=False):
        judged = qrels.get(int(row.query_id), {})
        ranked = service.rank_candidates(row.query, list(judged), mode=mode) if judged else []
        ranked_labels = [judged[product_id] for product_id, _ in ranked]
        ndcg, rr = query_metrics(ranked_labels, list(judged.values()), k)
        per_query.append({
            "query_id": int(row.query_id),
            "query": row.query,
            "judged_candidates": len(judged),
            "ndcg_at_k": ndcg,
            "reciprocal_rank_at_k": rr,
            "top_k": [
                {"product_id": product_id, "label": judged[product_id], "score": round(score, 6)}
                for product_id, score in ranked[:k]
            ],
        })
    ndcgs = [item["ndcg_at_k"] for item in per_query if item["ndcg_at_k"] is not None]
    reciprocal_ranks = [item["reciprocal_rank_at_k"] for item in per_query if item["reciprocal_rank_at_k"] is not None]
    return {
        "evaluation_type": "judged_candidate_reranking",
        "mode": mode,
        "catalog_products": len(service.ids),
        "k": k,
        "label_policy": {
            "Irrelevant": 0, "Partial": 1, "Exact": 2,
            "dcg_gain": "2^relevance - 1",
            "mrr_positive_label": "Exact",
            "unjudged_products": "excluded from candidate set; never treated as Irrelevant",
            "conflicting_pairs": "excluded",
            "zero_score_tie_break": "ascending numeric product_id",
        },
        "label_stats": label_stats,
        "query_count": len(per_query),
        "ndcg_at_k": mean(ndcgs) if ndcgs else None,
        "ndcg_eligible_queries": len(ndcgs),
        "ndcg_excluded_no_positive": len(per_query) - len(ndcgs),
        "mrr_at_k": mean(reciprocal_ranks) if reciprocal_ranks else None,
        "mrr_eligible_queries": len(reciprocal_ranks),
        "mrr_excluded_no_exact": len(per_query) - len(reciprocal_ranks),
        "per_query": per_query,
    }
