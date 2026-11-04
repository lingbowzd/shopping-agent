"""Validate and prepare the official tab-separated WANDS files."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

SOURCE_URL = "https://github.com/wayfair/WANDS"
PREPROCESSING_VERSION = "stage1a-2"
PRODUCT_COLUMNS = [
    "product_id", "product_name", "product_class", "category hierarchy",
    "product_description", "product_features", "rating_count",
    "average_rating", "review_count",
]
QUERY_COLUMNS = ["query_id", "query", "query_class"]
LABEL_COLUMNS = ["id", "query_id", "product_id", "label"]
DEFAULT_CATEGORIES = [
    "Coffee & Cocktail Tables", "Accent Chairs", "Desks", "Bookcases",
    "Table Lamps", "End Tables", "Dining Tables", "Beds", "Area Rugs",
    "Sofas", "Chandeliers",
]


def load_wands(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all three official .csv files as TSV and fail on schema drift."""
    frames = tuple(
        pd.read_csv(dataset_dir / filename, sep="\t")
        for filename in ("product.csv", "query.csv", "label.csv")
    )
    for filename, frame, columns in zip(
        ("product.csv", "query.csv", "label.csv"),
        frames,
        (PRODUCT_COLUMNS, QUERY_COLUMNS, LABEL_COLUMNS),
    ):
        if frame.columns.tolist() != columns:
            raise ValueError(f"Unexpected {filename} columns: {frame.columns.tolist()}")
    products, queries, labels = frames
    if products.product_id.isna().any() or not products.product_id.is_unique:
        raise ValueError("WANDS product IDs must be present and unique")
    if queries.query_id.isna().any() or not queries.query_id.is_unique:
        raise ValueError("WANDS query IDs must be present and unique")
    if not labels.product_id.isin(products.product_id).all():
        raise ValueError("Labels reference unknown product IDs")
    if not labels.query_id.isin(queries.query_id).all():
        raise ValueError("Labels reference unknown query IDs")
    return products, queries, labels


def clean(value: Any) -> Any:
    """Turn pandas missing values and blank strings into JSON nulls."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_features(raw: str | None) -> tuple[dict[str, list[str]], int]:
    """Keep duplicate keys; count segments that lack a usable key/value pair."""
    features: dict[str, list[str]] = {}
    malformed = 0
    if not raw:
        return features, malformed
    for segment in raw.split("|"):
        if ":" not in segment:
            if segment.strip():
                malformed += 1
            continue
        key, value = (part.strip() for part in segment.split(":", 1))
        if not key or not value:
            malformed += 1
            continue
        features.setdefault(key, []).append(value)
    return features, malformed


def category_quotas(available: dict[str, int], target: int) -> dict[str, int]:
    """Allocate one product at a time to the least represented eligible class."""
    quotas = {category: 0 for category in available}
    if sum(available.values()) < target:
        raise ValueError("Selected categories do not contain enough products")
    for _ in range(target):
        eligible = (category for category in available if quotas[category] < available[category])
        category = min(eligible, key=lambda name: (quotas[name], list(available).index(name)))
        quotas[category] += 1
    return quotas


def select_products(
    products: pd.DataFrame, categories: list[str] = DEFAULT_CATEGORIES,
    target: int = 3000, seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if target < 1:
        raise ValueError("target must be positive")
    if len(categories) != len(set(categories)):
        raise ValueError("Duplicate category names")
    eligible = products[products.product_name.notna() & products.product_name.astype(str).str.strip().ne("")]
    available = {name: int(eligible.product_class.eq(name).sum()) for name in categories}
    if any(count == 0 for count in available.values()):
        raise ValueError(f"Missing source categories: {[k for k, v in available.items() if v == 0]}")
    quotas = category_quotas(available, target)
    picked = []
    for index, name in enumerate(categories):
        group = eligible[eligible.product_class.eq(name)].sort_values("product_id")
        sparse = group.product_description.isna() | group.product_description.astype(str).str.strip().eq("")
        sparse |= group.product_features.isna() | group.product_features.astype(str).str.strip().eq("")
        sparse_target = min(int(sparse.sum()), round(quotas[name] * 0.10))
        sparse_rows = group[sparse].sample(n=sparse_target, random_state=seed + index) if sparse_target else group.iloc[:0]
        rich_rows = group[~sparse]
        rich_target = min(len(rich_rows), quotas[name] - sparse_target)
        rich_sample = rich_rows.sample(n=rich_target, random_state=seed + 1000 + index)
        shortfall = quotas[name] - sparse_target - rich_target
        extra_sparse = group[sparse & ~group.product_id.isin(sparse_rows.product_id)]
        extra_sample = extra_sparse.sample(n=shortfall, random_state=seed + 2000 + index) if shortfall else group.iloc[:0]
        picked.append(pd.concat((sparse_rows, rich_sample, extra_sample)))
    selected = pd.concat(picked).sort_values("product_id").reset_index(drop=True)
    if len(selected) != target or not selected.product_id.is_unique:
        raise AssertionError("Sampling failed to produce unique target count")
    return selected, quotas


def to_record(row: pd.Series) -> tuple[dict[str, Any], int]:
    raw = {column: clean(row[column]) for column in PRODUCT_COLUMNS}
    features, malformed = parse_features(raw["product_features"])
    searchable_features = " ".join(
        f"{key} {' '.join(values)}" for key, values in features.items()
    )
    search_text = " ".join(str(part) for part in (
        raw["product_name"], raw["product_class"], raw["category hierarchy"],
        (raw["product_description"] or "")[:1500], searchable_features[:1500],
    ) if part)
    record = {
        "product_id": str(raw["product_id"]),
        "title": raw["product_name"],
        "product_class": raw["product_class"],
        "category_hierarchy": raw["category hierarchy"],
        "description": raw["product_description"],
        "raw_features": raw["product_features"],
        "features": features,
        "rating_count": raw["rating_count"],
        "average_rating": raw["average_rating"],
        "review_count": raw["review_count"],
        "search_text": search_text,
        "raw": raw,
    }
    return record, malformed


def prepare(
    dataset_dir: Path, output_dir: Path, target: int = 3000, seed: int = 42,
    full: bool = False,
) -> dict[str, Any]:
    products, queries, labels = load_wands(dataset_dir)
    if full:
        selected = products.sort_values("product_id").reset_index(drop=True)
        categories = sorted(str(name) for name in selected.product_class.dropna().unique())
    else:
        selected, quotas = select_products(products, target=target, seed=seed)
        categories = list(quotas)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    malformed_total = 0
    for _, row in selected.iterrows():
        record, malformed = to_record(row)
        records.append(record)
        malformed_total += malformed
    products_path = output_dir / "products.jsonl"
    with products_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(dataset_dir.parent), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    source_sha256 = hashlib.sha256((dataset_dir / "product.csv").read_bytes()).hexdigest()
    ids_sha256 = hashlib.sha256("\n".join(r["product_id"] for r in records).encode()).hexdigest()
    missing = {column: int(selected[column].isna().sum()) for column in PRODUCT_COLUMNS}
    category_counts = dict(Counter(r["product_class"] or "<missing>" for r in records))
    sample_queries = [
        {"query_id": int(row.query_id), "query": row.query, "query_class": row.query_class}
        for row in queries[
            queries.query_class.isin(DEFAULT_CATEGORIES) if not full else queries.query_id.notna()
        ].head(12).itertuples(index=False)
    ]
    manifest = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "selection_mode": "full" if full else "category_sample",
        "source_url": SOURCE_URL,
        "source_commit": source_commit,
        "source_product_sha256": source_sha256,
        "seed": None if full else seed,
        "target_count": len(products) if full else target,
        "sampled_count": len(records),
        "selected_categories": categories,
        "category_counts": category_counts,
        "selected_product_ids_sha256": ids_sha256,
        "source_counts": {"products": len(products), "queries": len(queries), "labels": len(labels)},
        "source_columns": {"product.csv": PRODUCT_COLUMNS, "query.csv": QUERY_COLUMNS, "label.csv": LABEL_COLUMNS},
        "sample_missing_fields": missing,
        "sample_malformed_feature_segments": malformed_total,
        "real_sample_queries": sample_queries,
        "note": "Labels are validated but not used for sampling or search.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
