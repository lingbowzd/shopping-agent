"""Small, dependency-free BM25 index over prepared WANDS product records."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from .dense import load_dense_index
from .query_understanding import ProductFilters
from .rerank import cohere_rerank

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
RRF_K = 60
FUSION_DEPTH = 30
MATERIAL_FIELDS = {
    "material", "primarymaterial", "framematerial", "topmaterial", "basematerial",
    "legmaterial", "upholsterymaterial", "seatmaterial", "tabletopmaterial",
    "shadematerial", "covermaterial",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def material_tags(features: dict[str, list[str]]) -> set[str]:
    """Recognize material terms only in explicit material-valued feature fields."""
    tags = set()
    for key, values in features.items():
        if re.sub(r"[^a-z]", "", key.casefold()) not in MATERIAL_FIELDS:
            continue
        for value in values if isinstance(values, list) else []:
            text = str(value).casefold()
            if re.search(r"\bsolid\s+wood\b", text):
                tags.add("solid wood")
            if re.search(r"\bwood\b", text):
                tags.add("wood")
            for term in ("metal", "glass", "velvet"):
                if re.search(rf"\b{term}\b", text):
                    tags.add(term)
            if re.search(r"\bleather\b", text) and not re.search(r"\b(?:faux|synthetic|vegan)\s+leather\b", text):
                tags.add("leather")
    return tags


def evidence(record: dict[str, Any], terms: set[str]) -> list[dict[str, str]]:
    """Show literal source excerpts for matched query terms."""
    candidates = []
    for field in ("title", "product_class", "category_hierarchy", "description", "raw_features"):
        value = record.get(field) or ""
        if not isinstance(value, str):
            continue
        matches = [match for match in TOKEN_RE.finditer(value) if match.group().casefold() in terms]
        if matches:
            candidates.append((field, value, matches))
    if not candidates:
        return [{"field": "title", "text": record["title"]}]
    snippets = []
    seen = set()
    for term in sorted(terms):
        found = next(
            ((field, value, match) for field, value, matches in candidates
             for match in matches if match.group().casefold() == term),
            None,
        )
        if found is None:
            continue
        field, value, match = found
        start = max(0, match.start() - 75)
        end = min(len(value), match.end() + 125)
        excerpt = ("…" if start else "") + value[start:end].strip() + ("…" if end < len(value) else "")
        if (field, excerpt) not in seen:
            snippets.append({"field": field, "text": excerpt})
            seen.add((field, excerpt))
    return snippets


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion; duplicate IDs receive contributions from both lists."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1 / (k + rank)
    return dict(scores)


class ProductSearchService:
    def __init__(
        self, products_path: Path, k1: float = 1.5, b: float = 0.75,
        index_dir: Path | None = None, model_dir: Path | None = None,
    ):
        self.products_path = products_path
        self.index_dir = index_dir or products_path.parent / "indexes/minilm_l6"
        self.model_dir = model_dir or products_path.parent.parent / "models"
        self.dense_index = None
        self.dense_vectors = None
        self.dense_metadata = None
        self.embedding_model = None
        self.query_vector_cache: dict[str, np.ndarray] = {}
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        self.ids: list[str] = []
        self.id_to_index: dict[str, int] = {}
        self.offsets: list[int] = []
        self.class_postings: dict[str, set[int]] = defaultdict(set)
        self.material_postings: dict[str, set[int]] = defaultdict(set)
        self.historical_ratings: list[float | None] = []
        seen_ids = set()
        catalog_hash = hashlib.sha256()
        with products_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                catalog_hash.update(line)
                record = json.loads(line)
                product_id = record["product_id"]
                if product_id in seen_ids:
                    raise ValueError("Duplicate product IDs in processed catalog")
                seen_ids.add(product_id)
                index = len(self.ids)
                self.ids.append(product_id)
                self.id_to_index[product_id] = index
                self.offsets.append(offset)
                for name in str(record.get("product_class") or "").split("|"):
                    if name.strip():
                        self.class_postings[name.strip()].add(index)
                for material in material_tags(record.get("features") or {}):
                    self.material_postings[material].add(index)
                rating = record.get("average_rating")
                rating_count = record.get("rating_count")
                try:
                    value = float(rating)
                    count = float(rating_count or 0)
                    self.historical_ratings.append(value if count > 0 and math.isfinite(value)
                                                   and 0 <= value <= 5 else None)
                except (TypeError, ValueError):
                    self.historical_ratings.append(None)
                counts = Counter(tokenize(record["search_text"]))
                self.lengths.append(sum(counts.values()))
                for term, frequency in counts.items():
                    self.postings[term].append((index, frequency))
        self.catalog_file_sha256 = catalog_hash.hexdigest()
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0

    def _filter_candidates(self, filters: ProductFilters | None) -> set[int] | None:
        if filters is None or filters.is_empty():
            return None
        candidates: set[int] | None = None
        if filters.product_class:
            candidates = set(self.class_postings.get(filters.product_class, set()))
        if filters.material:
            matching = self.material_postings.get(filters.material, set())
            candidates = matching & candidates if candidates is not None else set(matching)
        if filters.min_historical_rating is not None:
            threshold = filters.min_historical_rating
            if not 0 <= threshold <= 5:
                raise ValueError("Historical rating filter must be between 0 and 5")
            universe = candidates if candidates is not None else range(len(self.ids))
            candidates = {index for index in universe if self.historical_ratings[index] is not None
                          and self.historical_ratings[index] >= threshold}
        return candidates

    def eligible_count(self, filters: ProductFilters | None) -> int:
        candidates = self._filter_candidates(filters)
        return len(self.ids) if candidates is None else len(candidates)

    def _scores(self, query: str, candidates: set[int] | None = None) -> dict[int, float]:
        terms = set(tokenize(query))
        if not terms or not self.ids:
            return {}
        scores: dict[int, float] = defaultdict(float)
        n = len(self.ids)
        for term in terms:
            matches = self.postings.get(term, [])
            if not matches:
                continue
            idf = math.log(1 + (n - len(matches) + 0.5) / (len(matches) + 0.5))
            for index, frequency in matches:
                if candidates is not None and index not in candidates:
                    continue
                normalization = 1 - self.b + self.b * self.lengths[index] / self.average_length
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * normalization)
        return scores

    def _load_dense(self) -> None:
        if self.dense_index is None:
            self.dense_index, self.dense_vectors, self.dense_metadata = load_dense_index(
                self.index_dir, self.ids, self.catalog_file_sha256
            )

    def _query_vector(self, query: str) -> np.ndarray:
        if query in self.query_vector_cache:
            return self.query_vector_cache[query]
        self._load_dense()
        if self.embedding_model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(1)
            self.embedding_model = SentenceTransformer(
                self.dense_metadata["model_name"], cache_folder=str(self.model_dir),
                device="cpu", local_files_only=True,
                revision=self.dense_metadata.get("model_revision"),
            )
        vector = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        vector = np.ascontiguousarray(vector, dtype="float32")
        self.query_vector_cache[query] = vector
        return vector

    def prefetch_query_vectors(self, queries: list[str], batch_size: int = 64) -> None:
        """Encode evaluation queries once so dense and hybrid reuse identical vectors."""
        self._load_dense()
        if self.embedding_model is None:
            self._query_vector(queries[0])
        missing = list(dict.fromkeys(query for query in queries if query not in self.query_vector_cache))
        if missing:
            vectors = self.embedding_model.encode(
                missing, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            for query, vector in zip(missing, vectors):
                self.query_vector_cache[query] = np.ascontiguousarray(vector.reshape(1, -1), dtype="float32")

    def _bm25_ranking(self, query: str, candidates: set[int] | None = None) -> list[tuple[int, float]]:
        scores = self._scores(query, candidates)
        indexes = candidates if candidates is not None else scores.keys()
        return [(index, scores.get(index, 0.0)) for index in sorted(
            indexes, key=lambda index: (-scores.get(index, 0.0), int(self.ids[index]))
        )]

    def _dense_ranking(self, query: str, candidates: set[int] | None = None,
                       top_k: int | None = None) -> list[tuple[int, float]]:
        if candidates is not None and not candidates:
            return []
        vector = self._query_vector(query)
        if candidates is None:
            count = min(top_k or len(self.ids), len(self.ids))
            scores, indexes = self.dense_index.search(vector, count)
            return [(int(index), float(score)) for index, score in zip(indexes[0], scores[0]) if index >= 0]
        indexes = np.array(sorted(candidates), dtype=np.int64)
        if not len(indexes):
            return []
        scores = np.asarray(self.dense_vectors[indexes]) @ vector[0]
        ranked = sorted(zip(indexes.tolist(), scores.tolist()),
                        key=lambda item: (-item[1], int(self.ids[item[0]])))
        return ranked[:top_k] if top_k else ranked

    def rank_candidates(self, query: str, product_ids: list[str],
                        mode: str = "bm25") -> list[tuple[str, float]]:
        """Rank every judged candidate, including zero-score products."""
        unknown = set(product_ids) - self.id_to_index.keys()
        if unknown:
            raise ValueError(f"Unknown candidate product IDs: {sorted(unknown)[:5]}")
        candidates = {self.id_to_index[product_id] for product_id in product_ids}
        if mode == "bm25":
            ranked = self._bm25_ranking(query, candidates)
        elif mode == "dense":
            ranked = self._dense_ranking(query, candidates)
        elif mode in {"hybrid", "hybrid_rerank"}:
            bm25 = self._bm25_ranking(query, candidates)[:FUSION_DEPTH]
            dense = self._dense_ranking(query, candidates, FUSION_DEPTH)
            scores = rrf_fuse([[index for index, _ in bm25], [index for index, _ in dense]])
            fused = [(index, scores.get(index, 0.0)) for index in sorted(
                candidates, key=lambda index: (-scores.get(index, 0.0), int(self.ids[index]))
            )]
            if mode == "hybrid_rerank":
                selected = fused[:len(scores)]
                with self.products_path.open("rb") as stream:
                    documents = []
                    for index, _ in selected:
                        stream.seek(self.offsets[index])
                        documents.append(json.loads(stream.readline())["search_text"][:2000])
                reranked = cohere_rerank(query, documents, len(documents))
                ranked = [(selected[position][0], score) for position, score in reranked] + fused[len(scores):]
            else:
                ranked = fused
        else:
            raise ValueError(f"Unsupported candidate-ranking mode: {mode}")
        return [(self.ids[index], score) for index, score in ranked]

    def get_products(self, product_ids: list[str], query: str,
                     filters: ProductFilters | None = None) -> list[dict[str, Any]]:
        """One direct catalog batch read for explicitly supplied product IDs."""
        eligible = self._filter_candidates(filters)
        terms = set(tokenize(query))
        hits = []
        with self.products_path.open("rb") as stream:
            for product_id in dict.fromkeys(product_ids):
                index = self.id_to_index.get(product_id)
                if index is None or (eligible is not None and index not in eligible):
                    continue
                stream.seek(self.offsets[index])
                record = json.loads(stream.readline())
                hits.append({
                    "rank": len(hits) + 1, "product_id": product_id,
                    "title": record["title"], "product_class": record.get("product_class"),
                    "features": record.get("features") or {},
                    "average_rating": record.get("average_rating"),
                    "rating_count": record.get("rating_count"),
                    "review_count": record.get("review_count"),
                    "evidence": evidence(record, terms),
                    "score": None, "score_type": "direct_lookup",
                })
        return hits

    def search(self, query: str, mode: str = "bm25", top_k: int = 5,
               filters: ProductFilters | None = None, rerank: bool = False,
               candidate_k: int = 30,
               stage_trace: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if mode not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
            raise ValueError(f"Unsupported search mode: {mode}")
        rerank = rerank or mode == "hybrid_rerank"
        if candidate_k > 500 or (rerank and candidate_k < top_k):
            raise ValueError("candidate_k must be between top_k and 500 when reranking")
        backend = "hybrid" if mode == "hybrid_rerank" else mode
        terms = set(tokenize(query))
        if not terms:
            return []
        candidates = self._filter_candidates(filters)
        if stage_trace is not None:
            stage_trace.update({
                "catalog_count": len(self.ids),
                "eligible_count": len(self.ids) if candidates is None else len(candidates),
                "backend": backend, "rerank": rerank, "candidate_k": candidate_k,
            })
        if candidates is not None and not candidates:
            return []
        depth = candidate_k if rerank else top_k
        if backend == "bm25":
            ranked = self._bm25_ranking(query, candidates)
            ranked = [(index, score) for index, score in ranked if score > 0][:depth]
            if stage_trace is not None:
                stage_trace["bm25_ids"] = [self.ids[index] for index, _ in ranked]
        elif backend == "dense":
            ranked = self._dense_ranking(query, candidates, top_k=depth)
            if stage_trace is not None:
                stage_trace["dense_ids"] = [self.ids[index] for index, _ in ranked]
        else:
            fusion_depth = max(FUSION_DEPTH, depth)
            bm25 = [(index, score) for index, score in self._bm25_ranking(query, candidates) if score > 0][:fusion_depth]
            dense = self._dense_ranking(query, candidates, top_k=fusion_depth)
            fused = rrf_fuse([[index for index, _ in bm25], [index for index, _ in dense]])
            ranked = sorted(fused.items(), key=lambda item: (-item[1], int(self.ids[item[0]])))
            ranked = ranked[:depth]
            if stage_trace is not None:
                stage_trace["bm25_ids"] = [self.ids[index] for index, _ in bm25]
                stage_trace["dense_ids"] = [self.ids[index] for index, _ in dense]
                stage_trace["rrf_ids"] = [self.ids[index] for index, _ in ranked]
        if stage_trace is not None:
            stage_trace["candidate_ids_before_rerank"] = [self.ids[index] for index, _ in ranked]
        if rerank and ranked:
            with self.products_path.open("rb") as stream:
                documents = []
                for index, _ in ranked:
                    stream.seek(self.offsets[index])
                    documents.append(json.loads(stream.readline())["search_text"][:2000])
            reranked = cohere_rerank(query, documents, top_k)
            ranked = [(ranked[position][0], score) for position, score in reranked]
            if stage_trace is not None:
                stage_trace["reranked_ids"] = [self.ids[index] for index, _ in ranked]
        ranked = ranked[:top_k]
        if stage_trace is not None:
            stage_trace["final_ids"] = [self.ids[index] for index, _ in ranked]
        hits = []
        with self.products_path.open("rb") as stream:
            for rank, (index, score) in enumerate(ranked, 1):
                stream.seek(self.offsets[index])
                record = json.loads(stream.readline())
                matched = terms & set(tokenize(record["search_text"]))
                filter_evidence = []
                if filters is not None:
                    if filters.product_class:
                        filter_evidence.append({"field": "product_class", "value": filters.product_class})
                    if filters.min_historical_rating is not None:
                        filter_evidence.append({"field": "historical_average_rating",
                                                "value": record.get("average_rating")})
                        filter_evidence.append({"field": "historical_rating_count",
                                                "value": record.get("rating_count")})
                    if filters.material:
                        for key, values in (record.get("features") or {}).items():
                            if filters.material in material_tags({key: values}):
                                filter_evidence.extend({"field": f"features.{key}", "value": value}
                                                       for value in values)
                hits.append({
                    "rank": rank,
                    "product_id": record["product_id"],
                    "title": record["title"],
                    "product_class": record["product_class"],
                    "features": record.get("features") or {},
                    "average_rating": record.get("average_rating"),
                    "rating_count": record.get("rating_count"),
                    "review_count": record.get("review_count"),
                    "score": round(score, 6),
                    "score_type": "Cohere_relevance" if rerank else
                                  {"bm25": "BM25", "dense": "cosine", "hybrid": "RRF"}[backend],
                    "evidence": evidence(record, terms),
                    "filter_evidence": filter_evidence,
                    "matched_terms": sorted(matched),
                    "unmatched_terms": sorted(terms - matched),
                })
        return hits
