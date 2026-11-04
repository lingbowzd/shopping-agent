"""Compare BM25, dense, and hybrid on WANDS judged candidate sets."""

import argparse
import json
import os
from statistics import mean
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.data import load_wands  # noqa: E402
from shopping_agent.evaluation import consolidate_labels, evaluate_judged_candidates, query_metrics  # noqa: E402
from shopping_agent.retrieval import ProductSearchService  # noqa: E402
from shopping_agent.rerank import DEFAULT_RERANK_MODEL  # noqa: E402


def evaluate_cohere_with_checkpoint(service, subset, labels, k, checkpoint, expected, template):
    """Score each query once and persist progress after every Cohere response."""
    qrels, _ = consolidate_labels(labels)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved["run"] != expected:
            raise ValueError(f"Cohere checkpoint belongs to another run: {checkpoint}")
        completed = {int(key): value for key, value in saved["per_query"].items()}
    else:
        completed = {}

    for row in subset.itertuples(index=False):
        query_id = int(row.query_id)
        if query_id in completed:
            continue
        judged = qrels.get(query_id, {})
        for attempt in range(5):
            try:
                ranked = service.rank_candidates(row.query, list(judged), mode="hybrid_rerank") if judged else []
                break
            except RuntimeError as error:
                if attempt == 4 or not any(f"HTTP {code}" in str(error) for code in (429, 500, 502, 503, 504)):
                    raise
                delay = min(5 * 2 ** attempt, 30)
                print(f"Cohere retry for query_id {query_id} in {delay}s: {error}", flush=True)
                time.sleep(delay)
        ranked_labels = [judged[product_id] for product_id, _ in ranked]
        ndcg, rr = query_metrics(ranked_labels, list(judged.values()), k)
        completed[query_id] = {
            "query_id": query_id,
            "query": row.query,
            "judged_candidates": len(judged),
            "ndcg_at_k": ndcg,
            "reciprocal_rank_at_k": rr,
            "top_k": [
                {"product_id": product_id, "label": judged[product_id], "score": round(score, 6)}
                for product_id, score in ranked[:k]
            ],
        }
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps({"run": expected, "per_query": completed}, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(checkpoint)
        if len(completed) % 10 == 0 or len(completed) == len(subset):
            print(f"Cohere recorded {len(completed)}/{len(subset)} queries in {checkpoint}", flush=True)

    per_query = [completed[int(query_id)] for query_id in subset["query_id"]]
    ndcgs = [item["ndcg_at_k"] for item in per_query if item["ndcg_at_k"] is not None]
    reciprocal_ranks = [item["reciprocal_rank_at_k"] for item in per_query if item["reciprocal_rank_at_k"] is not None]
    return {
        **template,
        "mode": "hybrid_rerank",
        "ndcg_at_k": mean(ndcgs) if ndcgs else None,
        "mrr_at_k": mean(reciprocal_ranks) if reciprocal_ranks else None,
        "ndcg_eligible_queries": len(ndcgs),
        "mrr_eligible_queries": len(reciprocal_ranks),
        "ndcg_excluded_no_positive": len(per_query) - len(ndcgs),
        "mrr_excluded_no_exact": len(per_query) - len(reciprocal_ranks),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cohere-queries", type=int, default=0,
                        help="Also evaluate Cohere reranking on this many seeded WANDS queries (API calls)")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/evaluations/retrievers_judged.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/processed/manifest.json").read_text())
    if manifest["selection_mode"] != "full":
        parser.error("Full-catalog products required; run prepare_data.py --full")
    _, queries, labels = load_wands(ROOT / "data/raw/WANDS/dataset")
    service = ProductSearchService(ROOT / "data/processed/products.jsonl")
    if len(service.ids) != manifest["source_counts"]["products"]:
        parser.error("Active catalog count differs from manifest")
    service.prefetch_query_vectors(queries["query"].tolist())
    reports = {}
    for mode in ("bm25", "dense", "hybrid"):
        report = evaluate_judged_candidates(service, queries, labels, k=args.k, mode=mode)
        reports[mode] = report
        print(f"{mode:7} nDCG@{args.k}={report['ndcg_at_k']:.4f} ({report['ndcg_eligible_queries']} queries)  "
              f"MRR@{args.k}={report['mrr_at_k']:.4f} ({report['mrr_eligible_queries']} queries)", flush=True)
    cohere_sample = None
    cohere_checkpoint = None
    if args.cohere_queries:
        if not 1 <= args.cohere_queries <= len(queries):
            parser.error(f"--cohere-queries must be between 1 and {len(queries)}")
        subset = queries.sample(n=args.cohere_queries, random_state=42).sort_values("query_id")
        scope = "full" if len(subset) == len(queries) else "sample"
        cohere_sample = {
            "query_ids": subset["query_id"].astype(int).tolist(),
            "reports": {},
        }
        for mode in ("bm25", "dense", "hybrid"):
            report = evaluate_judged_candidates(service, subset, labels, k=args.k, mode=mode)
            cohere_sample["reports"][mode] = report
            print(f"{scope} {mode:13} nDCG@{args.k}={report['ndcg_at_k']:.4f} "
                  f"MRR@{args.k}={report['mrr_at_k']:.4f}", flush=True)
        checkpoint = args.output.parent / f"cohere_judged_{len(subset)}_k{args.k}.checkpoint.json"
        cohere_checkpoint = str(checkpoint)
        expected = {
            "source_commit": manifest["source_commit"],
            "catalog_file_sha256": service.catalog_file_sha256,
            "model": os.getenv("COHERE_RERANK_MODEL", DEFAULT_RERANK_MODEL),
            "k": args.k,
            "query_ids": cohere_sample["query_ids"],
        }
        report = evaluate_cohere_with_checkpoint(
            service, subset, labels, args.k, checkpoint, expected, cohere_sample["reports"]["hybrid"]
        )
        cohere_sample["reports"]["hybrid_rerank"] = report
        print(f"{scope} {'hybrid_rerank':13} nDCG@{args.k}={report['ndcg_at_k']:.4f} "
              f"MRR@{args.k}={report['mrr_at_k']:.4f}", flush=True)
        if scope == "full":
            reports["hybrid_rerank"] = report
            cohere_sample = None
    output = {
        "source_commit": manifest["source_commit"],
        "catalog_ids_sha256": manifest["selected_product_ids_sha256"],
        "note": "Judged-candidate reranking only; not full-catalog retrieval evaluation.",
        "reports": reports,
        "cohere_sample": cohere_sample,
        "cohere_checkpoint": cohere_checkpoint,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
