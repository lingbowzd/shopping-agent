"""Score full-catalog BM25 on WANDS judged candidate sets."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.data import load_wands  # noqa: E402
from shopping_agent.evaluation import evaluate_judged_candidates  # noqa: E402
from shopping_agent.retrieval import ProductSearchService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/raw/WANDS/dataset")
    parser.add_argument("--products", type=Path, default=ROOT / "data/processed/products.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/evaluations/bm25_judged.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["selection_mode"] != "full":
        parser.error("This evaluation requires a full-catalog manifest; run prepare_data.py --full")
    _, queries, labels = load_wands(args.dataset_dir)
    service = ProductSearchService(args.products)
    if len(service.ids) != manifest["source_counts"]["products"]:
        parser.error("Catalog size does not match the full-catalog manifest")
    report = evaluate_judged_candidates(service, queries, labels, args.k)
    report["source_commit"] = manifest["source_commit"]
    report["catalog_ids_sha256"] = manifest["selected_product_ids_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Judged-candidate reranking over {report['catalog_products']:,} WANDS products")
    print(f"nDCG@{args.k}: {report['ndcg_at_k']:.4f} ({report['ndcg_eligible_queries']} eligible queries)")
    print(f"MRR@{args.k}:  {report['mrr_at_k']:.4f} ({report['mrr_eligible_queries']} eligible queries; Exact only)")
    print(f"Conflicting query-product label pairs excluded: {report['label_stats']['conflicting_pairs_excluded']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
