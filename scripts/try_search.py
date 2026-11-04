"""Search the prepared local WANDS catalog with BM25."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.retrieval import ProductSearchService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search words, e.g. 'modern coffee table'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid", "hybrid_rerank"), default="bm25")
    parser.add_argument("--products", type=Path, default=ROOT / "data/processed/products.jsonl")
    args = parser.parse_args()
    service = ProductSearchService(args.products)
    results = service.search(args.query, mode=args.mode, top_k=args.top_k)
    if not results:
        print("No lexical matches in the sampled catalog.")
    for hit in results:
        print(f"{hit['rank']}. ID {hit['product_id']} | {hit['title']} | {hit['product_class']} | {hit['score_type']} {hit['score']:.3f}")
        for item in hit["evidence"]:
            print(f"   Evidence ({item['field']}): {item['text']}")
        if hit["unmatched_terms"]:
            print(f"   Query terms absent from this record: {', '.join(hit['unmatched_terms'])}")


if __name__ == "__main__":
    main()
