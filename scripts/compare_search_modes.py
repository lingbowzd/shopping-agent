"""Compare three retrievers on one genuine WANDS query."""

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.retrieval import ProductSearchService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", type=int, default=1, help="WANDS query_id; default is 'smart coffee table'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="Also rerank hybrid candidates with Cohere")
    parser.add_argument("--products", type=Path, default=ROOT / "data/processed/products.jsonl")
    args = parser.parse_args()
    queries = pd.read_csv(ROOT / "data/raw/WANDS/dataset/query.csv", sep="\t")
    matches = queries[queries.query_id.eq(args.query_id)]
    if len(matches) != 1:
        parser.error(f"No unique WANDS query_id {args.query_id}")
    query = matches.iloc[0]["query"]
    print(f"WANDS query_id {args.query_id}: {query}")
    service = ProductSearchService(args.products)
    modes = ["bm25", "dense", "hybrid"] + (["hybrid_rerank"] if args.rerank else [])
    for mode in modes:
        print(f"\n{mode.upper()} Top {args.top_k}")
        for hit in service.search(query, mode=mode, top_k=args.top_k):
            print(f"{hit['rank']}. {hit['product_id']} | {hit['title']} | {hit['product_class']} | {hit['score_type']}={hit['score']:.4f}")
            if hit["evidence"]:
                item = hit["evidence"][0]
                print(f"   {item['field']}: {item['text']}")
    print("\nScores are only comparable within the same mode.")


if __name__ == "__main__":
    main()
