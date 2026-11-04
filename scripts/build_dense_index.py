"""Persist normalized English product embeddings and a FAISS inner-product index."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.dense import DEFAULT_MODEL, build_dense_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=ROOT / "data/processed/products.jsonl")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/processed/indexes/minilm_l6")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/models")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    metadata = build_dense_index(args.products, args.index_dir, args.model_dir, args.model, args.batch_size)
    print(f"Saved {metadata['count']:,} vectors ({metadata['dimension']}d) to {args.index_dir}")


if __name__ == "__main__":
    main()
