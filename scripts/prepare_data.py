"""Create the fixed-seed Stage 1A catalog."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.data import prepare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/raw/WANDS/dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--target", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Include every WANDS product; seed and target are ignored")
    args = parser.parse_args()
    manifest = prepare(args.dataset_dir, args.output_dir, args.target, args.seed, args.full)
    print(f"Saved {manifest['sampled_count']:,} products to {args.output_dir / 'products.jsonl'}")
    print(f"Selection mode: {manifest['selection_mode']}; classes: {len(manifest['selected_categories']):,}")
    print("Largest category counts:", dict(sorted(
        manifest["category_counts"].items(), key=lambda item: (-item[1], item[0])
    )[:12]))
    print("Selected IDs SHA256:", manifest["selected_product_ids_sha256"])


if __name__ == "__main__":
    main()
