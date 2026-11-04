"""Print actual WANDS TSV schemas and source category frequencies."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.data import load_wands  # noqa: E402


def main() -> None:
    products, queries, labels = load_wands(ROOT / "data/raw/WANDS/dataset")
    for name, frame in (("product.csv", products), ("query.csv", queries), ("label.csv", labels)):
        print(f"{name}: {len(frame):,} rows")
        print(f"  columns: {frame.columns.tolist()}")
        print(f"  missing: {frame.isna().sum().to_dict()}")
    print("\nTop 30 product classes:")
    print(products.product_class.value_counts(dropna=False).head(30).to_string())
    print("\nLabel values:")
    print(labels.label.value_counts().to_string())


if __name__ == "__main__":
    main()
