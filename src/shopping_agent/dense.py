"""Build and validate a persisted FAISS index for WANDS product text."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def catalog_id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def build_dense_index(
    products_path: Path, index_dir: Path, model_dir: Path,
    model_name: str = DEFAULT_MODEL, batch_size: int = 64,
) -> dict[str, Any]:
    """Embed one product per row and persist normalized vectors plus FAISS."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    ids = []
    catalog_hash = hashlib.sha256()
    with products_path.open("rb") as stream:
        for line in stream:
            if line.strip():
                catalog_hash.update(line)
                ids.append(json.loads(line)["product_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate product IDs")
    if not ids:
        raise ValueError("Empty catalog")
    model_dir.mkdir(parents=True, exist_ok=True)
    revision = DEFAULT_REVISION if model_name == DEFAULT_MODEL else None
    model = SentenceTransformer(model_name, cache_folder=str(model_dir), device="cpu", revision=revision)
    dimension = model.get_embedding_dimension()
    if not dimension:
        raise ValueError("Embedding model did not report a vector dimension")
    index_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = index_dir / "embeddings.npy"
    vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype="float32", shape=(len(ids), dimension))
    index = faiss.IndexFlatIP(dimension)
    batch_texts = []
    written = 0

    def flush() -> None:
        nonlocal written
        if not batch_texts:
            return
        encoded = model.encode(batch_texts, batch_size=batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        encoded = np.ascontiguousarray(encoded, dtype="float32")
        if encoded.shape != (len(batch_texts), dimension):
            raise ValueError(f"Unexpected embedding shape: {encoded.shape}")
        if not np.isfinite(encoded).all():
            raise ValueError("Embedding model returned a non-finite vector")
        vectors[written:written + len(batch_texts)] = encoded
        index.add(encoded)
        written += len(batch_texts)
        batch_texts.clear()
        if written % 2048 == 0 or written == len(ids):
            print(f"Embedded {written:,}/{len(ids):,} products", flush=True)

    with products_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                batch_texts.append(json.loads(line)["search_text"])
                if len(batch_texts) >= batch_size:
                    flush()
    flush()
    vectors.flush()
    if written != len(ids) or index.ntotal != len(ids):
        raise AssertionError("Embedding/index row count mismatch")
    faiss.write_index(index, str(index_dir / "products.faiss"))
    metadata = {
        "model_name": model_name,
        "dimension": dimension,
        "count": len(ids),
        "catalog_id_sha256": catalog_id_hash(ids),
        "catalog_file_sha256": catalog_hash.hexdigest(),
        "model_revision": revision,
        "normalized": True,
        "faiss_index": "IndexFlatIP",
        "row_order": "products.jsonl",
    }
    (index_dir / "index_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_dense_index(index_dir: Path, ids: list[str], catalog_file_sha256: str) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Reject a cache built for a different catalog or with mismatched vectors."""
    import faiss

    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dense index missing at {index_dir}; run scripts/build_dense_index.py")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata["count"] != len(ids) or metadata["catalog_id_sha256"] != catalog_id_hash(ids):
        raise ValueError("Dense index was built for a different catalog")
    if metadata.get("catalog_file_sha256") != catalog_file_sha256:
        raise ValueError("Dense index is stale: product text changed; rebuild it")
    vectors = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    index = faiss.read_index(str(index_dir / "products.faiss"))
    expected_shape = (len(ids), metadata["dimension"])
    if vectors.shape != expected_shape or index.ntotal != len(ids) or index.d != metadata["dimension"]:
        raise ValueError("Dense vector/index shape mismatch")
    if metadata.get("normalized") is not True or metadata.get("faiss_index") != "IndexFlatIP":
        raise ValueError("Dense index metadata is incompatible")
    return index, vectors, metadata
