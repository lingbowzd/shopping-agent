"""Validated runtime choices shared by the three pipelines."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["bm25", "dense", "hybrid"] = "hybrid"
    use_metadata_filter: bool = True
    use_reranker: bool = False
    candidate_k: int = Field(default=30, ge=5, le=500)
    top_k: int = Field(default=5, ge=1, le=50)
