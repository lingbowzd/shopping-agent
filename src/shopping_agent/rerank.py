"""Optional Cohere Rerank v2 call for a small hybrid candidate set."""

from __future__ import annotations

import json
import math
import os
import ssl
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import truststore

DEFAULT_RERANK_MODEL = "rerank-v5.0-fast"


def _cohere_credentials() -> tuple[str, str]:
    """Use a dedicated key, or the project's Cohere-compatible .env settings."""
    key = os.getenv("COHERE_API_KEY")
    if key:
        return key, os.getenv("COHERE_RERANK_URL", "https://api.cohere.com/v2/rerank")
    dotenv = Path(__file__).resolve().parents[2] / ".env"
    if dotenv.exists():
        values = {}
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip('"\'')
        base = urlparse(values.get("LLM_BASE_URL", ""))
        if (base.scheme == "https" and base.hostname and
                (base.hostname == "api.cohere.ai" or base.hostname.endswith(".api.cohere.ai")) and
                values.get("LLM_API_KEY")):
            return values["LLM_API_KEY"], f"{base.scheme}://{base.netloc}/v2/rerank"
    raise RuntimeError("Set COHERE_API_KEY, or provide Cohere LLM_API_KEY and LLM_BASE_URL in .env")


def cohere_rerank(
    query: str, documents: list[str], top_n: int,
    model: str | None = None,
) -> list[tuple[int, float]]:
    """Return (input index, relevance score) in Cohere's ranking order."""
    key, endpoint = _cohere_credentials()
    if not documents or top_n < 1:
        return []
    payload = json.dumps({
        "model": model or os.getenv("COHERE_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }).encode("utf-8")
    request = Request(
        endpoint, data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60, context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Cohere rerank returned HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach Cohere rerank: {error.reason}") from error
    ranked = []
    seen = set()
    for item in result.get("results", []):
        index = item["index"]
        score = float(item["relevance_score"])
        if not isinstance(index, int) or not 0 <= index < len(documents) or index in seen or not math.isfinite(score):
            raise ValueError("Invalid Cohere rerank response")
        ranked.append((index, score))
        seen.add(index)
    if len(ranked) != min(top_n, len(documents)):
        raise ValueError("Cohere returned an unexpected number of results")
    return ranked
