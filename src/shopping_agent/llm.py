"""Configured, tool-free chat completions for answer and intent generation."""

from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import truststore


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: dict
    finish_reason: str | None = None


def _config() -> tuple[str, str, str]:
    """Environment variables take precedence over the project's local .env."""
    dotenv = Path(__file__).resolve().parents[2] / ".env"
    values: dict[str, str] = {}
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip('"\'')
    model = os.getenv("LLM_MODEL_ID") or values.get("LLM_MODEL_ID")
    key = os.getenv("LLM_API_KEY") or values.get("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or values.get("LLM_BASE_URL")
    if not model or not key or not base_url:
        raise RuntimeError("Set LLM_MODEL_ID, LLM_API_KEY, and LLM_BASE_URL")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}):
        raise ValueError("LLM_BASE_URL must use HTTPS, except for localhost")
    return model, key, base_url.rstrip("/")


class ConfiguredLLM:
    """OpenAI-compatible chat endpoint; deliberately supplies no tools."""

    def __init__(self, max_tokens: int = 2500) -> None:
        self.model, self._key, self.base_url = _config()
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT) if self.base_url.startswith("https:") else None
            with urlopen(request, timeout=90, context=context) as response:
                data = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"LLM returned HTTP {error.code}") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach LLM: {error.reason}") from error
        choices = data.get("choices") or []
        if len(choices) != 1:
            raise ValueError("Expected exactly one LLM completion")
        message = choices[0].get("message") or {}
        if message.get("tool_calls") or message.get("function_call"):
            raise ValueError("Standard RAG received an unexpected tool call")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"LLM response has no text content (finish_reason={choices[0].get('finish_reason')}, usage={data.get('usage')})")
        return LLMResponse(
            content=content,
            model=data.get("model") or self.model,
            usage=data.get("usage") or {},
            finish_reason=choices[0].get("finish_reason"),
        )
