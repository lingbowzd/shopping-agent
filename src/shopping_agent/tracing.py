"""Measured request usage; cost appears only when rates are configured."""

from __future__ import annotations

import os
from typing import Any


def summarize_usage(trace: dict[str, Any], total_latency_ms: float) -> dict[str, Any]:
    input_tokens = output_tokens = 0
    token_usage_complete = True
    for call in trace.get("llm_calls", []):
        usage = call.get("usage") or {}
        input_value = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_value = usage.get("output_tokens", usage.get("completion_tokens"))
        if input_value is None or output_value is None:
            token_usage_complete = False
        input_tokens += int(input_value or 0)
        output_tokens += int(output_value or 0)
    input_rate = os.getenv("LLM_INPUT_COST_PER_1M")
    output_rate = os.getenv("LLM_OUTPUT_COST_PER_1M")
    estimated_cost = None
    if input_rate and output_rate and token_usage_complete:
        estimated_cost = round(
            input_tokens * float(input_rate) / 1_000_000
            + output_tokens * float(output_rate) / 1_000_000, 8
        )
    return {
        "llm_call_count": len(trace.get("llm_calls", [])),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_usage_complete": token_usage_complete,
        "estimated_llm_api_cost_usd": estimated_cost,
        "cost_note": (
            "Token totals are partial because an LLM call omitted usage; cost unavailable"
            if not token_usage_complete else
            "LLM rates not configured; reranker API cost not included"
            if estimated_cost is None else
            "LLM estimate from configured rates; reranker API cost not included"
        ),
        "total_latency_ms": round(total_latency_ms, 2),
    }
