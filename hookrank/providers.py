"""Shared provider-facing helpers: the error type, client identification and JSON extraction.

The HTTP client itself lives in ``llm.py``.
"""
from __future__ import annotations

import json
import re

# urllib's default "Python-urllib/x.y" agent is blocked by Cloudflare-fronted APIs
# (Groq returns HTTP 403, code 1010), so every request identifies the client (D4).
USER_AGENT = "hookrank/0.1"


class ProviderError(RuntimeError):
    """Actionable provider configuration or request error."""


def extract_json(text: str) -> dict:
    """Accept a raw JSON object or one wrapped in a markdown code fence."""
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.S)
        if not match:
            raise ProviderError("Model response did not contain a JSON object.")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ProviderError("Model response contained malformed JSON.") from exc
    if not isinstance(value, dict):
        raise ProviderError("Model response must be a JSON object.")
    return value
