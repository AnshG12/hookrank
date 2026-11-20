"""Cached, budget-guarded model calls shared by every pipeline stage (narrative engine, clip check).

One implementation of the cache-key scheme, the truncation memo, cache-only mode, per-model rate
limiters and the spending guard, so stages cannot drift apart.
"""
from __future__ import annotations

import logging
import threading

from pydantic import BaseModel

from .budget import BudgetGuard
from .cache import CallCache, call_key
from .llm import ChatClient, ModelSpec, RateLimiter, TruncatedResponse
from .providers import ProviderError

log = logging.getLogger("hookrank.calls")


class CallRecord(BaseModel):
    stage: str
    model: str
    cached: bool
    ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = ""
    latency_s: float | None = None
    waited_s: float | None = None
    error: str = ""
    superseded: bool = False    # truncated answer recovered by re-annotating in smaller windows

    @property
    def failed(self) -> bool:
        return not self.ok and not self.superseded


class CachedCaller:
    """Call models through the response cache, one client and rate limiter per model (thread-safe)."""

    def __init__(self, cache: CallCache, series_id: str, cache_only: bool = False,
                 budget: BudgetGuard | None = None) -> None:
        self.cache = cache
        self.series_id = series_id
        self.cache_only = cache_only
        self.budget = budget
        self._clients: dict[ModelSpec, ChatClient] = {}
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def client(self, spec: ModelSpec) -> ChatClient:
        with self._lock:
            if spec not in self._clients:
                limiter = self._limiters.setdefault(f"{spec.base_url}|{spec.model}", RateLimiter())
                self._clients[spec] = ChatClient(spec, limiter, budget=self.budget)
            return self._clients[spec]

    def call(self, stage: str, spec: ModelSpec, prompt: str, episode_id: str, label: str,
             min_output: int) -> tuple[str | None, CallRecord, bool]:
        """Return (raw text or None, call record, truncated?). ``label`` is used in log lines only."""
        params = {**spec.params_dict(), "max_output_tokens": spec.max_output_tokens,
                  "request_token_limit": spec.request_token_limit, "json_mode": spec.json_mode}
        key = call_key(stage, spec.model, spec.base_url, params, prompt)
        hit = self.cache.get(key)
        if hit and hit["finish_reason"] == "length":   # known to truncate: go straight to smaller windows
            return None, CallRecord(stage=stage, model=spec.model, cached=True, ok=False, finish_reason="length",
                                    error="truncated (cached)"), True
        if hit:
            usage = hit["usage"]
            return hit["raw"], CallRecord(stage=stage, model=spec.model, cached=True, ok=True,
                                          prompt_tokens=usage.get("prompt_tokens"),
                                          completion_tokens=usage.get("completion_tokens"),
                                          finish_reason=hit["finish_reason"], latency_s=hit["latency_s"]), False
        if self.cache_only:
            return None, CallRecord(stage=stage, model=spec.model, cached=False, ok=False,
                                    error="not in cache (cache-only run)"), False
        try:
            result = self.client(spec).complete(prompt, min_output_tokens=min_output)
        except ProviderError as exc:
            truncated = isinstance(exc, TruncatedResponse)
            if truncated:  # remember the outcome (not the incomplete text) so re-runs skip this call
                self.cache.put(key, stage=stage, series_id=self.series_id, episode_id=episode_id,
                               model=spec.model, params=params, prompt=prompt, raw="", usage={},
                               finish_reason="length", latency_s=0.0)
            log.warning("%s %s %s failed: %s", label, stage, spec.short_name, exc)
            return None, CallRecord(stage=stage, model=spec.model, cached=False, ok=False,
                                    finish_reason="length" if truncated else "", error=str(exc)[:500]), truncated
        self.cache.put(key, stage=stage, series_id=self.series_id, episode_id=episode_id, model=spec.model,
                       params=params, prompt=prompt, raw=result.text, usage=result.usage,
                       finish_reason=result.finish_reason, latency_s=result.latency_s)
        log.info("%s %s %s ok (%s in / %s out tokens, waited %.0fs)", label, stage, spec.short_name,
                 result.usage.get("prompt_tokens"), result.usage.get("completion_tokens"), result.waited_s)
        return result.text, CallRecord(stage=stage, model=spec.model, cached=False, ok=True,
                                       prompt_tokens=result.usage.get("prompt_tokens"),
                                       completion_tokens=result.usage.get("completion_tokens"),
                                       finish_reason=result.finish_reason, latency_s=result.latency_s,
                                       waited_s=result.waited_s), False
