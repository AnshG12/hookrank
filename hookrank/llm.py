"""Chat client for every configured provider: OpenAI-compatible endpoints and Claude.

Why this exists alongside ``providers.py``: the narrative engine needs things the simple
``complete(prompt) -> str`` interface cannot give it (see docs/decisions.md S1/D4):

* explicit per-model request parameters (gpt-oss fails with Groq defaults because hidden
  reasoning exhausts the output budget),
* usage and finish-reason reporting, so truncated answers are detected, not trusted,
* rate-limit pacing from response headers (Groq free tier: 8k tokens/min per model),
* a request budget that keeps prompt + output under the per-request token ceiling
  (requests above it are rejected outright with HTTP 413).

OpenAI-compatible providers (OpenAI, Gemini, Nebius, Groq, gateways) are called over plain HTTP;
Claude is called through the official Anthropic SDK (Messages API). Both paths share the same retry
rules, pacing, truncation detection and spending guard, so a model swap changes nothing downstream.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .budget import BudgetGuard
from .config import ANTHROPIC_MESSAGES, CHAT_COMPLETIONS, Settings
from .providers import USER_AGENT, ProviderError

log = logging.getLogger("hookrank.llm")

# Conservative characters-per-token for budget estimates. Measured on this pack:
# gpt-oss ~3.1, qwen ~2.6 chars/token (S1). Underestimating tokens causes HTTP 413.
CHARS_PER_TOKEN = 2.4


class RequestTooLarge(ProviderError):
    """The request cannot fit the provider's per-request token ceiling; retrying will not help."""


class TruncatedResponse(ProviderError):
    """The model stopped because it ran out of output budget; its answer is incomplete."""


class HTTPFailure(Exception):
    """A provider answered with an HTTP error; the same retry rules apply whichever provider it was."""

    def __init__(self, code: int, body: str, headers) -> None:
        super().__init__(f"HTTP {code}")
        self.code, self.body, self.headers = code, body, headers


# Request parameters a provider may reject for some models (e.g. reasoning models refuse a custom
# temperature). They are dropped and the request retried once, instead of hard-coding model lists.
OPTIONAL_PARAMS = ("temperature", "reasoning_effort")
# Claude models that accept server-side refusal fallbacks (beta); see ChatClient._post_anthropic.
FALLBACK_MODEL_PREFIXES = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5-5")


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass(frozen=True)
class ModelSpec:
    """One model endpoint plus the exact request parameters used with it."""
    model: str
    base_url: str
    api_key: str = field(repr=False)
    params: tuple[tuple[str, object], ...] = ()
    json_mode: bool = True
    request_token_limit: int = 8000   # per-request ceiling (prompt + max output)
    max_output_tokens: int = 3000
    concurrency: int = 1              # parallel requests allowed for this model
    api: str = CHAT_COMPLETIONS       # wire protocol (config.CHAT_COMPLETIONS or config.ANTHROPIC_MESSAGES)
    auth_header: str = "bearer"       # "bearer" -> Authorization: Bearer <key>; otherwise that header carries the key
    key_required: bool = True         # False for models served locally (e.g. Ollama)

    @property
    def short_name(self) -> str:
        return self.model.split("/")[-1]

    def params_dict(self) -> dict:
        return dict(self.params)


def default_params(model: str, settings: Settings | None = None, api: str = CHAT_COMPLETIONS) -> tuple[tuple[str, object], ...]:
    """Model-family request parameters, made explicit (never rely on provider defaults)."""
    if api == ANTHROPIC_MESSAGES:
        effort = settings.anthropic_effort if settings else ""
        return (("effort", effort),) if effort else ()
    if "gpt-oss" in model:
        # Low effort: medium needs ~4-6k reasoning tokens, which cannot fit an 8k
        # free-tier request together with an episode prompt (S1).
        return (("reasoning_effort", "low"),)
    return ()


def spec_for(ref: str, settings: Settings) -> ModelSpec:
    """ModelSpec for a ``provider:model`` reference (see config.split_model)."""
    provider, model = settings.split_model(ref)
    endpoint = settings.endpoint(provider)
    return ModelSpec(model=model, base_url=endpoint.base_url, api_key=endpoint.api_key,
                     params=default_params(model, settings, endpoint.api), request_token_limit=endpoint.request_token_limit,
                     max_output_tokens=endpoint.max_output_tokens, concurrency=endpoint.concurrency, api=endpoint.api,
                     auth_header=endpoint.auth_header, key_required=endpoint.key_required)


@dataclass
class ChatResult:
    text: str
    model: str
    usage: dict
    finish_reason: str
    latency_s: float
    attempts: int
    waited_s: float
    max_output_tokens: int


class RateLimiter:
    """Pacing for one model's quotas (thread-safe).

    * tokens/minute: from ``x-ratelimit-*`` response headers;
    * output tokens/minute (OTPM): Groq enforces this on some models (qwen: 1,000/min) without
      exposing it in headers. The limit is learned from the first OTPM rejection; afterwards
      each answer of N output tokens is followed by a gap of 60*N/limit seconds, so the next
      request starts in a refilled window instead of being rejected and retried.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._remaining: int | None = None
        self._reset_at: float = 0.0
        self._otpm: int | None = None
        self._last_done: float = 0.0
        self._last_output: int = 0

    def wait_for(self, tokens_needed: int) -> float:
        now = time.monotonic()
        with self._lock:
            remaining, reset_at = self._remaining, self._reset_at
            otpm, last_done, last_output = self._otpm, self._last_done, self._last_output
        token_delay = max(0.0, reset_at - now) if remaining is not None and remaining < tokens_needed else 0.0
        output_delay = max(0.0, last_done + 60.0 * last_output / otpm - now) if otpm and last_output else 0.0
        delay = max(token_delay, output_delay)
        if delay:
            reason = "output-token quota" if output_delay >= token_delay else f"token quota (need ~{tokens_needed}, {remaining} left)"
            log.info("pacing %.1fs for %s", delay, reason)
            time.sleep(delay)
        return delay

    def learn_output_limit(self, body: str) -> None:
        match = re.search(r"\(OTPM\): Limit (\d+)", body)
        if match:
            with self._lock:
                self._otpm = int(match.group(1))

    def note_output(self, completion_tokens: int) -> None:
        with self._lock:
            self._last_done = time.monotonic()
            self._last_output = completion_tokens

    def update(self, headers) -> None:
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = headers.get("x-ratelimit-reset-tokens")
        if remaining is None:
            return
        with self._lock:
            try:
                self._remaining = int(remaining)
            except ValueError:
                return
            self._reset_at = time.monotonic() + (_parse_duration(reset) if reset else 60.0)


def _parse_duration(value: str) -> float:
    """Parse Groq-style durations such as '577ms', '32.5s', '1m26.4s'."""
    total = 0.0
    for amount, unit in re.findall(r"([\d.]+)(ms|s|m|h)", value or ""):
        total += float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


def _retry_delay(exc: urllib.error.HTTPError, body: str, attempt: int) -> float:
    if "(OTPM)" in body:
        # Groq output-tokens-per-minute limit (qwen: 1,000/min, not exposed in headers and with
        # no retry hint). Only a fresh one-minute window helps.
        return 65.0
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    match = re.search(r"try again in ((?:[\d.]+(?:ms|s|m|h))+)", body)
    if match:
        return _parse_duration(match.group(1))
    return min(60.0, 5.0 * (attempt + 1))


class ChatClient:
    """Single-turn JSON chat against one OpenAI-compatible model."""

    def __init__(self, spec: ModelSpec, limiter: RateLimiter | None = None, max_attempts: int = 8,
                 timeout_s: int = 180, budget: BudgetGuard | None = None) -> None:
        if spec.key_required and not spec.api_key:
            raise ProviderError(f"No API key configured for {spec.model}.")
        self.spec = spec
        self.limiter = limiter or RateLimiter()
        self.budget = budget
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self._anthropic = None
        self._fallbacks = spec.api == ANTHROPIC_MESSAGES and spec.model.startswith(FALLBACK_MODEL_PREFIXES)

    def output_budget(self, prompt: str) -> int:
        """Largest output budget that keeps the request under the per-request ceiling."""
        room = self.spec.request_token_limit - estimate_tokens(prompt) - 150
        return min(self.spec.max_output_tokens, room)

    def complete(self, prompt: str, min_output_tokens: int = 800) -> ChatResult:
        budget = self.output_budget(prompt)
        if budget < min_output_tokens:
            raise RequestTooLarge(
                f"Prompt (~{estimate_tokens(prompt)} tokens) leaves only {budget} output tokens under the "
                f"{self.spec.request_token_limit}-token request ceiling for {self.spec.model}; split the input.")
        payload = {"model": self.spec.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.1, "max_completion_tokens": budget, **self.spec.params_dict()}
        if self.spec.json_mode:
            payload["response_format"] = {"type": "json_object"}
        reservation = 0.0
        if self.budget is not None:   # refuses (BudgetExceeded) before anything is sent
            reservation = self.budget.reserve(self.spec.base_url, self.spec.model, estimate_tokens(prompt), budget)
        call = {"settled": False}   # per-call state, so a client is safe to share between threads
        try:
            return self._complete(payload, prompt, budget, reservation, call)
        except BaseException:
            if self.budget is not None and reservation and not call["settled"]:
                self.budget.release(reservation)
            raise

    def _settle(self, reservation: float, usage: dict, call: dict) -> None:
        if self.budget is not None:
            self.budget.settle(reservation, self.spec.base_url, self.spec.model,
                               int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))
            call["settled"] = True

    def _complete(self, payload: dict, prompt: str, budget: int, reservation: float, call: dict) -> ChatResult:
        started, waited = time.monotonic(), 0.0
        for attempt in range(1, self.max_attempts + 1):
            waited += self.limiter.wait_for(estimate_tokens(prompt) + budget)
            try:
                data, headers = self._post(payload)
            except HTTPFailure as exc:
                body = exc.body
                if exc.headers:
                    self.limiter.update(exc.headers)
                self.limiter.learn_output_limit(body)
                if exc.code == 413:
                    raise RequestTooLarge(f"{self.spec.model} rejected the request as too large: {body}") from exc
                # Some models reject JSON mode; retry once without it (Python still validates the JSON).
                if exc.code == 400 and "response_format" in payload and ("response_format" in body or "json" in body.lower()) \
                        and "json_validate_failed" not in body:
                    log.info("%s rejected JSON mode; retrying without response_format", self.spec.short_name)
                    payload.pop("response_format")
                    continue
                rejected = [p for p in OPTIONAL_PARAMS if p in payload and p in body] if exc.code == 400 else []
                if rejected:
                    log.info("%s rejected %s; retrying without it", self.spec.short_name, ", ".join(rejected))
                    for name in rejected:
                        payload.pop(name)
                    continue
                if exc.code == 400 and "max_completion_tokens" in payload and "max_completion_tokens" in body:
                    payload["max_tokens"] = payload.pop("max_completion_tokens")   # older OpenAI-compatible gateways
                    continue
                if exc.code == 400 and self._fallbacks and "fallback" in body.lower():
                    log.info("%s: refusal fallbacks unavailable; retrying without them", self.spec.short_name)
                    self._fallbacks = False
                    continue
                # Server-side JSON validation failures are non-deterministic; one retry is cheap.
                json_retry = exc.code == 400 and "json_validate_failed" in body and attempt == 1
                if (exc.code in (429, 500, 502, 503, 504, 529) or json_retry) and attempt < self.max_attempts:
                    delay = _retry_delay(exc, body, attempt) + 0.5
                    log.info("%s HTTP %s; retrying in %.1fs (attempt %d)", self.spec.short_name, exc.code, delay, attempt)
                    if delay > 120:  # far beyond a per-minute window: record why (e.g. a daily quota, R4)
                        log.warning("%s long wait; provider said: %s", self.spec.short_name,
                                    re.sub(r"org_\w+", "org_<redacted>", body)[:400])
                    time.sleep(delay)
                    waited += delay
                    continue
                raise ProviderError(f"{self.spec.model} returned HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_attempts:
                    time.sleep(2.0 * attempt)
                    waited += 2.0 * attempt
                    continue
                raise ProviderError(f"Could not reach {self.spec.base_url}: {exc}") from exc
            self.limiter.update(headers)
            self.limiter.note_output(int((data.get("usage") or {}).get("completion_tokens") or 0))
            try:
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(f"Unexpected response shape from {self.spec.model}.") from exc
            finish = choice.get("finish_reason") or ""
            self._settle(reservation, data.get("usage") or {}, call)   # billed even if truncated
            if finish == "refusal":
                raise ProviderError(f"{self.spec.model} declined this request (refusal stop reason).")
            if finish == "length":
                raise TruncatedResponse(f"{self.spec.model} hit its {budget}-token output budget; answer is incomplete.")
            return ChatResult(text=text, model=self.spec.model, usage=data.get("usage") or {}, finish_reason=finish,
                              latency_s=round(time.monotonic() - started - waited, 2), attempts=attempt,
                              waited_s=round(waited, 1), max_output_tokens=budget)
        raise ProviderError(f"{self.spec.model}: request failed after {self.max_attempts} attempts.")

    def _post(self, payload: dict) -> tuple[dict, dict]:
        """Send one request; return an OpenAI-shaped response body (choices, usage) and the headers."""
        if self.spec.api == ANTHROPIC_MESSAGES:
            return self._post_anthropic(payload)
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.spec.api_key:
            if self.spec.auth_header.lower() in ("bearer", "authorization"):
                headers["Authorization"] = f"Bearer {self.spec.api_key}"
            else:                                     # e.g. Azure OpenAI's "api-key" header
                headers[self.spec.auth_header] = self.spec.api_key
        request = urllib.request.Request(f"{self.spec.base_url}/chat/completions",
                                         data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                headers = {k.lower(): v for k, v in response.headers.items()}
                try:
                    return json.loads(response.read().decode("utf-8")), headers
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"{self.spec.model} returned a non-JSON HTTP body.") from exc
        except urllib.error.HTTPError as exc:
            raise HTTPFailure(exc.code, exc.read().decode("utf-8", errors="replace")[:1000], exc.headers) from exc

    def _post_anthropic(self, payload: dict) -> tuple[dict, dict]:
        """One Claude Messages API call through the official SDK, normalised to the OpenAI shape.

        Temperature and JSON mode are not sent (current Claude models reject sampling parameters;
        the prompts ask for JSON and Python validates it). The SDK's own retries are off so the
        shared retry, pacing and budget logic above applies unchanged.
        """
        import anthropic   # imported here so OpenAI-compatible setups do not need the package

        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic(api_key=self.spec.api_key, base_url=self.spec.base_url,
                                                  max_retries=0, timeout=float(self.timeout_s))
        kwargs: dict = {"model": self.spec.model, "messages": payload["messages"],
                        "max_tokens": payload.get("max_completion_tokens") or payload.get("max_tokens")}
        if payload.get("effort"):
            kwargs["output_config"] = {"effort": payload["effort"]}
        try:
            if self._fallbacks:   # on a policy refusal the API re-runs the request on a suitable model
                message = self._anthropic.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                               fallbacks="default", **kwargs)
            else:
                message = self._anthropic.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise HTTPFailure(exc.status_code, (exc.response.text or str(exc.message))[:1000], exc.response.headers) from exc
        except anthropic.APIConnectionError as exc:   # includes timeouts
            raise urllib.error.URLError(str(exc)) from exc
        text = "".join(block.text for block in message.content if block.type == "text")
        finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}.get(
            message.stop_reason or "", message.stop_reason or "")
        usage = message.usage
        prompt_tokens = (usage.input_tokens + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
                         + (getattr(usage, "cache_read_input_tokens", 0) or 0))
        return ({"choices": [{"message": {"content": text}, "finish_reason": finish}], "model": message.model,
                 "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": usage.output_tokens}}, {})
