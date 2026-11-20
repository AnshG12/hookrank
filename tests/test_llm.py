import io
import urllib.error

import pytest

from hookrank.cache import CallCache, call_key
from hookrank.llm import ChatClient, ModelSpec, RequestTooLarge, TruncatedResponse, _parse_duration, _retry_delay


def _spec(**kw):
    return ModelSpec(model="m", base_url="http://x", api_key="k", **kw)


def test_parse_groq_durations():
    assert _parse_duration("577ms") == pytest.approx(0.577)
    assert _parse_duration("1m26.4s") == pytest.approx(86.4)


def test_otpm_rejection_waits_a_full_window():
    exc = urllib.error.HTTPError("u", 429, "x", {}, io.BytesIO(b""))
    assert _retry_delay(exc, "output tokens per minute (OTPM): Limit 1000", 1) == 65.0
    assert _retry_delay(exc, "Please try again in 2.5s", 1) == pytest.approx(2.5)


def test_oversized_prompt_fails_before_any_request():
    client = ChatClient(_spec(request_token_limit=1000))
    with pytest.raises(RequestTooLarge):
        client.complete("word " * 2000)


def test_truncated_answer_is_rejected(monkeypatch):
    client = ChatClient(_spec())
    monkeypatch.setattr(client, "_post", lambda payload: (
        {"choices": [{"message": {"content": '{"events": ['}, "finish_reason": "length"}]}, {}))
    with pytest.raises(TruncatedResponse):
        client.complete("short prompt")


def test_explicit_params_are_sent(monkeypatch):
    sent = {}
    client = ChatClient(_spec(params=(("reasoning_effort", "low"),)))

    def fake_post(payload):
        sent.update(payload)
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}, {}
    monkeypatch.setattr(client, "_post", fake_post)
    client.complete("hello")
    assert sent["reasoning_effort"] == "low" and sent["max_completion_tokens"] > 0
    assert sent["response_format"] == {"type": "json_object"}


def test_cache_key_tracks_every_input(tmp_path):
    base = call_key("annotation", "m", "u", {"a": 1}, "prompt")
    assert base != call_key("annotation", "m", "u", {"a": 1}, "prompt changed")
    assert base != call_key("annotation", "m", "u", {"a": 2}, "prompt")
    assert base != call_key("story", "m", "u", {"a": 1}, "prompt")
    cache = CallCache(tmp_path / "c.sqlite3")
    cache.put(base, stage="annotation", series_id="s", episode_id="e", model="m", params={}, prompt="prompt",
              raw="{}", usage={"prompt_tokens": 3}, finish_reason="stop", latency_s=0.1)
    assert cache.get(base)["usage"] == {"prompt_tokens": 3}
    cache.close()


def test_output_token_limit_is_learned_and_paced(monkeypatch):
    from hookrank.llm import RateLimiter
    slept = []
    monkeypatch.setattr("hookrank.llm.time.sleep", lambda s: slept.append(s))
    limiter = RateLimiter()
    limiter.note_output(500)
    assert limiter.wait_for(100) == 0.0                       # limit unknown: no pacing
    limiter.learn_output_limit('... on output tokens per minute (OTPM): Limit 1000, Requested 1324 ...')
    limiter.note_output(500)
    assert 29.0 < limiter.wait_for(100) <= 30.0              # 500 of 1000 tokens/min -> ~30s gap


def test_model_references_route_to_their_endpoints():
    from hookrank.config import Settings, split_model
    from hookrank.llm import spec_for
    assert split_model("gemini:gemini-3.5-flash-lite") == ("gemini", "gemini-3.5-flash-lite")
    assert split_model("openai/gpt-oss-120b") == ("openai", "openai/gpt-oss-120b")
    assert split_model("qwen2.5:3b") == ("openai", "qwen2.5:3b")            # colon inside a model ID
    assert split_model("nebius:openai/gpt-oss-120b") == ("nebius", "openai/gpt-oss-120b")
    assert split_model("anthropic:claude-opus-5-5") == ("anthropic", "claude-opus-5-5")
    s = Settings.from_mapping({"GROQ_API_KEY": "g", "GEMINI_API_KEY": "k"})
    gem, groq = spec_for("gemini:gemini-3.5-flash-lite", s), spec_for("groq:openai/gpt-oss-120b", s)
    assert gem.model == "gemini-3.5-flash-lite" and "generativelanguage" in gem.base_url and gem.api_key == "k"
    assert gem.request_token_limit == 32000 and groq.request_token_limit == 8000 and groq.api_key == "g"
    assert s.missing_keys() == []                                            # CLI defaults: Gemini + Groq
    assert Settings.from_mapping({"GROQ_API_KEY": "g"}).missing_keys() == ["GEMINI_API_KEY"]
    # older .env files: Groq key under OPENAI_* with OPENAI_BASE_URL pointing at Groq still works, and stays free
    legacy = Settings.from_mapping({"OPENAI_BASE_URL": "https://api.groq.com/openai/v1", "OPENAI_API_KEY": "old"})
    assert legacy.endpoint("groq").api_key == "old" and not legacy.endpoint("openai").paid


def test_budget_guard_refuses_calls_that_could_break_the_cap(tmp_path, monkeypatch):
    from hookrank.budget import BudgetExceeded, BudgetGuard, Price
    paid = "https://paid.example/v1"
    guard = BudgetGuard(cap_usd=0.01, prices={"m": Price(1e-6, 2e-6)}, ledger_path=tmp_path / "ledger.sqlite3",
                        paid_base_urls=(paid,))
    client = ChatClient(ModelSpec(model="m", base_url=paid, api_key="k", max_output_tokens=1000), budget=guard)
    monkeypatch.setattr(client, "_post", lambda payload: (
        {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}, {}))
    client.complete("x " * 400)                                   # costs 1000*1e-6 + 500*2e-6 = $0.002
    assert guard.spent() == pytest.approx(0.002)
    for _ in range(3):
        client.complete("x " * 400)
    assert guard.spent() == pytest.approx(0.008)
    with pytest.raises(BudgetExceeded):                           # worst case would pass $0.01: not sent
        client.complete("x " * 400)
    assert guard.spent() == pytest.approx(0.008) and guard._reserved == 0
    with pytest.raises(BudgetExceeded):                           # unknown price on a paid endpoint: refused
        ChatClient(ModelSpec(model="unpriced", base_url=paid, api_key="k"), budget=guard).complete("hi")
    free = ChatClient(ModelSpec(model="free", base_url="https://free.example/v1", api_key="k"), budget=guard)
    monkeypatch.setattr(free, "_post", lambda payload: (
        {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 9}}, {}))
    free.complete("hi")                                           # unpaid endpoint: never blocked, costs 0
    assert guard.spent() == pytest.approx(0.008)
    guard.close()
    reopened = BudgetGuard(cap_usd=0.01, prices={}, ledger_path=tmp_path / "ledger.sqlite3", paid_base_urls=(paid,))
    assert reopened.spent() == pytest.approx(0.008)               # cumulative across processes/runs
    reopened.close()


def test_per_run_spend_ignores_other_runs_sharing_the_ledger(tmp_path):
    from hookrank.budget import BudgetGuard, Price
    paid, ledger = "https://paid.example/v1", tmp_path / "ledger.sqlite3"
    prices = {"m": Price(1e-6, 1e-6)}
    run_a = BudgetGuard(10, prices, ledger, (paid,), run_label="run-a")
    run_b = BudgetGuard(10, prices, ledger, (paid,), run_label="run-b")   # a concurrent process
    run_a.settle(run_a.reserve(paid, "m", 10, 10), paid, "m", 1000, 0)
    run_b.settle(run_b.reserve(paid, "m", 10, 10), paid, "m", 3000, 0)
    assert run_a.spent_by_label("run-a") == pytest.approx(0.001)          # not 0.004
    assert run_a.spent() == pytest.approx(0.004)                          # the cap still sees everything
    run_a.close(); run_b.close()
