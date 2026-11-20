"""Model swapping through configuration only: OpenAI (GPT), Google (Gemini), Anthropic (Claude), Nebius and
OpenAI-compatible gateways, with Nebius as the default and the spending guard covering every provider."""
from types import SimpleNamespace

import pytest

from hookrank import studio
from hookrank.budget import BudgetExceeded, BudgetGuard, Price, parse_prices, resolve_prices, unpriced_models
from hookrank.config import ANTHROPIC_MESSAGES, CHAT_COMPLETIONS, Settings
from hookrank.llm import ChatClient, HTTPFailure, TruncatedResponse, spec_for
from hookrank.providers import ProviderError, extract_json

KEYS = {"NEBIUS_API_KEY": "nb-key", "OPENAI_API_KEY": "sk-key", "GEMINI_API_KEY": "gm-key",
        "ANTHROPIC_API_KEY": "ant-key"}


def _ok(content="{}", usage=None):
    return ({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
             "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5}}, {})


# ---- routing -----------------------------------------------------------------------------------

@pytest.mark.parametrize("ref, url_part, key, api", [
    ("openai:gpt-5.5", "api.openai.com/v1", "sk-key", CHAT_COMPLETIONS),
    ("gemini:gemini-3.5-pro", "generativelanguage.googleapis.com", "gm-key", CHAT_COMPLETIONS),
    ("anthropic:claude-opus-5-5", "api.anthropic.com", "ant-key", ANTHROPIC_MESSAGES),
    ("nebius:openai/gpt-oss-120b", "tokenfactory.nebius.com", "nb-key", CHAT_COMPLETIONS),
])
def test_each_provider_is_one_prefix_and_one_key(ref, url_part, key, api):
    spec = spec_for(ref, Settings.from_mapping(KEYS))
    assert url_part in spec.base_url and spec.api_key == key and spec.api == api


def test_a_custom_gateway_needs_only_its_url_and_key():
    s = Settings.from_mapping({"CUSTOM_BASE_URL": "https://llm.example.com/v1/", "CUSTOM_API_KEY": "c",
                               "HOOKRANK_STORY_MODEL": "custom:big-model", "HOOKRANK_ANNOTATOR_MODELS": "custom:big-model",
                               "HOOKRANK_VERIFIER_MODEL": "custom:big-model"})
    assert spec_for("custom:big-model", s).base_url == "https://llm.example.com/v1" and s.missing_keys() == []
    assert Settings.from_mapping({"HOOKRANK_STORY_MODEL": "custom:m", "HOOKRANK_ANNOTATOR_MODELS": "custom:m",
                                  "HOOKRANK_VERIFIER_MODEL": "custom:m"}).missing_keys() == ["CUSTOM_API_KEY", "CUSTOM_BASE_URL"]


# ---- the web app: Nebius by default, other providers only when their key is in secrets ------------

def test_app_defaults_to_nebius_and_swaps_roles_from_secrets_only(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    default = studio.studio_settings({"NEBIUS_API_KEY": "nb"}, studio.Path("/dev/null"), 0.5)
    assert all(ref.startswith("nebius:") for ref in default.model_refs)
    swapped = studio.studio_settings({**KEYS, "HOOKRANK_STORY_MODEL": "anthropic:claude-opus-5-5",
                                      "HOOKRANK_ANNOTATOR_MODELS": "openai:gpt-5.5,gemini:gemini-3.5-pro",
                                      "HOOKRANK_VERIFIER_MODEL": "anthropic:claude-sonnet-5-5"}, studio.Path("/dev/null"), 0.5)
    assert swapped.story_model == "anthropic:claude-opus-5-5"
    assert swapped.annotator_models == ("openai:gpt-5.5", "gemini:gemini-3.5-pro")
    assert "claude-opus-5-5 (Anthropic (Claude))" in studio.describe_models(swapped)
    assert all(e.paid for e in swapped.endpoints_in_use)            # every provider is capped in the app


def test_a_role_without_its_provider_key_falls_back_to_nebius(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    s = studio.studio_settings({"NEBIUS_API_KEY": "nb", "OPENAI_API_KEY": "sk",
                                "HOOKRANK_STORY_MODEL": "anthropic:claude-opus-5-5",          # no Anthropic key
                                "HOOKRANK_ANNOTATOR_MODELS": "openai:gpt-5.5,gemini:gemini-3.5-pro"},  # no Gemini key
                               studio.Path("/dev/null"), 0.5)
    assert s.story_model == studio.STUDIO_DEFAULTS["HOOKRANK_STORY_MODEL"]
    assert s.annotator_models == ("openai:gpt-5.5", "nebius:openai/gpt-oss-120b")             # two annotators kept
    assert s.missing_keys() == []
    single = studio.studio_settings({"NEBIUS_API_KEY": "nb", "OPENAI_API_KEY": "sk",
                                     "HOOKRANK_ANNOTATOR_MODELS": "openai:gpt-5.5"}, studio.Path("/dev/null"), 0.5)
    assert single.annotator_models == ("openai:gpt-5.5",)                       # a deliberate choice is kept


# ---- OpenAI and Gemini (chat completions) ------------------------------------------------------

def test_gpt_models_that_reject_temperature_are_retried_without_it(monkeypatch):
    client = ChatClient(spec_for("openai:gpt-5.5", Settings.from_mapping(KEYS)))
    sent = []

    def fake_post(payload):
        sent.append(dict(payload))
        if "temperature" in payload:
            raise HTTPFailure(400, "Unsupported value: 'temperature' does not support 0.1 with this model.", {})
        return _ok('{"status": "ok"}')
    monkeypatch.setattr(client, "_post", fake_post)
    assert client.complete("hi", min_output_tokens=16).text == '{"status": "ok"}'
    assert "temperature" in sent[0] and "temperature" not in sent[1] and sent[1]["response_format"]


def test_gemini_requests_use_the_openai_compatible_shape(monkeypatch):
    client = ChatClient(spec_for("gemini:gemini-3.5-pro", Settings.from_mapping(KEYS)))
    sent = {}
    monkeypatch.setattr(client, "_post", lambda payload: sent.update(payload) or _ok())
    client.complete("hi", min_output_tokens=16)
    assert sent["model"] == "gemini-3.5-pro" and sent["max_completion_tokens"] > 0
    assert sent["response_format"] == {"type": "json_object"}


# ---- Claude (Messages API through the Anthropic SDK) -------------------------------------------

class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _claude(text="{}", stop="end_turn", input_tokens=1000, output_tokens=500):
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           stop_reason=stop, model="claude-opus-5-5",
                           usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                                                 cache_creation_input_tokens=0, cache_read_input_tokens=0))


def _fake_sdk(monkeypatch, replies):
    import anthropic
    plain, beta = FakeMessages(replies), FakeMessages(replies)
    beta.replies = plain.replies                               # one shared reply queue
    made = []

    def factory(**kwargs):
        made.append(kwargs)
        return SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    monkeypatch.setattr(anthropic, "Anthropic", factory)
    return made, plain, beta


def _status_error(code, text):
    import anthropic
    exc = anthropic.APIStatusError.__new__(anthropic.APIStatusError)
    exc.status_code, exc.message, exc.response = code, text, SimpleNamespace(text=text, headers={})
    return exc


def test_claude_is_called_through_the_sdk_and_billed_like_any_model(monkeypatch, tmp_path):
    made, plain, beta = _fake_sdk(monkeypatch, [_claude('{"status": "ok"}')])
    settings = Settings.from_mapping({**KEYS, "ANTHROPIC_EFFORT": "low"})
    spec = spec_for("anthropic:claude-opus-5-5", settings)
    paid = spec.base_url
    guard = BudgetGuard(1.0, resolve_prices(Settings.from_mapping({})), tmp_path / "l.sqlite3", (paid,))
    result = ChatClient(spec, budget=guard).complete("Return JSON", min_output_tokens=16)
    assert result.text == '{"status": "ok"}' and result.finish_reason == "stop"
    assert made[0]["api_key"] == "ant-key" and made[0]["max_retries"] == 0      # shared retry logic, not the SDK's
    call = beta.calls[0]                                                         # refusal fallbacks on Opus 5.5
    assert call["fallbacks"] == "default" and call["output_config"] == {"effort": "low"}
    assert "temperature" not in call and "response_format" not in call and call["max_tokens"] > 0
    assert guard.spent() == pytest.approx(1000 * 4e-6 + 500 * 20e-6)          # built-in Claude price
    guard.close()


def test_claude_truncation_refusal_and_overload_are_handled(monkeypatch):
    spec = spec_for("anthropic:claude-haiku-5-5", Settings.from_mapping(KEYS))   # no fallbacks on Haiku
    monkeypatch.setattr("hookrank.llm.time.sleep", lambda s: None)
    _, plain, beta = _fake_sdk(monkeypatch, [_status_error(529, "overloaded"), _claude('{"a": 1}')])
    assert ChatClient(spec).complete("x", min_output_tokens=16).text == '{"a": 1}' and len(plain.calls) == 2
    assert not beta.calls
    _fake_sdk(monkeypatch, [_claude('{"events": [', stop="max_tokens")])
    with pytest.raises(TruncatedResponse):
        ChatClient(spec).complete("x", min_output_tokens=16)
    _fake_sdk(monkeypatch, [_claude("", stop="refusal")])
    with pytest.raises(ProviderError, match="declined"):
        ChatClient(spec).complete("x", min_output_tokens=16)


# ---- prices: every paid provider is priced or refused --------------------------------------------

def test_prices_come_from_secrets_catalogue_or_the_claude_list(monkeypatch, tmp_path):
    monkeypatch.setattr("hookrank.budget.fetch_prices", lambda base_url, api_key, timeout=30: {"claude-opus-5-5": Price(9, 9)})
    assert parse_prices("gpt-5.5=1.25/10, gemini:gemini-3.5-pro=0/0") == {
        "gpt-5.5": Price(1.25e-6, 10e-6), "gemini-3.5-pro": Price(0.0, 0.0)}
    for bad in ("gpt-5.5=1.25", "gpt-5.5", "=1/2", "m=-1/2"):
        with pytest.raises(ProviderError):
            parse_prices(bad)
    s = studio.studio_settings({**KEYS, "HOOKRANK_STORY_MODEL": "anthropic:claude-opus-5-5",
                                "HOOKRANK_ANNOTATOR_MODELS": "openai:gpt-5.5,nebius:openai/gpt-oss-120b",
                                "HOOKRANK_VERIFIER_MODEL": "openai:gpt-5.5", "HOOKRANK_PRICES": "claude-opus-5-5=3/15"}, studio.Path("/dev/null"), 0.5)
    prices = resolve_prices(s)
    assert prices["claude-opus-5-5"] == Price(3e-6, 15e-6)                       # secrets override list and catalogue
    assert unpriced_models(s, prices) == ["openai:gpt-5.5", "nebius:openai/gpt-oss-120b"]   # GPT needs a price
    with pytest.raises(studio.UploadError, match="HOOKRANK_PRICES"):
        studio.check_pricing(s, prices)
    guard = BudgetGuard(1.0, prices, tmp_path / "l.sqlite3", (spec_for("openai:gpt-5.5", s).base_url,))
    with pytest.raises(BudgetExceeded, match="HOOKRANK_PRICES"):
        guard.reserve(spec_for("openai:gpt-5.5", s).base_url, "gpt-5.5", 10, 10)
    guard.close()


def test_estimate_prices_each_role_separately():
    from conftest import SCRIPT
    from hookrank.parsing import parse_episode
    ep = parse_episode(SCRIPT.format(n=1), "e1.txt", "s")
    s = studio.studio_settings({**KEYS, "HOOKRANK_STORY_MODEL": "anthropic:claude-opus-5-5"}, studio.Path("/dev/null"), 0.5)
    cheap = {"openai/gpt-oss-120b": Price(1e-7, 1e-7), "Qwen/Qwen3-235B-A22B-Instruct-2507": Price(1e-7, 1e-7)}
    with_claude = studio.estimate([ep], s, {**cheap, **resolve_prices(Settings.from_mapping({}))})
    without = studio.estimate([ep], studio.studio_settings({**KEYS}, studio.Path("/dev/null"), 0.5), cheap)
    assert with_claude.usd > without.usd > 0


def test_extract_json_accepts_fenced_object_and_rejects_array():
    assert extract_json('```json\n{"ok":true}\n```') == {"ok": True}
    with pytest.raises(ProviderError):
        extract_json('[{"ok":true}]')


# ---- any OpenAI-compatible API: presets, providers added by name, other auth headers, local models -----

@pytest.mark.parametrize("ref, url_part", [
    ("openrouter:anthropic/claude-opus-5.5", "openrouter.ai/api/v1"),
    ("deepseek:deepseek-chat", "api.deepseek.com"),
    ("mistral:mistral-large-latest", "api.mistral.ai"),
    ("together:meta-llama/Llama-4-70B", "api.together.xyz"),
    ("xai:grok-4", "api.x.ai"),
])
def test_preset_providers_need_only_a_key(ref, url_part):
    name = ref.split(":")[0].upper()
    s = Settings.from_mapping({f"{name}_API_KEY": "k", "HOOKRANK_STORY_MODEL": ref,
                               "HOOKRANK_ANNOTATOR_MODELS": ref, "HOOKRANK_VERIFIER_MODEL": ref})
    spec = spec_for(ref, s)
    assert url_part in spec.base_url and spec.api_key == "k" and spec.model == ref.split(":", 1)[1]
    assert s.missing_keys() == []


def test_any_api_can_be_added_by_name_with_its_own_auth_header(monkeypatch):
    s = Settings.from_mapping({"KUKU_BASE_URL": "https://llm.kuku.example/openai/v1", "KUKU_API_KEY": "kk",
                               "KUKU_AUTH_HEADER": "api-key", "KUKU_CONCURRENCY": "8",
                               "HOOKRANK_STORY_MODEL": "kuku:frontier-model"})
    assert "kuku" in s.provider_names and s.split_model("kuku:frontier-model") == ("kuku", "frontier-model")
    assert s.split_model("qwen2.5:3b") == ("openai", "qwen2.5:3b")            # undeclared prefix: part of the ID
    spec = spec_for("kuku:frontier-model", s)
    assert spec.concurrency == 8 and spec.auth_header == "api-key"
    sent = {}

    class Response:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}'

    def fake_urlopen(request, timeout):
        sent["url"], sent["headers"] = request.full_url, dict(request.header_items())
        return Response()
    monkeypatch.setattr("hookrank.llm.urllib.request.urlopen", fake_urlopen)
    ChatClient(spec).complete("hi", min_output_tokens=16)
    assert sent["url"] == "https://llm.kuku.example/openai/v1/chat/completions"
    assert sent["headers"].get("Api-key") == "kk" and "Authorization" not in sent["headers"]


def test_local_models_need_no_key_and_cost_nothing():
    s = Settings.from_mapping({"HOOKRANK_STORY_MODEL": "ollama:llama3.3", "HOOKRANK_ANNOTATOR_MODELS": "ollama:qwen3",
                               "HOOKRANK_VERIFIER_MODEL": "ollama:qwen3", "HOOKRANK_FREE_ENDPOINTS": ""})
    e = s.endpoint("ollama")
    assert s.missing_keys() == [] and not e.key_required and not e.paid
    ChatClient(spec_for("ollama:qwen3", s))                                    # no "missing key" error


def test_openrouter_prices_load_from_its_catalogue(monkeypatch):
    seen = []
    monkeypatch.setattr("hookrank.budget.fetch_prices", lambda base_url, api_key, timeout=30: seen.append(base_url) or {
        "anthropic/claude-opus-5.5": Price(4e-6, 20e-6)})
    s = studio.studio_settings({"NEBIUS_API_KEY": "nb", "OPENROUTER_API_KEY": "or",
                                "HOOKRANK_STORY_MODEL": "openrouter:anthropic/claude-opus-5.5"}, studio.Path("/dev/null"), 0.5)
    prices = resolve_prices(s)
    assert any("openrouter.ai" in u for u in seen) and "anthropic/claude-opus-5.5" in prices
    assert "openrouter:anthropic/claude-opus-5.5" not in unpriced_models(s, prices)
    assert studio.provider_labels(s) == ["Nebius Token Factory", "OpenRouter"]


def test_catalogue_entries_with_variable_prices_are_not_trusted(monkeypatch):
    import io, json
    from hookrank.budget import fetch_prices
    body = {"data": [{"id": "fixed", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                     {"id": "router/auto", "pricing": {"prompt": "-1", "completion": "-1"}}]}
    monkeypatch.setattr("hookrank.budget.urllib.request.urlopen", lambda req, timeout: io.BytesIO(json.dumps(body).encode()))
    assert set(fetch_prices("https://x/v1", "k")) == {"fixed"}


def test_app_reads_a_provider_added_by_name_from_secrets(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(tmp_path)
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    at = AppTest.from_file(str(studio.Path(__file__).resolve().parent.parent / "streamlit_app.py"), default_timeout=30)
    for k, v in {"NEBIUS_API_KEY": "nb-secret", "KUKU_BASE_URL": "https://llm.kuku.example/v1", "KUKU_API_KEY": "kk-secret",
                 "HOOKRANK_VERIFIER_MODEL": "kuku:frontier-model"}.items():
        at.secrets[k] = v
    at.run()
    text = " ".join(str(getattr(el, "value", "")) for kind in ("caption", "markdown") for el in getattr(at, kind))
    assert "Clip check: frontier-model (kuku)" in text and "kk-secret" not in text and not at.exception
