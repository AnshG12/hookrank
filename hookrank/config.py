"""Runtime settings, read from the environment, a local .env file, or (in the web app) Streamlit secrets.

Models are addressed as ``provider:model``, for example ``anthropic:claude-opus-5-5``,
``openai:gpt-5.5``, ``gemini:gemini-3.5-flash-lite`` or ``nebius:openai/gpt-oss-120b``. A model
without a known provider prefix uses ``openai`` (OPENAI_BASE_URL), so model IDs that contain a colon
(e.g. ``qwen2.5:3b``) stay intact.

Providers come from two places:

* presets in ``PROVIDERS`` (OpenAI, Anthropic, Gemini, Nebius, Groq, OpenRouter, Together, Fireworks,
  DeepSeek, Mistral, xAI, Cerebras, Hugging Face, local Ollama, and a generic ``custom`` slot);
* any other name: setting ``<NAME>_BASE_URL`` (plus ``<NAME>_API_KEY``) makes ``name:model`` usable,
  for any API that speaks the OpenAI Chat Completions format. No code change is needed.

Each provider is configured by variables named after it: ``<NAME>_API_KEY``, ``<NAME>_BASE_URL``,
``<NAME>_AUTH_HEADER`` (default bearer token; e.g. ``api-key`` for Azure OpenAI), ``<NAME>_CONCURRENCY``,
``<NAME>_REQUEST_TOKEN_LIMIT`` and ``<NAME>_MAX_OUTPUT_TOKENS``. Swapping a model is a configuration
change only: set the provider's key and point a role (HOOKRANK_STORY_MODEL, HOOKRANK_ANNOTATOR_MODELS,
HOOKRANK_VERIFIER_MODEL) at ``provider:model``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"   # docs.tokenfactory.nebius.com quickstart
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

CHAT_COMPLETIONS = "chat_completions"     # OpenAI-style POST {base_url}/chat/completions
ANTHROPIC_MESSAGES = "anthropic_messages"  # Claude Messages API, through the official Anthropic SDK


@dataclass(frozen=True)
class Provider:
    """Static description of one model provider; deployment-specific values come from configuration."""
    name: str
    label: str
    base_url: str
    api: str
    request_token_limit: int     # per-request ceiling (prompt + max output) used to size output budgets
    max_output_tokens: int
    concurrency: int             # parallel requests per model; free tiers stay at 1 (quota-bound)
    free_by_default: bool = False
    key_required: bool = True


# Base URLs checked on 2026-11-20 (each answers GET /models: 200, or 401/403 without a key).
PROVIDERS: dict[str, Provider] = {p.name: p for p in (
    Provider("openai", "OpenAI", OPENAI_BASE_URL, CHAT_COMPLETIONS, 64000, 16000, 4),
    Provider("anthropic", "Anthropic (Claude)", ANTHROPIC_BASE_URL, ANTHROPIC_MESSAGES, 64000, 16000, 4),
    Provider("gemini", "Google Gemini", GEMINI_BASE_URL, CHAT_COMPLETIONS, 32000, 8000, 1, free_by_default=True),
    Provider("nebius", "Nebius Token Factory", NEBIUS_BASE_URL, CHAT_COMPLETIONS, 32000, 8000, 6),
    Provider("groq", "Groq", GROQ_BASE_URL, CHAT_COMPLETIONS, 8000, 3000, 1, free_by_default=True),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", CHAT_COMPLETIONS, 64000, 16000, 4),
    Provider("together", "Together AI", "https://api.together.xyz/v1", CHAT_COMPLETIONS, 32000, 8000, 4),
    Provider("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1", CHAT_COMPLETIONS, 32000, 8000, 4),
    Provider("deepseek", "DeepSeek", "https://api.deepseek.com/v1", CHAT_COMPLETIONS, 32000, 8000, 4),
    Provider("mistral", "Mistral AI", "https://api.mistral.ai/v1", CHAT_COMPLETIONS, 32000, 8000, 4),
    Provider("xai", "xAI (Grok)", "https://api.x.ai/v1", CHAT_COMPLETIONS, 64000, 16000, 4),
    Provider("cerebras", "Cerebras", "https://api.cerebras.ai/v1", CHAT_COMPLETIONS, 32000, 8000, 4),
    Provider("huggingface", "Hugging Face", "https://router.huggingface.co/v1", CHAT_COMPLETIONS, 32000, 8000, 4),
    Provider("ollama", "Ollama (local)", "http://localhost:11434/v1", CHAT_COMPLETIONS, 32000, 8000, 1,
             free_by_default=True, key_required=False),
    Provider("custom", "OpenAI-compatible gateway", "", CHAT_COMPLETIONS, 32000, 8000, 4),
)}
ENDPOINT_NAMES = tuple(PROVIDERS)
DEFAULT_ENDPOINT = "openai"

# Free-tier defaults for the command line (decisions D1, D4, S2, D9). The web app uses its own defaults.
DEFAULT_STORY_MODEL = "gemini:gemini-3.5-flash-lite"
DEFAULT_ANNOTATOR_MODELS = ("groq:openai/gpt-oss-120b", "gemini:gemini-3.5-flash-lite")
DEFAULT_VERIFIER_MODEL = "groq:openai/gpt-oss-120b"

ROLE_VARIABLES = ("HOOKRANK_STORY_MODEL", "HOOKRANK_ANNOTATOR_MODELS", "HOOKRANK_VERIFIER_MODEL")
PROVIDER_SUFFIXES = ("API_KEY", "BASE_URL", "AUTH_HEADER", "CONCURRENCY", "REQUEST_TOKEN_LIMIT", "MAX_OUTPUT_TOKENS")
SETTING_VARIABLES = (*ROLE_VARIABLES, "HOOKRANK_PRICES", "HOOKRANK_FREE_ENDPOINTS", "ANTHROPIC_EFFORT",
                     "HOOKRANK_BUDGET_USD", "HOOKRANK_SPEND_LEDGER")
# Variables for the preset providers (any other provider's variables are recognised by is_config_variable).
CONFIG_VARIABLES = (*SETTING_VARIABLES, *(f"{name.upper()}_{suffix}" for name in PROVIDERS for suffix in PROVIDER_SUFFIXES))
_PROVIDER_VARIABLE = re.compile(rf"^([A-Z][A-Z0-9_]*?)_({'|'.join(PROVIDER_SUFFIXES)})$")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")


def is_config_variable(key: str) -> bool:
    """True for every variable that configures models or providers, including providers added by name."""
    return key in SETTING_VARIABLES or bool(_PROVIDER_VARIABLE.match(key))


def is_local(base_url: str) -> bool:
    host = re.sub(r"^\w+://", "", base_url).split("/")[0].rsplit(":", 1)[0]
    return host in _LOCAL_HOSTS


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load a small .env file without overriding values already in the environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def split_model(ref: str, providers=PROVIDERS) -> tuple[str, str]:
    """'anthropic:claude-opus-5-5' -> ('anthropic', 'claude-opus-5-5'); 'qwen/x' -> ('openai', 'qwen/x').

    Only known provider names count as a prefix (pass ``Settings.provider_names`` to include providers
    added by configuration), so model IDs that contain a colon stay intact."""
    prefix, sep, rest = ref.partition(":")
    if sep and prefix in providers and rest:
        return prefix, rest
    return DEFAULT_ENDPOINT, ref


def split_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    key_variable: str
    request_token_limit: int
    max_output_tokens: int
    concurrency: int = 1
    api: str = CHAT_COMPLETIONS
    paid: bool = True
    label: str = ""
    auth_header: str = "bearer"      # "bearer" -> Authorization: Bearer <key>; otherwise the header name
    key_required: bool = True


@dataclass(frozen=True)
class Settings:
    # Provider variables (<NAME>_API_KEY, <NAME>_BASE_URL, ...) as sorted (name, value) pairs.
    config: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    story_model: str = DEFAULT_STORY_MODEL
    annotator_models: tuple[str, ...] = DEFAULT_ANNOTATOR_MODELS
    verifier_model: str = DEFAULT_VERIFIER_MODEL      # clip-level check of top candidates (D13)
    # Providers whose calls cost nothing (free tiers): their calls are never priced or blocked.
    free_endpoints: tuple[str, ...] = tuple(n for n, p in PROVIDERS.items() if p.free_by_default)
    # Extra or overriding prices, "model=input/output" in USD per million tokens (see budget.parse_prices).
    prices: str = ""
    # Optional Claude effort ("low" | "medium" | "high" | ...); empty means the model's own default.
    anthropic_effort: str = ""
    # Hard cap on cumulative spend at paid endpoints, across all runs (D10).
    budget_usd: float = 20.0
    spend_ledger: str = "outputs/spend_ledger.sqlite3"

    @classmethod
    def from_mapping(cls, values: Mapping[str, object], **overrides) -> "Settings":
        """Settings from any mapping of configuration variables (environment, .env or Streamlit secrets)."""
        get = lambda key, default="": str(values.get(key) or default).strip()
        config = tuple(sorted((k, get(k)) for k in values if _PROVIDER_VARIABLE.match(k) and get(k)))
        free = values.get("HOOKRANK_FREE_ENDPOINTS")
        kwargs = dict(
            config=config,
            story_model=get("HOOKRANK_STORY_MODEL", DEFAULT_STORY_MODEL),
            annotator_models=split_list(get("HOOKRANK_ANNOTATOR_MODELS", ",".join(DEFAULT_ANNOTATOR_MODELS))),
            verifier_model=get("HOOKRANK_VERIFIER_MODEL", DEFAULT_VERIFIER_MODEL),
            free_endpoints=split_list(str(free)) if free is not None else cls.free_endpoints,
            prices=get("HOOKRANK_PRICES"),
            anthropic_effort=get("ANTHROPIC_EFFORT"),
            budget_usd=float(get("HOOKRANK_BUDGET_USD", "20")),
            spend_ledger=get("HOOKRANK_SPEND_LEDGER", "outputs/spend_ledger.sqlite3"),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls.from_mapping(os.environ)

    def value(self, key: str, default: str = "") -> str:
        return dict(self.config).get(key, default)

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Preset providers plus any provider added by setting <NAME>_BASE_URL."""
        added = [k[:-len("_BASE_URL")].lower() for k, _ in self.config if k.endswith("_BASE_URL")]
        return tuple(dict.fromkeys([*PROVIDERS, *(n for n in added if n != "hookrank")]))

    def split_model(self, ref: str) -> tuple[str, str]:
        return split_model(ref, self.provider_names)

    def endpoint(self, name: str) -> Endpoint:
        prefix = name.upper()
        provider = PROVIDERS.get(name) or Provider(name, name, "", CHAT_COMPLETIONS, 32000, 8000, 4)
        base_url = self.value(f"{prefix}_BASE_URL", provider.base_url).rstrip("/")
        api_key = self.value(f"{prefix}_API_KEY")
        limit, output = provider.request_token_limit, provider.max_output_tokens
        if name == "openai":   # older configurations named the default endpoint's limits HOOKRANK_*
            limit = int(self.value("HOOKRANK_REQUEST_TOKEN_LIMIT", limit))
            output = int(self.value("HOOKRANK_MAX_OUTPUT_TOKENS", output))
        if name == "groq" and not api_key and "groq.com" in self.value("OPENAI_BASE_URL"):
            api_key = self.value("OPENAI_API_KEY")   # older .env files kept the Groq key under OPENAI_*
        # A provider listed as free is free under any name (e.g. an old OPENAI_BASE_URL pointing at Groq);
        # a model served on this machine costs nothing and needs no key.
        free_urls = {PROVIDERS[n].base_url for n in self.free_endpoints if n in PROVIDERS}
        local = bool(base_url) and is_local(base_url)
        paid = name not in self.free_endpoints and base_url not in free_urls and not local
        return Endpoint(name=name, base_url=base_url, api_key=api_key, key_variable=f"{prefix}_API_KEY",
                        request_token_limit=int(self.value(f"{prefix}_REQUEST_TOKEN_LIMIT", limit)),
                        max_output_tokens=int(self.value(f"{prefix}_MAX_OUTPUT_TOKENS", output)),
                        concurrency=max(1, int(self.value(f"{prefix}_CONCURRENCY", provider.concurrency))),
                        api=provider.api, paid=paid, label=provider.label,
                        auth_header=self.value(f"{prefix}_AUTH_HEADER", "bearer"),
                        key_required=provider.key_required and not local)

    @property
    def model_refs(self) -> list[str]:
        return list(dict.fromkeys([self.story_model, *self.annotator_models, self.verifier_model]))

    @property
    def endpoints_in_use(self) -> list[Endpoint]:
        return [self.endpoint(name) for name in dict.fromkeys(self.split_model(ref)[0] for ref in self.model_refs)]

    def missing_keys(self) -> list[str]:
        """Variables that the configured models need but are not set."""
        missing = set()
        for e in self.endpoints_in_use:
            if e.key_required and not e.api_key:
                missing.add(e.key_variable)
            if not e.base_url:
                missing.add(f"{e.name.upper()}_BASE_URL")
        return sorted(missing)
