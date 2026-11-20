"""Runtime settings, read from the environment and an optional local .env file.

Models are addressed as ``[endpoint:]model``. Without a prefix a model uses the default
OpenAI-compatible endpoint (OPENAI_BASE_URL, Groq by default); ``gemini:<model>`` uses Google's
OpenAI-compatible Gemini endpoint with GEMINI_API_KEY; ``nebius:<model>`` uses Nebius Token Factory
with NEBIUS_API_KEY (paid per token, without free-tier rate walls). Only known endpoint names count as a
prefix, so model IDs that contain a colon (e.g. ``qwen2.5:3b``) are left intact.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"   # docs.tokenfactory.nebius.com quickstart
# Story extraction on Gemini flash-lite offloads the quota-bound gpt-oss (decisions S2, D9).
DEFAULT_STORY_MODEL = "gemini:gemini-3.5-flash-lite"
# Two families (OpenAI open-weight on Groq, Google): gemini-lite replaced qwen after S3.
DEFAULT_ANNOTATOR_MODELS = ("openai/gpt-oss-120b", "gemini:gemini-3.5-flash-lite")
DEFAULT_ENDPOINT = "openai"
ENDPOINT_NAMES = (DEFAULT_ENDPOINT, "gemini", "nebius")


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


def split_model(ref: str) -> tuple[str, str]:
    """'gemini:gemini-3.5-flash-lite' -> ('gemini', 'gemini-3.5-flash-lite'); 'qwen/x' -> ('openai', 'qwen/x')."""
    prefix, sep, rest = ref.partition(":")
    if sep and prefix in ENDPOINT_NAMES and rest:
        return prefix, rest
    return DEFAULT_ENDPOINT, ref


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    key_variable: str
    request_token_limit: int     # per-request ceiling (prompt + max output)
    max_output_tokens: int
    concurrency: int = 1         # parallel requests per model; free tiers stay at 1 (quota-bound)


@dataclass(frozen=True)
class Settings:
    # Default OpenAI-compatible endpoint; matches the documented Groq free-tier setup (D4).
    openai_base_url: str = GROQ_BASE_URL
    openai_api_key: str = ""
    request_token_limit: int = 8000     # Groq free tier: per-request ceiling = tokens/minute
    max_output_tokens: int = 3000
    # Gemini OpenAI-compatible endpoint (S2). Large context, so no 8k ceiling.
    gemini_base_url: str = GEMINI_BASE_URL
    gemini_api_key: str = ""
    gemini_request_token_limit: int = 32000
    gemini_max_output_tokens: int = 8000
    # Nebius Token Factory OpenAI-compatible endpoint (paid per token).
    nebius_base_url: str = NEBIUS_BASE_URL
    nebius_api_key: str = ""
    nebius_request_token_limit: int = 32000
    nebius_max_output_tokens: int = 8000
    nebius_concurrency: int = 6   # Nebius allows ~400k-2M tokens/min per model (catalogue), so parallelise
    # Hard cap on cumulative spend at paid endpoints (Nebius), across all runs (D10).
    budget_usd: float = 20.0
    spend_ledger: str = "outputs/spend_ledger.sqlite3"
    # Narrative engine roles (docs/decisions.md D4/D6/D9), as [endpoint:]model references.
    story_model: str = DEFAULT_STORY_MODEL
    annotator_models: tuple[str, ...] = DEFAULT_ANNOTATOR_MODELS
    # Clip-level check of top candidates (D13): sees only the clip window.
    verifier_model: str = "openai/gpt-oss-120b"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", GROQ_BASE_URL).rstrip("/"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            request_token_limit=int(os.getenv("HOOKRANK_REQUEST_TOKEN_LIMIT", "8000")),
            max_output_tokens=int(os.getenv("HOOKRANK_MAX_OUTPUT_TOKENS", "3000")),
            gemini_base_url=os.getenv("GEMINI_BASE_URL", GEMINI_BASE_URL).rstrip("/"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_request_token_limit=int(os.getenv("GEMINI_REQUEST_TOKEN_LIMIT", "32000")),
            gemini_max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8000")),
            nebius_base_url=os.getenv("NEBIUS_BASE_URL", NEBIUS_BASE_URL).rstrip("/"),
            nebius_api_key=os.getenv("NEBIUS_API_KEY", ""),
            nebius_request_token_limit=int(os.getenv("NEBIUS_REQUEST_TOKEN_LIMIT", "32000")),
            nebius_max_output_tokens=int(os.getenv("NEBIUS_MAX_OUTPUT_TOKENS", "8000")),
            nebius_concurrency=int(os.getenv("NEBIUS_CONCURRENCY", "6")),
            budget_usd=float(os.getenv("HOOKRANK_BUDGET_USD", "20")),
            spend_ledger=os.getenv("HOOKRANK_SPEND_LEDGER", "outputs/spend_ledger.sqlite3"),
            story_model=os.getenv("HOOKRANK_STORY_MODEL", DEFAULT_STORY_MODEL),
            annotator_models=tuple(m.strip() for m in os.getenv(
                "HOOKRANK_ANNOTATOR_MODELS", ",".join(DEFAULT_ANNOTATOR_MODELS)).split(",") if m.strip()),
            verifier_model=os.getenv("HOOKRANK_VERIFIER_MODEL", "openai/gpt-oss-120b"),
        )

    def endpoint(self, name: str) -> Endpoint:
        if name == "gemini":
            return Endpoint("gemini", self.gemini_base_url, self.gemini_api_key, "GEMINI_API_KEY",
                            self.gemini_request_token_limit, self.gemini_max_output_tokens)
        if name == "nebius":
            return Endpoint("nebius", self.nebius_base_url, self.nebius_api_key, "NEBIUS_API_KEY",
                            self.nebius_request_token_limit, self.nebius_max_output_tokens, self.nebius_concurrency)
        return Endpoint(DEFAULT_ENDPOINT, self.openai_base_url, self.openai_api_key, "OPENAI_API_KEY",
                        self.request_token_limit, self.max_output_tokens)

    @property
    def model_refs(self) -> list[str]:
        return list(dict.fromkeys([self.story_model, *self.annotator_models, self.verifier_model]))

    def missing_keys(self) -> list[str]:
        """Key variables that the configured models need but are not set."""
        endpoints = {self.endpoint(split_model(ref)[0]) for ref in self.model_refs}
        return sorted({e.key_variable for e in endpoints if not e.api_key})
