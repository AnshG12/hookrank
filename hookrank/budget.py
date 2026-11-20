"""Hard spending cap for paid endpoints (decision D10).

* Prices come from, in order of precedence: the ``HOOKRANK_PRICES`` setting (USD per million
  tokens, any provider), the provider's own catalogue where one exists (Nebius Token Factory publishes
  ``pricing`` at ``GET /models?verbose=true``), and the built-in list of Claude prices below. A paid
  model with no price is refused: the guard fails closed rather than guessing.
* Before every paid call the guard *reserves* the worst case (estimated prompt tokens x input price
  + the full output budget x output price). If spent + reserved + this call would exceed the cap,
  the call is refused with ``BudgetExceeded`` and never sent.
* After the call the reservation is released and the *actual* cost (from the usage the provider
  reports, which includes reasoning tokens) is written to a ledger.
* The ledger is one SQLite file shared by every run and output folder, so the cap is cumulative
  and cannot be reset by switching ``--output-dir``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .providers import USER_AGENT, ProviderError

DEFAULT_LEDGER = Path("outputs") / "spend_ledger.sqlite3"


class BudgetExceeded(ProviderError):
    """The call could push total spend past the cap; it was not sent."""


@dataclass(frozen=True)
class Price:
    prompt_per_token: float
    completion_per_token: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.prompt_per_token + completion_tokens * self.completion_per_token


# Anthropic first-party list prices (USD per million tokens, input/output), as of 2026-11-03.
# Override or extend with HOOKRANK_PRICES when they change.
CLAUDE_PRICES_PER_MILLION = {
    "claude-fable-5-2": (8.0, 40.0), "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0), "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50), "claude-haiku-4-5": (1.0, 5.0),
}
# Providers whose /models endpoint publishes per-token prices as {"pricing": {"prompt", "completion"}}.
CATALOGUE_PROVIDERS = ("nebius", "openrouter")


def per_million(prompt_usd: float, completion_usd: float) -> Price:
    return Price(prompt_usd / 1e6, completion_usd / 1e6)


def parse_prices(spec: str, providers=None) -> dict[str, Price]:
    """Parse ``"model=in/out, model2=in/out"`` (USD per million tokens; ``provider:`` prefixes allowed).

    Example: ``"gpt-5.5=1.25/10, gemini:gemini-3.5-flash-lite=0/0"``. Malformed entries raise, so a
    typo can never silently leave a paid model unpriced or priced at zero.
    """
    from .config import PROVIDERS, split_model
    providers = providers or PROVIDERS
    prices = {}
    for entry in (e.strip() for e in spec.replace(";", ",").split(",")):
        if not entry:
            continue
        name, sep, value = entry.rpartition("=")
        try:
            prompt_usd, completion_usd = (float(x) for x in value.split("/"))
            if not sep or not name.strip() or prompt_usd < 0 or completion_usd < 0:
                raise ValueError
        except ValueError:
            raise ProviderError(f"HOOKRANK_PRICES entry {entry!r} is not 'model=input/output' (USD per million tokens).")
        prices[split_model(name.strip(), providers)[1]] = per_million(prompt_usd, completion_usd)
    return prices


def resolve_prices(settings) -> dict[str, Price]:
    """Prices for the configured models: built-in Claude list < provider catalogues < HOOKRANK_PRICES."""
    prices = {model: per_million(*usd) for model, usd in CLAUDE_PRICES_PER_MILLION.items()}
    for endpoint in settings.endpoints_in_use:
        if endpoint.name in CATALOGUE_PROVIDERS and endpoint.paid and endpoint.api_key:
            prices.update(fetch_prices(endpoint.base_url, endpoint.api_key))
    prices.update(parse_prices(settings.prices, settings.provider_names))
    return prices


def unpriced_models(settings, prices: dict[str, "Price"]) -> list[str]:
    """Configured paid models without a price (their calls would be refused)."""
    return [ref for ref in settings.model_refs
            if settings.endpoint(settings.split_model(ref)[0]).paid and settings.split_model(ref)[1] not in prices]


def fetch_prices(base_url: str, api_key: str, timeout: int = 30) -> dict[str, Price]:
    """Per-token prices from an OpenAI-compatible catalogue that publishes ``pricing`` (Nebius)."""
    request = urllib.request.Request(f"{base_url}/models?verbose=true",
                                     headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # network, HTTP or JSON errors all mean "no trustworthy prices"
        raise ProviderError(f"Could not load prices from {base_url}: {exc}") from exc
    prices = {}
    for model in data.get("data", []):
        pricing = model.get("pricing") or {}
        try:
            price = Price(float(pricing["prompt"]), float(pricing["completion"]))
        except (KeyError, TypeError, ValueError):
            continue
        if price.prompt_per_token >= 0 and price.completion_per_token >= 0:   # negative = variable price
            prices[model["id"]] = price
    return prices


class BudgetGuard:
    """Thread-safe cap on cumulative spend across all runs, backed by a SQLite ledger."""

    def __init__(self, cap_usd: float, prices: dict[str, Price], ledger_path: Path = DEFAULT_LEDGER,
                 paid_base_urls: tuple[str, ...] = (), run_label: str = "") -> None:
        self.cap_usd = cap_usd
        # Ledger rows are tagged with the run, so per-run spend stays exact even when runs overlap
        # (a ledger-total delta would also count other processes' spend).
        self.run_label = run_label
        self.prices = prices
        self.paid_base_urls = paid_base_urls
        self._lock = threading.Lock()
        self._reserved = 0.0
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(ledger_path, check_same_thread=False)
        self._db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS spend (
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          base_url TEXT NOT NULL, model TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
          completion_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL, label TEXT NOT NULL DEFAULT ''
        );""")
        self._db.commit()

    def is_paid(self, base_url: str) -> bool:
        return base_url in self.paid_base_urls

    def spent(self) -> float:
        with self._lock:
            return self._spent_locked()

    def _spent_locked(self) -> float:
        return float(self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM spend").fetchone()[0])

    def price(self, model: str) -> Price:
        price = self.prices.get(model)
        if price is None:
            raise BudgetExceeded(f"No price known for {model}; refusing a paid call without a price. Add it to "
                                 "HOOKRANK_PRICES as 'model=input/output' (USD per million tokens), or list its "
                                 "provider in HOOKRANK_FREE_ENDPOINTS if it is a free tier.")
        return price

    def reserve(self, base_url: str, model: str, prompt_tokens_est: int, max_output_tokens: int) -> float:
        """Reserve the worst-case cost of one call; raises BudgetExceeded if it might break the cap."""
        if not self.is_paid(base_url):
            return 0.0
        worst = self.price(model).cost(prompt_tokens_est, max_output_tokens)
        with self._lock:
            spent = self._spent_locked()
            if spent + self._reserved + worst > self.cap_usd:
                raise BudgetExceeded(f"Budget cap ${self.cap_usd:.2f}: spent ${spent:.4f}, in flight ${self._reserved:.4f}, "
                                     f"this call could cost up to ${worst:.4f}. Call not sent.")
            self._reserved += worst
        return worst

    def settle(self, reservation: float, base_url: str, model: str, prompt_tokens: int, completion_tokens: int,
               label: str | None = None) -> float:
        """Release a reservation and record the actual cost (0 for unpaid endpoints)."""
        if not self.is_paid(base_url):
            return 0.0
        cost = self.price(model).cost(prompt_tokens, completion_tokens)
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)
            self._db.execute("INSERT INTO spend(base_url, model, prompt_tokens, completion_tokens, cost_usd, label) "
                             "VALUES(?,?,?,?,?,?)", (base_url, model, prompt_tokens, completion_tokens, cost,
                                                     self.run_label if label is None else label))
            self._db.commit()
        return cost

    def spent_by_label(self, label: str) -> float:
        with self._lock:
            return float(self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM spend WHERE label=?",
                                          (label,)).fetchone()[0])

    def release(self, reservation: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)

    def summary(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), SUM(cost_usd) "
                                    "FROM spend GROUP BY model ORDER BY SUM(cost_usd) DESC").fetchall()
            spent = self._spent_locked()
        return {"cap_usd": self.cap_usd, "spent_usd": round(spent, 4), "remaining_usd": round(self.cap_usd - spent, 4),
                "by_model": [{"model": m, "calls": n, "prompt_tokens": p, "completion_tokens": c, "cost_usd": round(cost, 4)}
                             for m, n, p, c, cost in rows]}

    def close(self) -> None:
        with self._lock:
            self._db.close()


def build_guard(settings, run_label: str = "") -> BudgetGuard:
    """Guard for the configured models: every provider not declared free is paid and must be priced."""
    from .config import PROVIDERS
    paid = tuple(dict.fromkeys(e.base_url for e in (settings.endpoint(n) for n in PROVIDERS) if e.paid and e.base_url))
    return BudgetGuard(settings.budget_usd, resolve_prices(settings), Path(settings.spend_ledger),
                       paid_base_urls=paid, run_label=run_label)
