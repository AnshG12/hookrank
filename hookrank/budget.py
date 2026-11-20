"""Hard spending cap for paid endpoints (decision D10).

* Prices come from the provider's own catalogue (``GET /models?verbose=true`` -> ``pricing``
  per token, as Nebius Token Factory publishes it). A model with no published price is refused:
  the guard fails closed rather than guessing.
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
            prices[model["id"]] = Price(float(pricing["prompt"]), float(pricing["completion"]))
        except (KeyError, TypeError, ValueError):
            continue
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
            raise BudgetExceeded(f"No published price for {model}; refusing a paid call without a price.")
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
    """Guard for the configured models. Prices are fetched only if a paid (Nebius) model is used."""
    from .config import split_model   # local import keeps budget.py free of config at import time
    nebius = settings.endpoint("nebius")
    uses_paid = any(split_model(ref)[0] == "nebius" for ref in settings.model_refs)
    prices = fetch_prices(nebius.base_url, nebius.api_key) if uses_paid and nebius.api_key else {}
    return BudgetGuard(settings.budget_usd, prices, Path(settings.spend_ledger), paid_base_urls=(nebius.base_url,),
                       run_label=run_label)
