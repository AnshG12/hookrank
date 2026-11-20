"""S4: choose the Nebius Token Factory annotator (protocol and selection rule: docs/decisions.md S4).

Every candidate annotates the same episodes twice (run A, run B) with identical Gemini story context;
Gemini flash-lite annotations are the fixed partner. Run B copies run A's cache and deletes only the
candidates' entries, so story context and partner labels are identical while candidates answer afresh.
All paid calls go through the shared $-cap ledger (D10).

Usage: python experiments/s4_nebius_models.py [--episodes 1,3,25,37,50]
Outputs (screenplay-derived, git-ignored): outputs/spikes/s4/
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hookrank.budget import BudgetGuard, fetch_prices  # noqa: E402
from hookrank.cache import CallCache  # noqa: E402
from hookrank.config import Settings  # noqa: E402
from hookrank.llm import spec_for  # noqa: E402
from hookrank.narrative import NarrativeEngine  # noqa: E402
from hookrank.parsing import chronology_key, load_episodes  # noqa: E402
from hookrank.promos import scoring as cfg  # noqa: E402

GEMINI = "gemini:gemini-3.5-flash-lite"
CANDIDATES = ["openai/gpt-oss-120b", "Qwen/Qwen3-235B-A22B-Instruct-2507", "deepseek-ai/DeepSeek-V4-Pro-0813",
              "zai-org/GLM-5.3", "moonshotai/Kimi-K3.1-Pro"]
STRONG_CLIFF = {t for t, w in cfg.CLIFF_WEIGHTS.items() if w >= 0.9}       # open_question, threat
HOOK = {t for t, w in cfg.HOOK_WEIGHTS.items() if w >= cfg.HOOK_MIN}
ENDING_MS = 10_000
OUT = ROOT / "outputs" / "spikes" / "s4"


def near(a, b, gap: int = 1000) -> bool:
    return a.start_ms <= b.end_ms + gap and b.start_ms <= a.end_ms + gap


def overlap_f1(xs, ys) -> float | None:
    """Symmetric tolerant overlap: share of moments in either set matched by the other set."""
    if not xs and not ys:
        return None
    hit = sum(any(near(x, y) for y in ys) for x in xs) + sum(any(near(y, x) for x in xs) for y in ys)
    return hit / (len(xs) + len(ys))


def run(label: str, db: Path, episodes, settings, guard) -> dict:
    cache = CallCache(db)
    specs = [spec_for(f"nebius:{m}", settings) for m in CANDIDATES] + [spec_for(GEMINI, settings)]
    result = NarrativeEngine(spec_for(GEMINI, settings), specs, cache, "broken-soldier", budget=guard).run(episodes)
    cache.close()
    return result.episodes


def main() -> int:
    global CANDIDATES
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="1,3,25,37,50")
    parser.add_argument("--models", default=",".join(CANDIDATES), help="Comma-separated Nebius model IDs to compare")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    CANDIDATES = [m.strip() for m in args.models.split(",") if m.strip()]
    wanted = {int(x) for x in args.episodes.split(",")}
    settings = Settings.from_env()
    neb = settings.endpoint("nebius")
    prices = fetch_prices(neb.base_url, neb.api_key)
    guard = BudgetGuard(settings.budget_usd, prices, Path(settings.spend_ledger), (neb.base_url,))
    spent_start = guard.spent()
    episodes = sorted((e for e in load_episodes(ROOT / "data", "broken-soldier")[0] if chronology_key(e)[0] in wanted),
                      key=chronology_key)
    OUT.mkdir(parents=True, exist_ok=True)
    db_a, db_b = OUT / "run_a.sqlite3", OUT / "run_b.sqlite3"
    runs = {"A": run("A", db_a, episodes, settings, guard)}
    if not db_b.exists():  # B = A's story + partner labels, candidates forced to answer again
        shutil.copy(db_a, db_b)
        con = sqlite3.connect(db_b)
        con.execute("DELETE FROM llm_calls WHERE model IN (%s) AND stage='annotation'" % ",".join("?" * len(CANDIDATES)),
                    CANDIDATES)
        con.commit()
        con.close()
    runs["B"] = run("B", db_b, episodes, settings, guard)

    report = {"episodes": [e.episode_number for e in episodes], "models": {}}
    for model in CANDIDATES + ["gemini-3.5-flash-lite"]:
        m = {"calls": 0, "calls_ok": 0, "truncated_then_split": 0, "failed": 0, "kept": 0, "unsupported_dropped": 0,
             "repaired": 0, "elided_quotes": 0, "annotated_per_ep": [], "strong_cliff_per_ep": [], "endings_tagged": 0,
             "open_question_share": [], "stability_hook": [], "stability_cliff": [], "agree_gemini_hook": [],
             "agree_gemini_cliff": [], "chance_cliff": [], "latency_s": [], "cost_usd": 0.0}
        for run_label, episodes_out in runs.items():
            for ep in episodes:
                n = episodes_out[ep.episode_id]
                by_id = {e.event_id: e for e in ep.events}
                ann = next((a for a in n.annotations if a.model == model), None)
                calls = [c for c in n.calls if c.model == model and c.stage == "annotation"]
                if run_label == "B" and model == "gemini-3.5-flash-lite":
                    continue  # partner is identical in B (cached); count it once
                m["calls"] += len(calls)
                m["calls_ok"] += sum(c.ok for c in calls)
                m["truncated_then_split"] += sum(c.superseded for c in calls)
                m["failed"] += sum(c.failed for c in calls)
                m["latency_s"] += [c.latency_s for c in calls if c.ok and c.latency_s is not None]
                price = prices.get(model)
                if price:
                    m["cost_usd"] += sum(price.cost(c.prompt_tokens or 0, c.completion_tokens or 0) for c in calls if c.ok)
                if ann is None:
                    continue
                kinds = Counter(i.kind for i in ann.issues)
                m["kept"] += len(ann.annotations)
                m["unsupported_dropped"] += kinds["quote_not_found"] + kinds["missing_quote"]
                m["repaired"] += kinds["id_repaired"]
                m["elided_quotes"] += sum(a.quote_match == "elided" for a in ann.annotations)
                if run_label == "A":
                    ends = {e.event_id for e in ep.events if e.end_ms >= ep.duration_ms - ENDING_MS}
                    cliff = [by_id[a.event_id] for a in ann.annotations if STRONG_CLIFF & set(a.tags)]
                    m["annotated_per_ep"].append(len(ann.annotations))
                    m["strong_cliff_per_ep"].append(len(cliff))
                    m["endings_tagged"] += any(e.event_id in ends for e in cliff)
                    m["open_question_share"].append(round(sum("open_question" in a.tags for a in ann.annotations)
                                                          / max(1, len(ann.annotations)), 2))
        # stability (A vs B) and agreement with the Gemini partner (run A)
        for ep in episodes:
            by_id = {e.event_id: e for e in ep.events}

            def moments(run_label, mdl, tags):
                a = next((x for x in runs[run_label][ep.episode_id].annotations if x.model == mdl), None)
                return [by_id[x.event_id] for x in (a.annotations if a else []) if tags & set(x.tags)]
            if model != "gemini-3.5-flash-lite":
                for key, tags in (("stability_hook", HOOK), ("stability_cliff", STRONG_CLIFF)):
                    m[key].append(overlap_f1(moments("A", model, tags), moments("B", model, tags)))
                for key, tags in (("agree_gemini_hook", HOOK), ("agree_gemini_cliff", STRONG_CLIFF)):
                    m[key].append(overlap_f1(moments("A", model, tags), moments("A", "gemini-3.5-flash-lite", tags)))
                gem = moments("A", "gemini-3.5-flash-lite", STRONG_CLIFF)
                m["chance_cliff"].append(sum(any(near(e, g) for g in gem) for e in ep.events) / len(ep.events))
        mean = lambda xs: round(sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs)), 2)
        total_claims = m["kept"] + m["unsupported_dropped"]
        report["models"][model] = {
            "calls_ok": f"{m['calls_ok']}/{m['calls']}", "failed": m["failed"], "truncated_then_split": m["truncated_then_split"],
            "verbatim_rate": round(m["kept"] / max(1, total_claims), 3), "repaired": m["repaired"],
            "elided_quotes": m["elided_quotes"], "annotated_per_ep": m["annotated_per_ep"],
            "strong_cliff_per_ep": m["strong_cliff_per_ep"], "endings_tagged": f"{m['endings_tagged']}/{len(episodes)}",
            "open_question_share_max": max(m["open_question_share"], default=0),
            "stability_hook": mean(m["stability_hook"]), "stability_cliff": mean(m["stability_cliff"]),
            "agree_gemini_hook": mean(m["agree_gemini_hook"]), "agree_gemini_cliff": mean(m["agree_gemini_cliff"]),
            "chance_cliff_vs_gemini": mean(m["chance_cliff"]),
            "latency_s_mean": round(sum(m["latency_s"]) / max(1, len(m["latency_s"])), 1),
            "cost_usd_measured": round(m["cost_usd"], 4)}
    report["spend"] = {"this_experiment_usd": round(guard.spent() - spent_start, 4), **guard.summary()}
    guard.close()
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
