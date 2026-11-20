"""S7: does the D14 clip check discriminate on the user's diagnostic cases, and with which model?

Runs the clip check (prompts/clip_check.md, enforced by hookrank.promos.verify.parse_check) on the exact
baseline cuts named by the user, with each candidate verifier model. Expectations are the user's stated
hypotheses (docs/decisions.md S7), not ground truth. Outputs: outputs/spikes/s7/report.json (git-ignored).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hookrank.budget import build_guard  # noqa: E402
from hookrank.cache import CallCache  # noqa: E402
from hookrank.calls import CachedCaller  # noqa: E402
from hookrank.config import Settings  # noqa: E402
from hookrank.llm import spec_for  # noqa: E402
from hookrank.promos.schemas import Promo  # noqa: E402
from hookrank.promos.verify import check_promos  # noqa: E402
from hookrank.schemas import Episode  # noqa: E402

# (episode, first-segment start, run folder, user's hypothesis)
CASES = [("Episode 47", "01:21.80", "d13", "weak: ending resolved"), ("Episode 21", "00:58.20", "d13", "weak: routine opening"),
         ("Episode 17", "00:09.30", "d13", "weak: routine opening"), ("Episode 1", "01:19.00", "d13", "weak: routine opening"),
         ("Episode 9", "00:02.00", "d13", "open: sound-effect opening"), ("Episode 24", "00:32.50", "d13", "mixed: weak ending?"),
         ("Episode 3", "01:21.80", "d13", "strong"), ("Episode 18", "00:56.50", "d13", "strong"),
         ("Episode 32", "01:56.90", "d13", "strong"), ("Episode 43", "01:09.00", "d13", "strong"),
         ("Episode 34", "01:25.00", "d13", "strong (two segments)"), ("Episode 1", "00:35.40", "d11", "strong (no action)")]
RUNS = {"d13": ROOT / "outputs" / "broken-soldier-nebius-d13", "d11": ROOT / "outputs" / "broken-soldier"}
MODELS = ["nebius:openai/gpt-oss-120b", "nebius:Qwen/Qwen3-235B-A22B-Instruct-2507"]


def main() -> int:
    settings = Settings.from_env()
    episodes = {e["episode_id"]: Episode.model_validate(e) for e in json.loads((RUNS["d13"] / "episodes.json").read_text())}
    promos = []
    for epnum, start, run, hypothesis in CASES:
        allp = json.loads((RUNS[run] / "promos_all.json").read_text())
        raw = next(p for p in allp if p["segments"][0]["episode"] == epnum and p["segments"][0]["start"] == start)
        raw.pop("clip_check", None)
        promos.append((Promo.model_validate(raw), hypothesis))
    guard = build_guard(settings, run_label="s7-calibration")
    cache = CallCache(ROOT / "outputs" / "spikes" / "s7" / "hookrank.sqlite3")
    report = {}
    for ref in MODELS:
        checks = check_promos([p for p, _ in promos], episodes, CachedCaller(cache, "broken-soldier", budget=guard),
                              spec_for(ref, settings))
        rows = []
        for promo, hypothesis in promos:
            c = checks[promo.promo_id]
            rows.append({"case": f"{promo.segments[0].episode} {promo.segments[0].start}", "hypothesis": hypothesis,
                         "recommendation": c.recommendation if c.status == "verified" else "unverified",
                         "opening": c.opening.get("pass"), "ending": c.ending.get("pass"), "standalone": c.standalone.get("pass"),
                         "cut": c.cut_quality.get("pass"), "opening_type": c.opening.get("type"), "ending_type": c.ending.get("type"),
                         "question": c.ending.get("unresolved_question"), "answered_within_clip": c.ending.get("answered_within_clip"),
                         "missing": c.standalone.get("missing_context"), "issues": c.issues})
        report[ref] = rows
    report["spend_usd"] = guard.spent_by_label("s7-calibration")
    cache.close(); guard.close()
    (ROOT / "outputs" / "spikes" / "s7" / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for ref in MODELS:
        print(f"\n=== {ref}")
        for r in report[ref]:
            flags = "".join("P" if r[k] else ("F" if r[k] is False else "?") for k in ("opening", "ending", "standalone", "cut"))
            print(f"  {r['case']:<22} {r['hypothesis']:<28} -> {r['recommendation']:<10} [o/e/s/c {flags}] "
                  f"open={r['opening_type']} end={r['ending_type']} answered={r['answered_within_clip']} missing={len(r['missing'] or [])}")
    print(f"\nS7 spend: ${report['spend_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
