"""S3: does the annotation prompt make every model tag cliffhanger moments?

S2 found that qwen and Gemini flash-lite never used `open_question`; only gpt-oss did, so
cliffhanger evidence rested on one model. This script annotates the same episodes with every
candidate annotator under whatever prompt is current, and writes per-model tag statistics and
cross-model agreement. Run it once per prompt version with a different --label and compare.

The story model is fixed (Gemini flash-lite, D9), so story context is identical across
prompt versions and only the annotation prompt differs.

Usage: python experiments/s3_cliffhanger_prompt.py --label before [--episodes 1,2,4]
Outputs (screenplay-derived, git-ignored): outputs/spikes/s3/<label>.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hookrank.cache import CallCache  # noqa: E402
from hookrank.config import Settings  # noqa: E402
from hookrank.llm import spec_for  # noqa: E402
from hookrank.narrative import NarrativeEngine  # noqa: E402
from hookrank.parsing import chronology_key, load_episodes  # noqa: E402
from hookrank.promos import scoring as cfg  # noqa: E402

STORY = "gemini:gemini-3.5-flash-lite"
ANNOTATORS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "gemini:gemini-3.5-flash-lite"]
# Strong cliffhanger evidence only (open_question 1.0, threat 0.9). The anchor threshold
# CLIFF_MIN (0.55) admits almost every tagged event, which made a first metric uninformative.
CLIFF_CLASS = {t for t, w in cfg.CLIFF_WEIGHTS.items() if w >= 0.9}
HOOK_CLASS = {t for t, w in cfg.HOOK_WEIGHTS.items() if w >= cfg.HOOK_MIN}
ENDING_WINDOW_MS = 10_000


def near(a, b, gap: int = 1000) -> bool:
    return a.start_ms <= b.end_ms + gap and b.start_ms <= a.end_ms + gap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--episodes", default="1,2,4")
    args = parser.parse_args()
    wanted = {int(x) for x in args.episodes.split(",")}
    settings = Settings.from_env()
    episodes = sorted((e for e in load_episodes(ROOT / "data", "broken-soldier")[0] if chronology_key(e)[0] in wanted),
                      key=chronology_key)
    cache = CallCache(ROOT / "outputs" / "spikes" / "s3" / "hookrank.sqlite3")
    result = NarrativeEngine(spec_for(STORY, settings), [spec_for(m, settings) for m in ANNOTATORS],
                             cache, "broken-soldier").run(episodes)
    cache.close()
    prompt_sha = hashlib.sha256((ROOT / "prompts" / "narrative_annotate.md").read_bytes()
                                + (ROOT / "hookrank" / "narrative" / "vocab.py").read_bytes()).hexdigest()[:12]
    report = {"label": args.label, "prompt_sha": prompt_sha, "episodes": [], "totals": {}}
    totals: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for ep in episodes:
        n = result.episodes[ep.episode_id]
        by_id = {e.event_id: e for e in ep.events}
        ending_ids = {e.event_id for e in ep.events if e.end_ms >= ep.duration_ms - ENDING_WINDOW_MS}
        row = {"episode": ep.episode_number, "events": len(ep.events), "models": {}, "calls_failed": sum(c.failed for c in n.calls)}
        sets = {}
        for a in n.annotations:
            name = a.model.split("/")[-1]
            tags = collections.Counter(t for x in a.annotations for t in x.tags)
            cliff = [by_id[x.event_id] for x in a.annotations if CLIFF_CLASS & set(x.tags)]
            hook = [by_id[x.event_id] for x in a.annotations if HOOK_CLASS & set(x.tags)]
            sets[name] = {"cliff": cliff, "hook": hook}
            oq = [x for x in a.annotations if "open_question" in x.tags]
            row["models"][name] = {
                "ok": a.ok, "annotated": len(a.annotations), "open_question": tags["open_question"],
                "open_question_with_question": sum(1 for x in oq if x.question), "cliff_class_events": len(cliff),
                "hook_class_events": len(hook), "ending_has_cliff": any(e.event_id in ending_ids for e in cliff),
                "open_question_share": round(tags["open_question"] / max(1, len(a.annotations)), 2),
                "issues": dict(collections.Counter(i.kind for i in a.issues)), "tags": dict(tags.most_common()),
            }
            for k in ("annotated", "open_question", "cliff_class_events", "hook_class_events"):
                totals[name][k] += row["models"][name][k]
            totals[name]["episodes_with_ending_cliff"] += row["models"][name]["ending_has_cliff"]
        names = list(sets)
        # Agreement: share of A's strong-cliff events with a strong-cliff event of B within 1s,
        # against chance: the share of ALL events that would match B's set by the same rule.
        row["cliff_agreement"] = {}
        for a in names:
            for b in names:
                if a == b or not sets[a]["cliff"]:
                    continue
                hit = sum(any(near(e, o) for o in sets[b]["cliff"]) for e in sets[a]["cliff"])
                chance = sum(any(near(e, o) for o in sets[b]["cliff"]) for e in ep.events) / len(ep.events)
                row["cliff_agreement"][f"{a} vs {b}"] = f"{hit}/{len(sets[a]['cliff'])} (chance {chance:.0%})"
        report["episodes"].append(row)
    report["totals"] = {k: dict(v) for k, v in totals.items()}
    out = ROOT / "outputs" / "spikes" / "s3" / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["totals"], indent=1))
    for row in report["episodes"]:
        print(row["episode"], {m: (v["open_question"], v["cliff_class_events"], v["ending_has_cliff"]) for m, v in row["models"].items()},
              "| cliff agreement:", row["cliff_agreement"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
