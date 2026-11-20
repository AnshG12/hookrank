"""End-to-end run: parse -> narrative engine -> promo construction/ranking -> outputs.

Outputs (all screenplay-derived; keep local unless sharing is authorised):
  episodes.json            parsed episodes and parser diagnostics
  narrative.json           story model, verified annotations, consensus, every model call
  promos_ranked.json/.csv  the ranked list to post first (variety-aware)
  promos_by_episode.json   the best valid promo per episode (coverage)
  promos_all.json          every evaluated candidate, valid or not, with its checks
  promos_report.md         human-readable report for editors
  failures.json            parse diagnostics and failed model calls
  run_manifest.json        run ID, inputs, models, parameters, prompt hashes, counts
"""
from __future__ import annotations

import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .budget import build_guard
from .cache import CallCache
from .calls import CachedCaller
from .config import Settings
from .llm import spec_for
from .narrative import NarrativeEngine, NarrativeResult
from .parsing import chronology_key, load_episodes
from .prompts import PROMPT_DIR
from .promos import Promo, build_promos
from .promos import scoring as promo_scoring
from .promos.verify import select_episode_best, select_verified

LIMITATIONS = [
    "Timestamps are script time; the final edited video may differ (R1).",
    "Text cannot judge performance, visuals or audio; camera-described hooks are unverified on screen.",
    "Quality weights are provisional and not fitted to editor or audience data (D2, D5).",
    "Model agreement is reported as confidence, not as proof of quality.",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _spec_record(spec) -> dict:
    return {"model": spec.model, "endpoint": spec.base_url, "params": spec.params_dict(),
            "request_token_limit": spec.request_token_limit, "max_output_tokens": spec.max_output_tokens}


def _segments_text(p: Promo) -> str:
    return " + ".join(f"{s.episode} {s.start}-{s.end}" for s in p.segments)


def best_per_episode(evaluated: list[Promo]) -> dict[str, Promo]:
    best: dict[str, Promo] = {}
    for p in sorted((p for p in evaluated if p.scores.valid), key=lambda p: (-p.scores.quality, -p.scores.confidence)):
        best.setdefault(p.segments[0].episode, p)
    return best


def run(input_path: Path, output_dir: Path, series_id: str, settings: Settings | None = None, *,
        episode_numbers: set[int] | None = None, top_n: int = 25, cache_only: bool = False,
        memory_db: Path | None = None, verify_top: int = 80, episode_tries: int = 3) -> dict:
    settings = settings or Settings.from_env()
    started = datetime.now(timezone.utc)
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, read_failures = load_episodes(input_path, series_id=series_id)
    episodes.sort(key=chronology_key)
    if episode_numbers:
        episodes = [e for e in episodes if chronology_key(e)[0] in episode_numbers]
    if not episodes:
        raise ValueError(f"No episodes to process from {input_path}.")

    story_spec = spec_for(settings.story_model, settings)
    annotator_specs = [spec_for(m, settings) for m in settings.annotator_models]
    cache = CallCache(memory_db or output_dir / "hookrank.sqlite3")
    guard = build_guard(settings, run_label=run_id)
    checks: dict = {}
    by_episode_verified: dict | None = None
    try:
        result: NarrativeResult = NarrativeEngine(story_spec, annotator_specs, cache, series_id,
                                                  cache_only=cache_only, budget=guard).run(episodes)
        ranked, evaluated, stats = build_promos(result, top_n=top_n)
        if verify_top > 0:   # D13/D14: verify the shortlist on the clip text alone; rejects make room for alternatives
            verifier = spec_for(settings.verifier_model, settings)
            caller = CachedCaller(cache, series_id, cache_only=cache_only, budget=guard)
            episode_map = {e.episode_id: e for e in episodes}
            ranked, checks = select_verified(evaluated, episode_map, caller, verifier, top_n=top_n, max_checks=verify_top)
            if episode_tries > 0:   # every episode's recommended promo is clip-checked too
                by_episode_verified = select_episode_best(evaluated, checks, episode_map, caller, verifier, tries=episode_tries)
    finally:
        cache.close()
    spend = {**guard.summary(), "this_run_usd": round(guard.spent_by_label(run_id), 4)}
    guard.close()
    stats["ranked"] = len(ranked)
    by_episode = by_episode_verified if by_episode_verified is not None else best_per_episode(evaluated)
    by_episode = dict(sorted(by_episode.items(), key=lambda kv: int(kv[0].split()[-1]) if kv[0].split()[-1].isdigit() else 0))

    calls = [c for n in result.episodes.values() for c in n.calls] + [c.call for c in checks.values() if c.call]
    failed_calls = [{"episode": n.episode.episode_number, **c.model_dump()}
                    for n in result.episodes.values() for c in n.calls if c.failed]
    diagnostics = [f"{e.source_name}: {d}" for e in episodes for d in e.diagnostics]

    _dump(output_dir / "episodes.json", [e.model_dump() for e in episodes])
    _dump(output_dir / "narrative.json", result.to_json())
    _dump(output_dir / "promos_ranked.json", [p.model_dump() for p in ranked])
    _dump(output_dir / "promos_by_episode.json", {ep: p.model_dump() for ep, p in by_episode.items()})
    _dump(output_dir / "promos_all.json", [p.model_dump() for p in sorted(evaluated, key=lambda p: -p.scores.quality)])
    _dump(output_dir / "failures.json", {"unreadable_inputs": read_failures, "parse_diagnostics": diagnostics,
                                         "failed_model_calls": failed_calls})
    with (output_dir / "promos_ranked.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["rank", "promo_id", "kind", "segments", "duration_s", "hook_time", "hook", "cliffhanger_time",
                         "cliffhanger", "why", "quality", "confidence", "rank_reason"])
        for p in ranked:
            writer.writerow([p.rank, p.promo_id, p.kind, _segments_text(p), p.duration_seconds,
                             f"{p.hook.episode} {p.hook.time}", p.hook.text,
                             f"{p.cliffhanger.episode} {p.cliffhanger.time}", p.cliffhanger.text, p.why,
                             p.scores.quality, p.scores.confidence_label, p.rank_reason])
    rejected = sorted((p for p in evaluated if p.promo_id in checks and checks[p.promo_id].recommendation == "reject"),
                      key=lambda p: -(p.scores.label_quality or p.scores.quality))
    (output_dir / "promos_report.md").write_text(render_report(series_id, ranked, by_episode, stats, result, rejected),
                                                 encoding="utf-8")

    manifest = {
        "run_id": run_id, "series_id": series_id, "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "input": str(input_path),
        "input_hashes": {e.source_name: e.source_hash for e in episodes},
        "episodes": len(episodes), "events": sum(len(e.events) for e in episodes),
        "models": {"story": _spec_record(story_spec), "annotators": [_spec_record(s) for s in annotator_specs]},
        "prompt_hashes": {p.name: _sha(p) for p in sorted(PROMPT_DIR.glob("narrative_*.md"))},
        "scoring_config": {k: v for k, v in vars(promo_scoring).items() if k.isupper()},
        "model_calls": {"total": len(calls), "cached": sum(c.cached for c in calls),
                        "live": sum(c.ok and not c.cached for c in calls), "failed": len(failed_calls),
                        "truncated_then_split": sum(c.superseded for c in calls),
                        "prompt_tokens_live": sum(c.prompt_tokens or 0 for c in calls if c.ok and not c.cached),
                        "completion_tokens_live": sum(c.completion_tokens or 0 for c in calls if c.ok and not c.cached)},
        "narrative": {"characters": len(result.series.characters),
                      "principal_characters": sum(c.principal for c in result.series.characters),
                      "series_threads": len(result.series.threads),
                      "annotated_events": sum(len(n.consensus) for n in result.episodes.values()),
                      "episodes_with_two_annotators": sum(
                          1 for n in result.episodes.values() if sum(a.ok and bool(a.annotations) for a in n.annotations) >= 2)},
        "promos": stats, "spend": spend,
        "clip_check": {"model": settings.verifier_model if checks else None, "checked": len(checks),
                       "recommendations": {r: sum(c.recommendation == r for c in checks.values())
                                           for r in ("accept", "revise", "reject", "unverified")},
                       "failed": {k: sum(getattr(c, k).get("pass") is False for c in checks.values())
                                  for k in ("opening", "ending", "cut_quality")},
                       "shortlist_verified": sum(1 for p in ranked if p.clip_check and p.clip_check["status"] == "verified"),
                       "episode_picks_accepted": sum(1 for p in by_episode.values() if p.clip_check and p.clip_check.get("recommendation") == "accept"),
                       "episodes": len(by_episode)}, "cache_only": cache_only, "limitations": LIMITATIONS,
    }
    _dump(output_dir / "run_manifest.json", manifest)
    return {"manifest": manifest, "ranked": ranked, "by_episode": by_episode, "narrative": result}


def render_report(series_id: str, ranked: list[Promo], by_episode: dict[str, Promo], stats: dict,
                  result: NarrativeResult, rejected: list[Promo] | None = None) -> str:
    rejected = rejected or []
    lines = [f"# Promo recommendations: {series_id}", "",
             f"{stats['valid_candidates']} valid candidates from {stats['episodes_total']} episodes "
             f"({stats['episodes_with_valid_promo']} episodes have at least one). Ranked by quality estimate with a "
             "variety adjustment; confidence is shown separately and never substitutes for quality.", "",
             "Times are script time (MM:SS.cc from episode start).", "", "## Post first", ""]
    for p in ranked:
        lines += [f"### {p.rank}. {_segments_text(p)} ({p.duration_seconds:.0f}s, {p.kind.replace('_', ' ')})", "",
                  f"- **Hook** ({p.hook.episode} {p.hook.time}, {p.hook.event_type.lower()}"
                  f"{', ' + p.hook.speaker if p.hook.speaker else ''}): {p.hook.text}",
                  f"- **Cliffhanger** ({p.cliffhanger.episode} {p.cliffhanger.time}): {p.cliffhanger.text}",
                  f"- **Why it works:** {p.why}",
                  f"- **Why this rank:** {p.rank_reason}",
                  f"- Scores: quality {p.scores.quality:.0f}/100 · confidence {p.scores.confidence_label} "
                  f"({p.scores.confidence:.2f}) · " + ", ".join(f"{k} {v:.2f}" for k, v in p.scores.components.items())]
        if p.clip_check and p.clip_check["status"] == "verified":
            cc = p.clip_check
            mark = lambda part: "pass" if cc[part].get("pass") else "FAIL"
            lines.append(f"- Clip check: **{cc['recommendation']}** (opening {mark('opening')} - {cc['opening'].get('type')}, "
                         f"ending {mark('ending')} - {cc['ending'].get('type')}, cut {mark('cut_quality')})")
            if cc["standalone"].get("missing_context"):
                lines.append(f"- Missing context a new viewer may need (advisory): {'; '.join(cc['standalone']['missing_context'][:3])}")
            if cc["ending"].get("unresolved_question"):
                lines.append(f"- Unresolved at the cut: {cc['ending']['unresolved_question']}")
            failed = [f"{k}: {cc[k].get('reason')}" for k in ("opening", "ending", "cut_quality") if not cc[k].get("pass")]
            if failed or cc["issues"]:
                lines.append(f"- Clip-check findings: {'; '.join(failed + cc['issues'])}")
        elif p.clip_check:
            lines.append(f"- Clip check: unverified ({'; '.join(p.clip_check['issues'])})")
        lines.append(f"- Evidence: hook {p.hook.event_id} [{', '.join(p.hook.tags)}], cliffhanger {p.cliffhanger.event_id} "
                     f"[{', '.join(p.cliffhanger.tags)}]; raw quality {p.scores.quality:.0f}"
                     + (f" (labels alone {p.scores.label_quality:.0f})" if p.scores.label_quality not in (None, p.scores.quality) else "")
                     + (f", variety-adjusted {p.scores.variety_adjusted:.0f}" if p.scores.variety_adjusted is not None else ""))
        if p.notes:
            lines.append(f"- Notes: {'; '.join(p.notes)}")
        if p.scores.confidence_notes:
            lines.append(f"- Confidence caveats: {'; '.join(p.scores.confidence_notes)}")
        lines.append("")
    lines += ["## Best promo per episode", "", "| Episode | Segments | Quality | Confidence | Clip check | Why |",
              "|---|---|---|---|---|---|"]
    for ep, p in by_episode.items():
        status = (p.clip_check or {}).get("recommendation", "not checked")
        lines.append(f"| {ep} | {_segments_text(p)} | {p.scores.quality:.0f} | {p.scores.confidence_label} | {status} | {p.why} |")
    missing = [n.episode.episode_number for n in result.episodes.values() if n.episode.episode_number not in by_episode]
    if missing:
        lines += ["", f"No valid promo found for: {', '.join(missing)}."]
    if rejected:
        lines += ["", "## Rejected by the clip check", "",
                  "Candidates that ranked high on labels but failed the clip-level check (kept for diagnostics).", ""]
        for p in rejected:
            cc = p.clip_check
            failed = [f"{k} - {cc[k].get('reason')}" for k in ("opening", "ending", "cut_quality") if not cc[k].get("pass")]
            lines.append(f"- {_segments_text(p)} (labels alone {p.scores.label_quality or p.scores.quality:.0f}): "
                         + "; ".join(failed + cc["issues"]))
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in LIMITATIONS]
    return "\n".join(lines) + "\n"
