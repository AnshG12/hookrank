"""Clip-level verification of shortlisted candidates (decisions D13, D14).

Event labels say what single events do. Whether a cut works as a promo is a clip-level question:
the opening must work in its first 3 seconds, the ending must leave a consequential question that is
not already answered inside the clip, and a stranger must be able to follow it. Labels are evidence
of *potential*; this step checks the actual window.

The verifier sees ONLY the clip (promo-relative times, dialogue, action, camera and sound text):
no story memory, logline or annotator rationale. It is NOT asked for verdicts - S7 showed two models
accepting 11-12 of 12 diagnostic cuts when asked "does this pass?". It classifies and extracts facts,
and Python decides:

  * opening: the first-3s event type from a fixed list; strong types pass only with cited evidence that
    starts inside the first 3s (the hook offset is computed from the script); routine activity, scene
    setting or a reply to an unseen question fail;
  * ending: the central question, every event INSIDE the clip that settles it, and the final-5s beat type;
    it fails if anything settles the question, if the final beat is a reaction after resolution, closure
    or music/camera only, if the question is missing or generic, or if no evidence ends in the last 5s;
  * standalone: missing context is reported as advisory only (S7: non-discriminating); cut quality: no
    jump problem at a [CUT];
  * recommendation: reject if opening, ending or cut fail; else accept;
  * every cited event must exist in the clip; unknown categories make the check "unverified";
  * failed checks scale the matching quality components (opening -> hook, ending -> cliffhanger),
    keeping the label-based values visible for audit.

Selection runs in rounds: rank, verify the unverified candidates in the provisional shortlist, drop
rejects, re-rank (so alternative cuts around the same story beat can enter), until the shortlist is
verified or the check budget is spent. Unverified answers (call or JSON failure) are marked as such,
never as pass.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from ..calls import CachedCaller, CallRecord
from ..llm import ModelSpec
from ..parsing import fmt_ms
from ..prompts import render
from ..providers import ProviderError, extract_json
from ..schemas import Episode, Event
from . import scoring as cfg
from .rank import rank_promos
from .schemas import Promo

STAGE = "clip_check"
OPENING_WINDOW_MS = 3_000
ENDING_WINDOW_MS = 5_000
MIN_OUTPUT = 600
TIERS = {"accept": 0, "revise": 1, "unverified": 2, "not_checked": 2}
FAIL_FACTOR = {"opening": ("hook", 0.4), "ending": ("cliffhanger", 0.4)}
_GENERIC_QUESTION = re.compile(r"what (happens|will happen) next|what will \w+(?: \w+)? do(?: next)?\s*\??$", re.I)


class ClipCheck(BaseModel):
    model: str
    status: str = "verified"                 # "verified" | "unverified" (call or JSON failure)
    opening: dict = Field(default_factory=dict)       # pass, reason, evidence (event_ids), hook_offset_seconds
    ending: dict = Field(default_factory=dict)        # pass, reason, evidence, unresolved_question, answered_within_clip
    standalone: dict = Field(default_factory=dict)    # pass, reason, missing_context
    cut_quality: dict = Field(default_factory=dict)   # pass, reason
    issues: list[str] = Field(default_factory=list)
    recommendation: str = "unverified"       # accept | revise | reject | unverified
    call: CallRecord | None = None


def render_clip(promo: Promo, episodes: dict[str, Episode]) -> tuple[str, dict[str, tuple[Event, int, int]], int]:
    """Clip text with promo-relative times; short ID -> (event, relative start, relative end); duration."""
    lines: list[str] = []
    events: dict[str, tuple[Event, int, int]] = {}
    offset = 0
    for i, seg in enumerate(promo.segments):
        if i:
            lines.append("[CUT]")
        for e in episodes[seg.episode_id].events:
            if e.start_ms < seg.end_ms and e.end_ms > seg.start_ms:
                rel_start = offset + max(0, e.start_ms - seg.start_ms)
                rel_end = offset + min(seg.end_ms, e.end_ms) - seg.start_ms
                short = f"C{len(events) + 1:02d}"
                events[short] = (e, rel_start, rel_end)
                label = e.event_type if e.event_type != "DIALOGUE" else (e.speaker or "SPEAKER")
                if e.event_type == "DIALOGUE" and e.delivery:
                    label += f" ({e.delivery})"
                lines.append(f"{short} [{fmt_ms(rel_start)}-{fmt_ms(rel_end)}] {label}: {' '.join(e.content.split())}")
        offset += seg.end_ms - seg.start_ms
    return "\n".join(lines), events, offset


# The verifier classifies and extracts; Python decides (S7: asked for pass/fail verdicts, both candidate
# models accepted 11-12 of 12 diagnostic cuts, including a clip whose question is answered on screen).
OPENING_TYPES_STRONG = {"attack_or_danger", "confrontation_or_accusation", "shocking_statement", "revelation",
                        "emotional_breakdown", "intimacy_or_romance", "humiliation", "proposal_or_decision",
                        "mystery_or_discovery"}
OPENING_TYPES_WEAK = {"routine_activity", "scene_setting", "reply_to_unseen_question", "other"}
ENDING_TYPES_OPEN = {"action_about_to_happen", "question_or_demand_awaiting_answer", "threat_hanging",
                     "revelation_pending", "decision_pending"}
ENDING_TYPES_CLOSED = {"reaction_after_resolution", "calm_or_closure", "music_or_camera_only", "other"}


def _ids(value, events: dict) -> tuple[list[str], list[str]]:
    shorts = [str(v).strip().upper() for v in (value if isinstance(value, list) else [value] if value else [])]
    return [s for s in shorts if s in events], [s for s in shorts if s not in events]


def parse_check(raw: str, events: dict[str, tuple[Event, int, int]], duration_ms: int, model: str,
                has_cut: bool = False) -> ClipCheck:
    check = ClipCheck(model=model)
    try:
        data = extract_json(raw)
    except ProviderError as exc:
        check.status, check.issues = "unverified", [f"unparseable verifier answer: {exc}"]
        return check
    op = data.get("opening") if isinstance(data.get("opening"), dict) else {}
    en = data.get("ending") if isinstance(data.get("ending"), dict) else {}
    cq = data.get("central_question") if isinstance(data.get("central_question"), dict) else {}
    op_type, en_type = str(op.get("type") or "").strip(), str(en.get("type") or "").strip()
    if op_type not in OPENING_TYPES_STRONG | OPENING_TYPES_WEAK or en_type not in ENDING_TYPES_OPEN | ENDING_TYPES_CLOSED:
        check.status, check.issues = "unverified", [f"unknown opening/ending type ({op_type!r}, {en_type!r})"]
        return check

    def evidence(value, name: str) -> list[str]:
        valid, invalid = _ids(value, events)
        if invalid:
            check.issues.append(f"{name}: cited events not in the clip: {invalid}")
        return valid

    # opening: a strong type, evidenced inside the first 3 seconds
    op_ev = evidence(op.get("evidence_event_ids"), "opening")
    early = [s for s in op_ev if events[s][1] < OPENING_WINDOW_MS]
    op_pass = op_type in OPENING_TYPES_STRONG and bool(early)
    if op_type in OPENING_TYPES_STRONG and not early:
        check.issues.append("opening: no cited evidence starts within the first 3s")
    check.opening = {"pass": op_pass, "type": op_type, "reason": f"{op_type}: {str(op.get('what_happens') or '')[:160]}",
                     "evidence": [events[s][0].event_id for s in op_ev],
                     "hook_offset_seconds": round(min(events[s][1] for s in early) / 1000, 2) if early else None}
    # ending: an open final beat, nothing inside the clip settles the central question, specific question
    question = str(cq.get("text") or "").strip()
    answers = evidence(data.get("answering_event_ids"), "answering events")
    en_ev = evidence(en.get("evidence_event_ids"), "ending")
    en_pass = en_type in ENDING_TYPES_OPEN and not answers
    if answers:
        settled_by = "; ".join(f"{s} \u201c{' '.join(events[s][0].content.split())[:60]}\u201d" for s in answers[:2])
        check.issues.append(f"ending: the central question is settled inside the clip by {settled_by}")
    if en_pass and (not question or _GENERIC_QUESTION.search(question)):
        en_pass = False
        check.issues.append(f"ending: no specific central question ({question!r})")
    if en_pass and not any(events[s][2] >= duration_ms - ENDING_WINDOW_MS for s in en_ev):
        en_pass = False
        check.issues.append("ending: no cited evidence ends within the last 5s")
    check.ending = {"pass": en_pass, "type": en_type, "reason": f"{en_type}: {str(en.get('what_happens') or '')[:160]}",
                    "evidence": [events[s][0].event_id for s in en_ev], "unresolved_question": question[:160],
                    "answered_within_clip": bool(answers), "answering_events": [events[s][0].event_id for s in answers]}
    # standalone and cut quality
    missing = [str(m)[:120] for m in (data.get("missing_context") or []) if isinstance(m, str) and m.strip()][:5]
    # Advisory only: in S7 both models listed 3-5 "missing context" items for nearly every clip, strong ones
    # included, so the list does not discriminate. It is reported, never used to decide or to score.
    check.standalone = {"pass": None, "advisory": True, "missing_context": missing,
                        "reason": "; ".join(missing) if missing else "no gaps listed"}
    jump = str(data.get("jump_problem") or "none").strip()
    cut_ok = jump.lower() in ("none", "", "n/a", "no") or not has_cut
    check.cut_quality = {"pass": cut_ok, "reason": "no jump problem" if cut_ok else jump[:160]}
    check.recommendation = "reject" if not (op_pass and en_pass and cut_ok) else "accept"
    return check


def apply_check(promo: Promo, check: ClipCheck) -> None:
    """Attach the check and scale failed components (label values stay visible as *_label)."""
    promo.clip_check = check.model_dump(exclude={"call"})
    if check.status != "verified":
        return
    comps = promo.scores.components
    if promo.scores.label_quality is None:
        promo.scores.label_quality = promo.scores.quality
    for part, (component, factor) in FAIL_FACTOR.items():
        if getattr(check, part).get("pass") is False:
            base = comps.get(f"{component}_label", comps[component])
            comps[component] = round(base * factor, 3)
    promo.scores.quality = cfg.quality_total(comps)


def check_promos(promos: list[Promo], episodes: dict[str, Episode], caller: CachedCaller,
                 spec: ModelSpec) -> dict[str, ClipCheck]:
    def one(promo: Promo) -> tuple[str, ClipCheck]:
        text, events, duration = render_clip(promo, episodes)
        raw, record, _ = caller.call(STAGE, spec, render("clip_check.md", clip=text), promo.segments[0].episode_id,
                                     f"promo {promo.promo_id}", MIN_OUTPUT)
        check = parse_check(raw, events, duration, spec.model, has_cut=len(promo.segments) > 1) if raw is not None else ClipCheck(
            model=spec.model, status="unverified", issues=[f"verifier call failed: {record.error[:120]}"])
        check.call = record
        return promo.promo_id, check

    with ThreadPoolExecutor(max_workers=max(1, spec.concurrency)) as pool:
        return dict(pool.map(one, promos))


def _tier(promo: Promo, checks: dict[str, ClipCheck]) -> int:
    check = checks.get(promo.promo_id)
    if check is None:
        return TIERS["not_checked"]
    return TIERS["unverified"] if check.status != "verified" else TIERS[check.recommendation]


def select_verified(evaluated: list[Promo], episodes: dict[str, Episode], caller: CachedCaller, spec: ModelSpec,
                    top_n: int, max_checks: int) -> tuple[list[Promo], dict[str, ClipCheck]]:
    """Rank -> verify the shortlist -> drop rejects -> re-rank, within a budget of ``max_checks``."""
    by_id = {p.promo_id: p for p in evaluated}
    checks: dict[str, ClipCheck] = {}
    ranked: list[Promo] = []
    while True:
        pool = [p for p in evaluated if p.scores.valid
                and not (p.promo_id in checks and checks[p.promo_id].recommendation == "reject")]
        ranked = rank_promos(pool, top_n=top_n)
        todo = [p for p in ranked if p.promo_id not in checks][:max(0, max_checks - len(checks))]
        if not todo:
            break
        for pid, check in check_promos(todo, episodes, caller, spec).items():
            checks[pid] = check
            apply_check(by_id[pid], check)
    final = sorted(ranked, key=lambda p: _tier(p, checks))    # stable: diversity order kept within each tier
    for rank, promo in enumerate(final, 1):
        promo.rank = rank
        check = checks.get(promo.promo_id)
        if check is None:
            promo.rank_reason += " Provisional: not clip-checked (check budget spent)."
        elif check.status != "verified":
            promo.rank_reason += " Provisional: clip check could not be verified."
        elif check.recommendation == "accept":
            promo.rank_reason += " Clip check: accept."
        else:
            failed = [k for k in ("opening", "ending", "cut_quality") if getattr(check, k).get("pass") is False]
            promo.rank_reason += f" Clip check: revise ({', '.join(failed) or 'verifier recommendation'})."
    return final, checks


def select_episode_best(evaluated: list[Promo], checks: dict[str, ClipCheck], episodes: dict[str, Episode],
                        caller: CachedCaller, spec: ModelSpec, tries: int = 3) -> dict[str, Promo]:
    """Best clip-checked promo per episode: check each episode's top candidates in rounds (parallel across
    episodes), up to ``tries`` per episode, and keep the first accepted one. Episodes with no accepted
    candidate keep their best non-rejected candidate, marked provisional by its clip_check field."""
    by_episode: dict[str, list[Promo]] = {}
    for p in sorted((p for p in evaluated if p.scores.valid), key=lambda p: (-p.scores.quality, -p.scores.confidence)):
        by_episode.setdefault(p.segments[0].episode, []).append(p)
    accepted = lambda p: p.promo_id in checks and checks[p.promo_id].status == "verified" and checks[p.promo_id].recommendation == "accept"
    rejected = lambda p: p.promo_id in checks and checks[p.promo_id].recommendation == "reject"
    for _ in range(tries):
        todo = []
        for cands in by_episode.values():
            if any(accepted(p) for p in cands):
                continue
            nxt = next((p for p in cands if p.promo_id not in checks), None)
            if nxt is not None and sum(p.promo_id in checks for p in cands) < tries:
                todo.append(nxt)
        if not todo:
            break
        by_id = {p.promo_id: p for p in todo}
        for pid, check in check_promos(todo, episodes, caller, spec).items():
            checks[pid] = check
            apply_check(by_id[pid], check)
    best: dict[str, Promo] = {}
    for ep, cands in by_episode.items():
        pick = next((p for p in cands if accepted(p)), None) or next((p for p in cands if not rejected(p)), None)
        if pick is not None:
            best[ep] = pick
    return best
