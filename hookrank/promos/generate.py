"""Promo construction and evidence-based scoring (Python only; no model calls).

Candidates are anchored on annotated events: the clip starts on a hook event (so the hook
lands in the first 3s) and is cut right after a cliffhanger event. Two-segment promos exist
for exactly two jobs: joining a strong hook to a cliffhanger more than 60s away within an
episode, or bridging into one of the next two episodes on the same series thread. Every
boundary is a legal cut point (never inside a dialogue line).

Each candidate gets validity checks, a quality estimate from tag evidence, and a separate
confidence from model agreement and data issues (docs/decisions.md D5; weights in scoring.py).
"""
from __future__ import annotations

import bisect
import hashlib
from collections import Counter
from dataclasses import dataclass, field

from ..narrative.schemas import ConsensusEvent
from ..narrative.story import CharacterIndex
from ..parsing import fmt_ms
from ..schemas import Episode, Event
from . import scoring as cfg
from .schemas import Moment, Promo, Scores, Segment

# Mechanism words describe the supported label neutrally; the actual event text is always quoted next to
# them (D14). Earlier "physical_action" was rendered as "a physical clash" for every event - a raised
# hand, herb preparation and a sound effect alike.
PHRASES = {
    "physical_action": "a physical action", "shocking_line": "a shocking line", "betrayal": "a betrayal",
    "reveal": "a reveal", "threat": "a threat", "humiliation": "a humiliation",
    "confrontation": "a confrontation", "reversal": "a reversal", "striking_visual": "a striking image",
    "new_conflict": "a new conflict", "emotional_shift": "an emotional moment", "escalation": "an escalation",
    "open_question": "an open question", "resolution": "a resolution",
}
SOUND_TYPES = {"SOUND FX", "SOUND EFFECT", "SOUND EFFECTS"}


# ---- per-episode context ------------------------------------------------------------------

@dataclass
class EpisodeContext:
    episode: Episode
    evidence: dict[str, ConsensusEvent]
    index: CharacterIndex
    rank: int                                   # chronological position in the series
    by_id: dict[str, Event] = field(default_factory=dict)
    legal_starts: list[int] = field(default_factory=list)
    legal_ends: list[int] = field(default_factory=list)
    flagged_events: set[str] = field(default_factory=set)   # events named in parse diagnostics

    def __post_init__(self) -> None:
        events = self.episode.events
        self.by_id = {e.event_id: e for e in events}
        spans = [(e.start_ms, e.end_ms) for e in events if e.event_type == "DIALOGUE"]
        legal = lambda t: not any(s < t < e for s, e in spans)
        self.legal_starts = sorted({e.start_ms for e in events if legal(e.start_ms)})
        self.legal_ends = sorted({e.end_ms for e in events if legal(e.end_ms)})
        self.flagged_events = {e.event_id for e in events
                               if any(e.source_reference in d for d in self.episode.diagnostics)}

    def is_legal_cut(self, t: int) -> bool:
        return not any(e.event_type == "DIALOGUE" and e.start_ms < t < e.end_ms for e in self.episode.events)

    def best_tag(self, event_id: str, weights: dict[str, float]) -> tuple[float, str | None]:
        rec = self.evidence.get(event_id)
        if not rec:
            return 0.0, None
        scored = [(weights[t], t) for t in rec.tags if t in weights]
        return max(scored) if scored else (0.0, None)

    def start_for_hook(self, hook: Event) -> int | None:
        """Latest legal start at or before the hook event, at most HOOK_LEAD_MAX before it."""
        i = bisect.bisect_right(self.legal_starts, hook.start_ms) - 1
        if i >= 0 and hook.start_ms - self.legal_starts[i] <= cfg.HOOK_LEAD_MAX_MS:
            return self.legal_starts[i]
        return None

    def end_for_cliff(self, cliff: Event) -> int | None:
        """Earliest legal end at or after the cliffhanger event, at most CLIFF_TAIL_MAX after it."""
        i = bisect.bisect_left(self.legal_ends, cliff.end_ms)
        if i < len(self.legal_ends) and self.legal_ends[i] - cliff.end_ms <= cfg.CLIFF_TAIL_MAX_MS:
            return self.legal_ends[i]
        return None

    def events_in(self, start: int, end: int) -> list[Event]:
        return [e for e in self.episode.events if e.start_ms < end and e.end_ms > start]

    def resolves(self, event_id: str) -> bool:
        rec = self.evidence.get(event_id)
        return bool(rec and "resolution" in rec.tags)

    def anchors(self, weights: dict[str, float], minimum: float, exclude_resolutions: bool = False) -> list[tuple[float, Event]]:
        out = [(w, self.by_id[eid]) for eid in self.evidence
               if (w := self.best_tag(eid, weights)[0]) >= minimum and not (exclude_resolutions and self.resolves(eid))]
        return sorted(out, key=lambda x: (-x[0], x[1].start_ms))


# ---- candidates ---------------------------------------------------------------------------

@dataclass
class Candidate:
    kind: str
    segments: list[tuple[EpisodeContext, int, int]]
    hook: Event
    cliff: Event

    @property
    def duration_ms(self) -> int:
        return sum(e - s for _, s, e in self.segments)


def _coherent(hook_ctx: EpisodeContext, hook: Event, cliff_ctx: EpisodeContext, cliff: Event, *, require_thread: bool) -> bool:
    h, c = hook_ctx.evidence.get(hook.event_id), cliff_ctx.evidence.get(cliff.event_id)
    if not h or not c:
        return False
    shared_thread = {t for t in h.threads if not t.startswith("?")} & {t for t in c.threads if not t.startswith("?")}
    if require_thread:
        return bool(shared_thread)
    return bool(shared_thread) or bool(set(h.characters) & set(c.characters))


def _resolution_between(ctx: EpisodeContext, start: int, end: int) -> bool:
    return any("resolution" in rec.tags and start <= rec.start_ms < end for rec in ctx.evidence.values())


def _segment_ends(ctx: EpisodeContext, start: int, must_include_until: int) -> list[int]:
    """Up to three hook-segment end points: shortest, middle and longest within limits."""
    low, high = max(start + cfg.MIN_SEGMENT_MS, must_include_until), start + 25_000
    options = [t for t in ctx.legal_ends if low <= t <= high]
    return sorted({options[0], options[len(options) // 2], options[-1]}) if options else []


def _segment_starts(ctx: EpisodeContext, earliest: int, cliff: Event, end: int, max_len: int) -> list[int]:
    """Up to three cliffhanger-segment start points, preferring starts of annotated events."""
    latest = min(cliff.start_ms, end - cfg.MIN_SEGMENT_MS)
    options = [t for t in ctx.legal_starts if max(earliest, end - max_len) <= t <= latest]
    if not options:
        return []
    annotated = [t for t in options if any(r.start_ms == t for r in ctx.evidence.values())]
    picks = {options[0], options[-1]}
    if annotated:
        picks.add(annotated[len(annotated) // 2])
    return sorted(picks)


def generate(contexts: list[EpisodeContext], top_anchors: int = 8) -> list[Candidate]:
    candidates: list[Candidate] = []
    for ctx in contexts:
        hooks = ctx.anchors(cfg.HOOK_WEIGHTS, cfg.HOOK_MIN)
        cliffs = ctx.anchors(cfg.CLIFF_WEIGHTS, cfg.CLIFF_MIN, exclude_resolutions=True)
        for _, hook in hooks:
            start = ctx.start_for_hook(hook)
            if start is None:
                continue
            for _, cliff in cliffs:
                end = ctx.end_for_cliff(cliff)
                if end is None or cliff.start_ms <= hook.start_ms:
                    continue
                if cfg.MIN_DURATION_MS <= end - start <= cfg.MAX_DURATION_MS:
                    candidates.append(Candidate("single", [(ctx, start, end)], hook, cliff))
        # Two segments within the episode: only for hook->cliff spans that are too long, or that
        # contain a resolution which a continuous clip would carry into the middle of the promo.
        for _, hook in hooks[:top_anchors]:
            start = ctx.start_for_hook(hook)
            if start is None:
                continue
            for _, cliff in cliffs[:top_anchors]:
                end = ctx.end_for_cliff(cliff)
                if end is None or cliff.start_ms <= hook.start_ms:
                    continue
                if end - start <= cfg.MAX_DURATION_MS and not _resolution_between(ctx, hook.end_ms, cliff.start_ms):
                    continue
                if not _coherent(ctx, hook, ctx, cliff, require_thread=False):
                    continue
                for end1 in _segment_ends(ctx, start, hook.end_ms):
                    budget = cfg.MAX_DURATION_MS - (end1 - start)
                    for start2 in _segment_starts(ctx, end1 + cfg.MIN_SEGMENT_GAP_MS, cliff, end, budget):
                        if _resolution_between(ctx, start2, end) or (end1 - start) + (end - start2) < cfg.MIN_DURATION_MS:
                            continue
                        candidates.append(Candidate("two_segment", [(ctx, start, end1), (ctx, start2, end)], hook, cliff))
    # Cross-episode: hook segment in episode N, cliffhanger segment in N+1 or N+2 on a shared thread.
    by_rank = {c.rank: c for c in contexts}
    for ctx in contexts:
        hooks = ctx.anchors(cfg.HOOK_WEIGHTS, cfg.HOOK_MIN)[:top_anchors // 2]
        for distance in range(1, cfg.CROSS_EPISODE_MAX_DISTANCE + 1):
            other = by_rank.get(ctx.rank + distance)
            if other is None:
                continue
            for _, cliff in other.anchors(cfg.CLIFF_WEIGHTS, cfg.CLIFF_MIN, exclude_resolutions=True)[:top_anchors // 2]:
                end = other.end_for_cliff(cliff)
                if end is None:
                    continue
                for _, hook in hooks:
                    start = ctx.start_for_hook(hook)
                    if start is None or not _coherent(ctx, hook, other, cliff, require_thread=True):
                        continue
                    for end1 in _segment_ends(ctx, start, hook.end_ms):
                        budget = cfg.MAX_DURATION_MS - (end1 - start)
                        for start2 in _segment_starts(other, 0, cliff, end, budget):
                            if _resolution_between(other, start2, end) or (end1 - start) + (end - start2) < cfg.MIN_DURATION_MS:
                                continue
                            candidates.append(Candidate("cross_episode", [(ctx, start, end1), (other, start2, end)], hook, cliff))
    return candidates


# ---- evaluation: validity, quality, confidence, explanation -------------------------------

def _validity(c: Candidate) -> dict[str, bool]:
    first_ctx, start, _ = c.segments[0]
    last_ctx, _, end = c.segments[-1]
    checks = {
        "duration_20_60s": cfg.MIN_DURATION_MS <= c.duration_ms <= cfg.MAX_DURATION_MS,
        "segment_count": 1 <= len(c.segments) <= cfg.MAX_SEGMENTS,
        "segment_min_length": all(e - s >= cfg.MIN_SEGMENT_MS for _, s, e in c.segments),
        "no_cut_inside_dialogue": all(ctx.is_legal_cut(s) and ctx.is_legal_cut(e) for ctx, s, e in c.segments),
        "hook_in_first_3s": (start <= c.hook.start_ms <= start + cfg.HOOK_LEAD_MAX_MS
                             and first_ctx.best_tag(c.hook.event_id, cfg.HOOK_WEIGHTS)[0] >= cfg.HOOK_MIN),
        "cliffhanger_in_last_6s": (end - cfg.CLIFF_WINDOW_MS < c.cliff.end_ms <= end
                                   and end - c.cliff.end_ms <= cfg.CLIFF_TAIL_MAX_MS
                                   and last_ctx.best_tag(c.cliff.event_id, cfg.CLIFF_WEIGHTS)[0] >= cfg.CLIFF_MIN),
        # A resolution anywhere in the final window means the tension is answered before the cut, even if a
        # later reaction shot carries an open_question tag (observed: "I accept, Your Grace." then a bow).
        "ending_unresolved": not _resolution_between(last_ctx, min(c.cliff.start_ms, end - cfg.CLIFF_WINDOW_MS), end),
    }
    if len(c.segments) > 1:
        (ctx1, s1, e1), (ctx2, s2, e2) = c.segments
        same = ctx1 is ctx2
        checks["segments_in_story_order"] = (e1 + cfg.MIN_SEGMENT_GAP_MS <= s2) if same else ctx1.rank < ctx2.rank
        checks["segments_coherent"] = _coherent(ctx1, c.hook, ctx2, c.cliff, require_thread=not same)
    return checks


def _moment(ctx: EpisodeContext, event: Event, weights: dict[str, float], at_end: bool) -> Moment:
    rec = ctx.evidence.get(event.event_id)
    tags = sorted(rec.tags, key=lambda t: -weights.get(t, 0)) if rec else []
    speaker = ctx.index.name(ctx.index.resolve(event.speaker)) if event.speaker and ctx.index.resolve(event.speaker) else event.speaker
    question = ""
    if at_end and rec and rec.questions:
        question = next(iter(rec.questions.values()))
    return Moment(event_id=event.event_id, episode=ctx.episode.episode_number,
                  time=fmt_ms(event.end_ms if at_end else event.start_ms), event_type=event.event_type,
                  speaker=speaker, text=" ".join(event.content.split()), tags=tags,
                  supporting_models={t: rec.tags[t] for t in tags} if rec else {}, question=question)


def evaluate(c: Candidate) -> Promo:
    checks = _validity(c)
    first_ctx, start, _ = c.segments[0]
    last_ctx, _, end = c.segments[-1]
    seg_events = [(ctx, e) for ctx, s, en in c.segments for e in ctx.events_in(s, en)]
    annotated = [(ctx, ctx.evidence[e.event_id]) for ctx, e in seg_events if e.event_id in ctx.evidence]
    notes: list[str] = []

    # Hook strength: strongest hook tag among events starting in the opening window.
    opening = [e for e in first_ctx.events_in(start, start + cfg.HOOK_WINDOW_MS) if e.start_ms >= start]
    hook_scores = [(first_ctx.best_tag(e.event_id, cfg.HOOK_WEIGHTS), e) for e in opening]
    (hook_w, hook_tag), hook_event = max(hook_scores, key=lambda x: x[0][0], default=((0.0, None), c.hook))
    hook_rec_tags = {t for t in (first_ctx.evidence[hook_event.event_id].tags if hook_event.event_id in first_ctx.evidence
                                 else {}) if t in cfg.HOOK_WEIGHTS}
    hook_score = hook_w + min(cfg.EXTRA_HOOK_TAG_CAP, cfg.EXTRA_HOOK_TAG_BONUS * max(0, len(hook_rec_tags) - 1))
    if hook_event.start_ms - start > cfg.LATE_HOOK_MS:
        hook_score *= cfg.LATE_HOOK_FACTOR

    # Cliffhanger strength: strongest cliff tag among events ending in the final window.
    ending = [e for e in last_ctx.events_in(end - cfg.CLIFF_WINDOW_MS, end)
              if e.end_ms <= end and not last_ctx.resolves(e.event_id)]
    cliff_scores = [(last_ctx.best_tag(e.event_id, cfg.CLIFF_WEIGHTS), e.end_ms, e) for e in ending]
    (cliff_w, cliff_tag), _, cliff_event = max(cliff_scores, key=lambda x: (x[0][0], x[1]),
                                              default=((0.0, None), 0, c.cliff))
    cliff_score = cliff_w
    if any(e.event_type != "DIALOGUE" and any(cue in e.content.lower() for cue in cfg.WRITER_CUES) for e in ending):
        notes.append("stage directions call this ending a cliffhanger (production cue; not scored)")
    if end >= last_ctx.episode.duration_ms - cfg.CLIFF_TAIL_MAX_MS:
        notes.append("ends on the episode's closing beat (production cue; not scored)")
    if not checks["ending_unresolved"]:
        cliff_score *= cfg.RESOLUTION_AFTER_CLIFF_FACTOR

    # Progression and engagement: distinct annotated events carrying those tags.
    progression = cfg.saturate(sum(1 for _, r in annotated if cfg.DEVELOPMENT_TAGS & set(r.tags)))
    engagement = cfg.saturate(sum(1 for _, r in annotated if cfg.ENGAGEMENT_TAGS & set(r.tags)))

    # Standalone comprehension proxy.
    levels = [r.prior_context_level for _, r in annotated if r.prior_context_level]
    dependency = sum(cfg.PRIOR_CONTEXT_WEIGHT[l] for l in levels) / len(levels) if levels else 0.0
    present: set[str] = set()
    seen: set[str] = set()
    for ctx, e in seg_events:
        speaker = ctx.index.resolve(e.speaker) if e.speaker else None
        mentioned = ctx.index.mentions(e.content)
        present |= mentioned | ({speaker} if speaker else set())
        if speaker:
            seen.add(speaker)
        if e.event_type != "DIALOGUE":
            seen |= mentioned
    unseen = present - seen
    standalone = 1.0 - cfg.CONTEXT_PENALTY_SCALE * dependency
    standalone -= cfg.CROWD_PENALTY * max(0, len(present) - cfg.CROWD_FREE_CHARACTERS)
    standalone -= min(cfg.UNSEEN_REFERENCE_CAP, cfg.UNSEEN_REFERENCE_PENALTY * len(unseen))
    standalone -= cfg.EXTRA_SEGMENT_PENALTY * (len(c.segments) - 1)
    if c.kind == "cross_episode":
        standalone -= cfg.CROSS_EPISODE_PENALTY
    if dependency >= 0.5:
        notes.append("leans on earlier episodes for context")
    if unseen:
        notes.append("mentions " + ", ".join(sorted(first_ctx.index.name(u) for u in unseen)) + " without showing them")

    components = {"hook": round(min(1.0, hook_score), 3), "cliffhanger": round(min(1.0, cliff_score), 3),
                  "hook_label": round(min(1.0, hook_score), 3), "cliffhanger_label": round(min(1.0, cliff_score), 3),
                  "standalone": round(max(0.0, min(1.0, standalone)), 3),
                  "progression": progression, "engagement": engagement}
    quality = cfg.quality_total(components)

    # Confidence: how reliable the hook and cliffhanger evidence is (never feeds quality).
    # Agreement = how many models saw *a* hook in the opening window and *an* unresolved moment in
    # the final window (any tag of that class, on any event there). Models often tag neighbouring
    # events of one beat (the line vs the reaction shot), so per-event agreement understates it.
    conf_notes: list[str] = []
    hook_rec, cliff_rec = first_ctx.evidence.get(hook_event.event_id), last_ctx.evidence.get(cliff_event.event_id)

    def class_models(ctx: EpisodeContext, events: list[Event], weights: dict[str, float], minimum: float) -> set[str]:
        return {m for e in events if (r := ctx.evidence.get(e.event_id))
                for t, ms in r.tags.items() if weights.get(t, 0) >= minimum for m in ms}

    hook_models = class_models(first_ctx, opening, cfg.HOOK_WEIGHTS, cfg.HOOK_MIN)
    cliff_models = class_models(last_ctx, ending, cfg.CLIFF_WEIGHTS, cfg.CLIFF_MIN)
    hook_agree = cfg.AGREEMENT_SCORE.get(min(2, len(hook_models)), 0.0)
    cliff_agree = cfg.AGREEMENT_SCORE.get(min(2, len(cliff_models)), 0.0)
    if hook_agree < 1:
        conf_notes.append("only one model marked the opening as a hook")
    if cliff_agree < 1:
        conf_notes.append("only one model marked the ending as unresolved")
    confidence = 0.5 * hook_agree + 0.5 * cliff_agree
    for rec in (hook_rec, cliff_rec):
        if rec and rec.repaired:
            confidence -= cfg.REPAIR_PENALTY
            conf_notes.append(f"{rec.short_id} evidence ID was repaired from a neighbouring event")
        if rec and len(set(rec.prior_context.values())) > 1:
            confidence -= cfg.CONTEXT_DISAGREEMENT_PENALTY
            conf_notes.append(f"models disagree on how much context {rec.short_id} needs")
    if any(e.event_id in ctx.flagged_events for ctx, e in seg_events):
        confidence -= cfg.DIAGNOSTIC_PENALTY
        conf_notes.append("an included event has a parser diagnostic")
    if c.kind == "cross_episode":
        confidence -= cfg.WEAK_THREAD_LINK_PENALTY
        conf_notes.append("joins episodes through a heuristic storyline link")
    confidence = round(max(0.0, min(1.0, confidence)), 2)

    hook_m = _moment(first_ctx, hook_event, cfg.HOOK_WEIGHTS, at_end=False)
    cliff_m = _moment(last_ctx, cliff_event, cfg.CLIFF_WEIGHTS, at_end=True)
    why = (f"Opens{_lead(hook_event.start_ms - start)} on {_phrase(first_ctx, hook_event, hook_tag)} "
           f"(\u201c{_excerpt(hook_event)}\u201d) and cuts on {_phrase(last_ctx, cliff_event, cliff_tag)} "
           f"(\u201c{_excerpt(cliff_event)}\u201d)"
           + (f" - open question: {cliff_m.question.rstrip('?')}?" if cliff_m.question and cliff_tag == "open_question" else "")
           + ".")

    segments = [Segment(episode=ctx.episode.episode_number, episode_id=ctx.episode.episode_id, start_ms=s, end_ms=e,
                        start=fmt_ms(s), end=fmt_ms(e)) for ctx, s, e in c.segments]
    pid = hashlib.sha1("|".join(f"{s.episode_id}:{s.start_ms}-{s.end_ms}" for s in segments).encode()).hexdigest()[:12]
    threads = sorted({t for _, r in annotated for t in r.threads if not t.startswith("?")})
    return Promo(
        promo_id=pid, kind=c.kind, segments=segments, duration_seconds=round(c.duration_ms / 1000, 2),
        hook=hook_m, cliffhanger=cliff_m, why=why, notes=notes,
        scores=Scores(quality=quality, components=components, confidence=confidence,
                      confidence_label=cfg.confidence_label(confidence), confidence_notes=conf_notes,
                      valid=all(checks.values()), validity_checks=checks),
        characters=sorted(first_ctx.index.name(p) for p in present), threads=threads,
        evidence_event_ids=[r.event_id for _, r in annotated])


def _lead(ms: int) -> str:
    return f" {ms / 1000:.1f}s in" if ms >= 500 else ""


def _excerpt(event: Event, words: int = 12) -> str:
    text = " ".join(event.content.replace('"', "").split())
    parts = text.split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


def _phrase(ctx: EpisodeContext, event: Event, tag: str | None) -> str:
    """'Clarissa's shocking line' for dialogue; 'a sound cue' for sound events; 'a reveal (Gerald)' otherwise."""
    if event.event_type in SOUND_TYPES:
        return "a sound cue"
    phrase = PHRASES.get(tag, "an unresolved moment")
    speaker = ctx.index.resolve(event.speaker) if event.speaker else None
    if speaker:
        return f"{ctx.index.name(speaker)}'s {phrase.split(' ', 1)[1]}"
    rec = ctx.evidence.get(event.event_id)
    names = [ctx.index.name(cid) for cid in (rec.characters if rec else [])][:2]
    return f"{phrase} ({', '.join(names)})" if names else phrase


def summarize_rejections(promos: list[Promo]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for p in promos:
        for check, ok in p.scores.validity_checks.items():
            if not ok:
                counts[check] += 1
    return dict(counts)
