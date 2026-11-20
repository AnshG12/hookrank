"""Merge independent annotators' labels into per-event evidence with support counts.

Support is a *confidence* signal (docs/decisions.md D5), not a quality score. Two models
often label adjacent events that describe one on-screen moment (a dialogue line and the
close-up showing it), so a tag counts as supported by another model when that model put the
same tag on the same event or on a co-referent one: overlapping in time or at most
``COREF_GAP_MS`` apart.
"""
from __future__ import annotations

from ..schemas import Episode, Event
from .schemas import ConsensusEvent, EpisodeAnnotations

COREF_GAP_MS = 1000


def coreferent(a: Event, b: Event, gap_ms: int = COREF_GAP_MS) -> bool:
    return a.start_ms <= b.end_ms + gap_ms and b.start_ms <= a.end_ms + gap_ms


def build_consensus(episode: Episode, annotation_sets: list[EpisodeAnnotations]) -> list[ConsensusEvent]:
    events = {e.event_id: e for e in episode.events}
    records: dict[str, ConsensusEvent] = {}
    for annotations in annotation_sets:
        model = annotations.model
        for a in annotations.annotations:
            event = events[a.event_id]
            rec = records.get(a.event_id)
            if rec is None:
                rec = records[a.event_id] = ConsensusEvent(
                    event_id=a.event_id, short_id=a.short_id, episode_id=episode.episode_id,
                    start_ms=event.start_ms, end_ms=event.end_ms, event_type=event.event_type)
            if model not in rec.direct_models:
                rec.direct_models.append(model)
            for tag in a.tags:
                supporters = rec.tags.setdefault(tag, [])
                if model not in supporters:
                    supporters.append(model)
            if a.question:
                rec.questions[model] = a.question
            rec.quotes[model] = a.quote
            if a.prior_context:
                rec.prior_context[model] = a.prior_context
            if a.repaired_from:
                rec.repaired[model] = a.repair_distance
            if a.thread and a.thread not in rec.threads:
                rec.threads.append(a.thread)  # episode-local key; mapped to series IDs later

    # Cross-model support through co-referent events.
    by_model = {s.model: s.annotations for s in annotation_sets}
    for rec in records.values():
        event = events[rec.event_id]
        for tag, supporters in rec.tags.items():
            for model, annotations in by_model.items():
                if model in supporters:
                    continue
                if any(tag in a.tags and coreferent(event, events[a.event_id]) for a in annotations):
                    supporters.append(model)
    return sorted(records.values(), key=lambda r: (r.start_ms, r.end_ms))
