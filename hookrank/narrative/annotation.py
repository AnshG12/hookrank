"""Per-episode event annotation with evidence verification and quote-anchored ID repair.

Every annotation must carry a quote from the event it cites. S1 found gpt-oss cites the
right words under a neighbouring ID ~8% of the time; since timestamps come from IDs, an
unrepaired error would silently move a cut point. Repair policy (docs/decisions.md S1):
  1. quote matches the cited event                      -> accept
  2. quote matches exactly one event within +/-2        -> move ID, record repair
  3. quote matches exactly one event anywhere (>=4 words) -> move ID, record repair
  4. otherwise                                           -> drop, record issue
"""
from __future__ import annotations

from ..prompts import render
from ..providers import ProviderError, extract_json
from .render import RenderedEpisode
from .schemas import EpisodeAnnotations, EpisodeStory, EventAnnotation, Issue
from .text import quote_match, tokens
from .vocab import PRIOR_CONTEXT_HELP, PRIOR_CONTEXT_LEVELS, TAG_EVENT_TYPES, TAGS, render_tag_definitions

STAGE = "annotation"
NEIGHBOUR_WINDOW = 2
GLOBAL_REPAIR_MIN_WORDS = 4


def build_annotation_prompt(rendered: RenderedEpisode, context: str, start: int = 0, end: int | None = None) -> str:
    episode_block = rendered.header() + "\n\nEVENTS:\n" + rendered.body(start, end)
    return render("narrative_annotate.md", tag_definitions=render_tag_definitions(), story_context=context,
                  prior_context_help=PRIOR_CONTEXT_HELP, episode=episode_block)


def _locate(quote: str, cited: str | None, rendered: RenderedEpisode) -> tuple[str | None, str | None, int]:
    """Return (short_id, match_kind, distance) for the event the quote really belongs to."""
    order = rendered.short_ids
    if cited in rendered.by_short:
        kind = quote_match(quote, rendered.by_short[cited].content)
        if kind:
            return cited, kind, 0
        index = order.index(cited)
        near = [(abs(i - index), order[i]) for i in range(max(0, index - NEIGHBOUR_WINDOW),
                                                          min(len(order), index + NEIGHBOUR_WINDOW + 1)) if i != index]
        hits = sorted((d, s, k) for d, s in near if (k := quote_match(quote, rendered.by_short[s].content)))
        if hits and (len(hits) == 1 or hits[0][0] < hits[1][0]):  # nearest match, unambiguous
            distance, short, kind = hits[0]
            return short, kind, distance
    if len(tokens(quote)) >= GLOBAL_REPAIR_MIN_WORDS:
        hits = [(s, quote_match(quote, rendered.by_short[s].content)) for s in order]
        hits = [h for h in hits if h[1]]
        if len(hits) == 1:
            short, kind = hits[0]
            distance = abs(order.index(short) - order.index(cited)) if cited in rendered.by_short else -1
            return short, kind, distance
    return None, None, 0


def parse_annotations(raw: str, rendered: RenderedEpisode, model: str, story: EpisodeStory | None,
                      result: EpisodeAnnotations) -> None:
    """Verify one raw annotation response and merge it into ``result`` (supports chunked calls)."""
    issues = result.issues
    try:
        data = extract_json(raw)
    except ProviderError as exc:
        issues.append(Issue(stage=STAGE, model=model, kind="unparseable", detail=str(exc)))
        result.ok = False
        return
    items = data.get("events")
    if not isinstance(items, list):
        issues.append(Issue(stage=STAGE, model=model, kind="missing_events", detail="no 'events' list"))
        result.ok = False
        return
    thread_keys = {t.key for t in story.threads} if story and story.ok else set()
    existing = {a.event_id: a for a in result.annotations}

    for item in items:
        if not isinstance(item, dict):
            continue
        cited = str(item.get("id") or "").strip().upper() or None
        quote = str(item.get("quote") or "").strip()
        raw_tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        tags = []
        for tag in raw_tags:
            tag = str(tag).strip().lower()
            if tag in TAGS and tag not in tags:
                tags.append(tag)
            elif tag:
                issues.append(Issue(stage=STAGE, model=model, kind="unknown_tag", detail=f"{cited}: {tag}"))
        if not tags:
            continue
        if not quote:
            issues.append(Issue(stage=STAGE, model=model, kind="missing_quote", detail=f"{cited}: dropped"))
            continue
        short, kind, distance = _locate(quote, cited, rendered)
        if short is None:
            issues.append(Issue(stage=STAGE, model=model, kind="quote_not_found",
                                detail=f"{cited}: {quote[:80]!r} dropped"))
            continue
        if short != cited:
            issues.append(Issue(stage=STAGE, model=model, kind="id_repaired",
                                detail=f"{cited} -> {short} (distance {distance})"))
        event = rendered.by_short[short]
        # Tag applicability is checked against the event the evidence actually belongs to (after any repair),
        # so a strong tag is never carried onto an event type it cannot describe.
        inapplicable = [t for t in tags if t in TAG_EVENT_TYPES and event.event_type not in TAG_EVENT_TYPES[t]]
        if inapplicable:
            issues.append(Issue(stage=STAGE, model=model, kind="tag_not_applicable",
                                detail=f"{short} [{event.event_type}]: dropped {inapplicable}"))
            tags = [t for t in tags if t not in inapplicable]
            if not tags:
                continue
        question = str(item.get("question") or "").strip()[:200]
        prior = str(item.get("prior_context") or "").strip().lower()
        thread = str(item.get("thread") or "").strip().upper()
        if event.event_id in existing:  # duplicate (e.g. chunk overlap): merge tags, keep first evidence
            merged = existing[event.event_id]
            merged.tags += [t for t in tags if t not in merged.tags]
            merged.question = merged.question or question
            continue
        annotation = EventAnnotation(
            event_id=event.event_id, short_id=short, model=model, quote=quote, quote_match=kind, tags=tags,
            question=question, thread=thread if thread in thread_keys else "",
            prior_context=prior if prior in PRIOR_CONTEXT_LEVELS else None,
            repaired_from=cited if short != cited else None, repair_distance=distance)
        result.annotations.append(annotation)
        existing[event.event_id] = annotation
