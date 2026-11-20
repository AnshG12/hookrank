"""Per-episode story extraction: characters, relationships, threads, revelations.

Each episode is extracted independently (no prior-episode context), which keeps calls
parallel, cacheable per episode, and invalidated only when that episode changes. The
series-level story model is assembled afterwards in Python (``story.py``).

Verification is item-by-item: a bad item is dropped and logged, the rest is kept.
"""
from __future__ import annotations

from ..prompts import render
from ..providers import ProviderError, extract_json
from .render import RenderedEpisode
from .schemas import (CharacterRecord, EpisodeStory, Issue, RelationshipRecord, RevelationRecord,
                      ThreadRecord)
from .text import name_in_text, normalize
from .vocab import THREAD_KINDS, THREAD_STATUSES

STAGE = "story"
MAX_THREADS = 4


def build_story_prompt(rendered: RenderedEpisode, skip_camera: bool = False) -> str:
    episode_block = rendered.header() + "\n\nEVENTS:\n" + rendered.body(skip_camera=skip_camera)
    return render("narrative_story.md", thread_kinds=", ".join(THREAD_KINDS),
                  thread_statuses=", ".join(THREAD_STATUSES), episode=episode_block)


def episode_text(rendered: RenderedEpisode) -> str:
    """Everything the screenplay itself says about this episode (used to verify names)."""
    ep = rendered.episode
    return " ".join([ep.logline, ep.continuity_notes, *rendered.speaker_labels, *(e.content for e in ep.events)])


def _strings(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value or [] if isinstance(v, (str, int, float)) and str(v).strip()]


def _evidence(value, rendered: RenderedEpisode, issues: list[Issue], model: str, where: str) -> list[str]:
    ids: list[str] = []
    for short in _strings(value):
        short = short.upper()
        event = rendered.by_short.get(short)
        if event is None:
            issues.append(Issue(stage=STAGE, model=model, kind="invalid_id", detail=f"{where}: {short}"))
        elif event.event_id not in ids:
            ids.append(event.event_id)
    return ids


def parse_story(raw: str, rendered: RenderedEpisode, model: str) -> EpisodeStory:
    ep = rendered.episode
    story = EpisodeStory(episode_id=ep.episode_id, episode_number=ep.episode_number, model=model)
    issues = story.issues
    try:
        data = extract_json(raw)
    except ProviderError as exc:
        story.ok = False
        issues.append(Issue(stage=STAGE, model=model, kind="unparseable", detail=str(exc)))
        return story

    story.summary = str(data.get("summary") or "").strip()
    full_text = episode_text(rendered)
    labels = {label.casefold(): label for label in rendered.speaker_labels}

    seen_labels: set[str] = set()
    for item in data.get("characters") or []:
        if not isinstance(item, dict):
            continue
        raw_label = str(item.get("label") or "").strip()
        speaking = raw_label.casefold() in labels
        if not raw_label or (not speaking and not name_in_text(raw_label, full_text)):
            issues.append(Issue(stage=STAGE, model=model, kind="unknown_character", detail=raw_label or "(empty label)"))
            continue
        label = labels.get(raw_label.casefold(), raw_label)
        if label.casefold() in seen_labels:
            continue
        seen_labels.add(label.casefold())
        name = str(item.get("name") or "").strip() or label
        verified = normalize(name) != normalize(label) and name_in_text(name, full_text)
        if normalize(name) != normalize(label) and not verified:
            issues.append(Issue(stage=STAGE, model=model, kind="unverified_name",
                                detail=f"{label} -> {name!r} not stated in this episode; kept label"))
            name = label
        story.characters.append(CharacterRecord(
            label=label, name=name, name_verified=verified, speaking=speaking,
            description=str(item.get("description") or "").strip()[:120],
            evidence=_evidence(item.get("evidence"), rendered, issues, model, f"character {label}")))
    for label in rendered.speaker_labels:  # every speaker must exist, even if the model skipped it
        if label.casefold() not in seen_labels:
            story.characters.append(CharacterRecord(label=label, name=label, name_verified=False, speaking=True))
            issues.append(Issue(stage=STAGE, model=model, kind="missing_speaker", detail=label))

    for item in data.get("relationships") or []:
        if not isinstance(item, dict):
            continue
        a, b, relation = (str(item.get(k) or "").strip() for k in ("a", "b", "relation"))
        if not (a and b and relation) or normalize(a) == normalize(b):
            issues.append(Issue(stage=STAGE, model=model, kind="invalid_relationship", detail=str(item)[:160]))
            continue
        story.relationships.append(RelationshipRecord(
            a=a, b=b, relation=relation[:80], change=str(item.get("change") or "").strip()[:120],
            evidence=_evidence(item.get("evidence"), rendered, issues, model, f"relationship {a}/{b}")))

    keys: set[str] = set()
    for item in (data.get("threads") or [])[:MAX_THREADS]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip().upper() or f"T{len(story.threads) + 1}"
        summary = str(item.get("summary") or "").strip()
        participants = _strings(item.get("participants"))
        if not summary or key in keys:
            issues.append(Issue(stage=STAGE, model=model, kind="invalid_thread", detail=str(item)[:160]))
            continue
        keys.add(key)
        kind = str(item.get("kind") or "").strip().lower()
        status = str(item.get("status") or "").strip().lower()
        if kind not in THREAD_KINDS:
            issues.append(Issue(stage=STAGE, model=model, kind="unknown_thread_kind", detail=f"{key}: {kind!r} -> other"))
            kind = "other"
        if status not in THREAD_STATUSES:
            status = "ongoing"
        story.threads.append(ThreadRecord(
            key=key, summary=summary[:160], kind=kind, participants=participants, status=status,
            evidence=_evidence(item.get("evidence"), rendered, issues, model, f"thread {key}")))

    for item in (data.get("revelations") or [])[:5]:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            story.revelations.append(RevelationRecord(
                text=str(item["text"]).strip()[:160],
                evidence=_evidence(item.get("evidence"), rendered, issues, model, "revelation")))
    story.open_questions = [q[:160] for q in _strings(data.get("open_questions"))[:3]]
    return story


def story_context(story: EpisodeStory | None) -> str:
    """Compact story context handed to annotators (names and thread keys only)."""
    if story is None or not story.ok:
        return "(not available - use the speaker labels as names)"
    lines = ["Characters (speaker label = name: description):"]
    for c in story.characters:
        name = c.name if c.name != c.label else c.label.title()
        lines.append(f"- {c.label} = {name}" + (f": {c.description}" if c.description else ""))
    if story.threads:
        lines.append("Threads:")
        lines += [f"- {t.key} [{t.kind}] {t.summary} (participants: {', '.join(t.participants) or 'unspecified'})"
                  for t in story.threads]
    return "\n".join(lines)
