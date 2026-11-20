"""Verified narrative records. Everything here has passed checks against the screenplay;
anything rejected or repaired along the way is kept as an ``Issue`` for audit."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..calls import CallRecord  # noqa: F401  (re-exported; defined with the shared caller)


class Issue(BaseModel):
    stage: str          # "story" | "annotation" | "series"
    model: str
    kind: str           # e.g. "invalid_id", "id_repaired", "unverified_name", "unknown_tag"
    detail: str


# ---- per-episode story extraction (one model) -------------------------------------------

class CharacterRecord(BaseModel):
    label: str                    # speaker label as written, or the name for non-speakers
    name: str                     # personal name if verified in this episode's text, else the label
    name_verified: bool
    speaking: bool
    description: str = ""
    evidence: list[str] = Field(default_factory=list)   # event_ids


class RelationshipRecord(BaseModel):
    a: str
    b: str
    relation: str
    change: str = ""
    evidence: list[str] = Field(default_factory=list)


class ThreadRecord(BaseModel):
    key: str                      # episode-local key, e.g. "T1"
    summary: str
    kind: str
    participants: list[str] = Field(default_factory=list)
    status: str
    evidence: list[str] = Field(default_factory=list)


class RevelationRecord(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list)


class EpisodeStory(BaseModel):
    episode_id: str
    episode_number: str
    model: str
    ok: bool = True
    summary: str = ""
    characters: list[CharacterRecord] = Field(default_factory=list)
    relationships: list[RelationshipRecord] = Field(default_factory=list)
    threads: list[ThreadRecord] = Field(default_factory=list)
    revelations: list[RevelationRecord] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


# ---- per-episode event annotation (each annotator model) --------------------------------

class EventAnnotation(BaseModel):
    event_id: str
    short_id: str
    model: str
    quote: str
    quote_match: str               # "exact" | "elided"
    tags: list[str]
    question: str = ""
    thread: str = ""               # episode-local thread key
    prior_context: str | None = None
    repaired_from: str | None = None   # short ID the model originally cited, if repaired
    repair_distance: int = 0


class EpisodeAnnotations(BaseModel):
    episode_id: str
    model: str
    ok: bool = True
    chunks: int = 1
    annotations: list[EventAnnotation] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


# ---- cross-model consensus ---------------------------------------------------------------

class ConsensusEvent(BaseModel):
    event_id: str
    short_id: str
    episode_id: str
    start_ms: int
    end_ms: int
    event_type: str
    # tag -> models supporting it on this event or a co-referent event (overlapping / <=1s apart)
    tags: dict[str, list[str]] = Field(default_factory=dict)
    direct_models: list[str] = Field(default_factory=list)   # models that annotated this exact event
    questions: dict[str, str] = Field(default_factory=dict)
    quotes: dict[str, str] = Field(default_factory=dict)
    # character_ids present: the speaker plus verified names mentioned in the event text,
    # computed deterministically in Python (never taken from a model claim)
    characters: list[str] = Field(default_factory=list)
    threads: list[str] = Field(default_factory=list)         # series thread IDs after linking
    prior_context: dict[str, str] = Field(default_factory=dict)
    repaired: dict[str, int] = Field(default_factory=dict)   # model -> repair distance

    def support(self, tag: str) -> int:
        return len(self.tags.get(tag, []))

    @property
    def prior_context_level(self) -> str | None:
        """Most conservative (highest) dependency any model reported."""
        order = {"none": 0, "some": 1, "heavy": 2}
        values = [v for v in self.prior_context.values() if v in order]
        return max(values, key=order.__getitem__) if values else None
