"""Promo output records: every field traces back to screenplay events."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    episode: str            # episode number as written in the script
    episode_id: str
    start_ms: int
    end_ms: int
    start: str              # MM:SS.cc, script time (R1: may differ from the final edited video)
    end: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class Moment(BaseModel):
    """The hook or cliffhanger moment: one screenplay event plus its evidence."""
    event_id: str
    episode: str
    time: str               # event start (hook) or end (cliffhanger), script time
    event_type: str
    speaker: str = ""
    text: str               # verbatim screenplay text of the event
    tags: list[str]
    supporting_models: dict[str, list[str]] = Field(default_factory=dict)   # tag -> models
    question: str = ""      # model-written unanswered question (cliffhanger only); interpretation, not script text


class Scores(BaseModel):
    quality: float                       # 0-100 quality estimate (narrative evidence)
    components: dict[str, float]         # hook, cliffhanger, standalone, progression, engagement (0-1)
    confidence: float                    # 0-1 reliability of the evidence
    confidence_label: str
    confidence_notes: list[str] = Field(default_factory=list)
    valid: bool
    validity_checks: dict[str, bool] = Field(default_factory=dict)
    label_quality: float | None = None      # quality from labels alone, before the clip check scaled components
    variety_adjusted: float | None = None   # ranking value after the variety adjustment (raw quality unchanged)


class Promo(BaseModel):
    promo_id: str
    rank: int | None = None
    kind: str                            # "single" | "two_segment" | "cross_episode"
    segments: list[Segment]
    duration_seconds: float
    hook: Moment
    cliffhanger: Moment
    why: str                             # one line on why it works
    rank_reason: str = ""
    scores: Scores
    characters: list[str] = Field(default_factory=list)   # canonical names present
    threads: list[str] = Field(default_factory=list)      # series thread IDs touched
    evidence_event_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)        # standalone/structure observations
    clip_check: dict | None = None                        # D13 clip-level check (top candidates only)
