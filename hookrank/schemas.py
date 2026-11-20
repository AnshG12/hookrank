"""Parsed screenplay records: scenes, timed events and episodes."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    scene_id: str = "scene-unknown"
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    event_type: str
    speaker: str = ""
    delivery: str = ""
    content: str
    source_reference: str
    source_order: int = Field(ge=0)

    @field_validator("end_ms")
    @classmethod
    def valid_end(cls, value: int, info):
        start = info.data.get("start_ms")
        if start is not None and value < start:
            raise ValueError("event end must be at or after event start")
        return value


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_id: str
    heading: str = ""
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)

    @field_validator("end_ms")
    @classmethod
    def valid_end(cls, value: int | None, info):
        start = info.data.get("start_ms")
        if value is not None and start is not None and value < start:
            raise ValueError("scene end must be at or after scene start")
        return value


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series_id: str = "default"
    episode_id: str
    episode_number: str
    title: str
    logline: str = ""
    continuity_notes: str = ""
    duration_ms: int = Field(default=0, ge=0)
    scenes: list[Scene] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    source_name: str
    source_hash: str
    diagnostics: list[str] = Field(default_factory=list)
