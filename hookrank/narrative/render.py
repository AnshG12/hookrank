"""Compact, ID-addressed rendering of an episode for model prompts.

Events get short sequential IDs (E001...) because long hash IDs invite copying errors.
Timestamps are deliberately omitted: models never decide timing (Python does), and dropping
them saves ~10% of prompt tokens, which matters under an 8k-token request ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Episode, Event


@dataclass
class RenderedEpisode:
    episode: Episode
    short_ids: list[str] = field(default_factory=list)
    by_short: dict[str, Event] = field(default_factory=dict)
    short_of: dict[str, str] = field(default_factory=dict)  # event_id -> short id
    lines: list[str] = field(default_factory=list)           # one rendered line per event
    scene_markers: dict[int, str] = field(default_factory=dict)  # event index -> scene marker line

    @property
    def speaker_labels(self) -> list[str]:
        return sorted({e.speaker for e in self.episode.events if e.speaker})

    def header(self) -> str:
        ep = self.episode
        parts = [f"EPISODE: {ep.episode_number}"]
        if ep.logline:
            parts.append(f"LOGLINE: {ep.logline}")
        if ep.continuity_notes:
            parts.append(f"CONTINUITY NOTES: {' '.join(ep.continuity_notes.split())}")
        parts.append("SPEAKER LABELS: " + (", ".join(self.speaker_labels) or "(none)"))
        return "\n".join(parts)

    def body(self, start: int = 0, end: int | None = None, skip_camera: bool = False) -> str:
        """Event lines in [start, end). ``skip_camera`` is a degraded mode for oversized
        episodes in story extraction only (camera lines carry little who-is-who information)."""
        end = len(self.lines) if end is None else end
        out: list[str] = []
        for index in range(start, end):
            marker = self.scene_markers.get(index)
            if marker or (index == start and start > 0):
                out.append(marker or self._scene_marker_before(index))
            if skip_camera and self.by_short[self.short_ids[index]].event_type == "CAMERA":
                continue
            out.append(self.lines[index])
        return "\n".join(out)

    def _scene_marker_before(self, index: int) -> str:
        for i in range(index, -1, -1):
            if i in self.scene_markers:
                return self.scene_markers[i] + " (continued)"
        return "-- (continued) --"

    def chunks(self, max_lines: int, overlap: int = 4) -> list[tuple[int, int]]:
        """Contiguous event windows for episodes too large for one request."""
        n = len(self.lines)
        if n <= max_lines:
            return [(0, n)]
        windows, start = [], 0
        while start < n:
            end = min(n, start + max_lines)
            windows.append((start, end))
            if end == n:
                break
            start = end - overlap
        return windows


def _event_line(short: str, event: Event) -> str:
    content = " ".join(event.content.split())
    if event.event_type == "DIALOGUE":
        label = event.speaker or "UNKNOWN SPEAKER"
        if event.delivery:
            label += f" ({event.delivery})"
        return f"{short} {label}: {content}"
    return f"{short} {event.event_type}: {content}"


def render_episode(episode: Episode) -> RenderedEpisode:
    rendered = RenderedEpisode(episode=episode)
    headings = {s.scene_id: s.heading for s in episode.scenes}
    last_scene = None
    for index, event in enumerate(episode.events):
        short = f"E{index + 1:03d}"
        rendered.short_ids.append(short)
        rendered.by_short[short] = event
        rendered.short_of[event.event_id] = short
        if event.scene_id != last_scene:
            heading = headings.get(event.scene_id, "")
            label = event.scene_id.replace("scene-", "SCENE ").upper()
            rendered.scene_markers[index] = f"-- {label}{': ' + heading if heading else ''} --"
            last_scene = event.scene_id
        rendered.lines.append(_event_line(short, event))
    return rendered
