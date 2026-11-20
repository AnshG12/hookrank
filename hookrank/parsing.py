"""Screenplay ingestion: parse the studio's timestamped screenplay template into typed events.

Supported template (see README "Parser assumptions"): EPISODE NUMBER / EPISODE LOGLINE /
CONTINUITY NOTES headers, SCENE blocks with heading/start/end metadata, `**LABEL (delivery):**`
event labels and `Start: (MM:SS.cc) | End: (MM:SS.cc)` timing lines. Anything unrecognised is
reported as a diagnostic rather than silently dropped.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

from pydantic import ValidationError

from .schemas import Episode, Event, Scene

RANGE_RE = re.compile(r"Start\s*:\s*\((\d{1,2}:\d{2}(?:\.\d{1,3})?)\)\s*\|\s*End\s*:\s*\((\d{1,2}:\d{2}(?:\.\d{1,3})?)\)", re.I)
EVENT_LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
SCENE_RE = re.compile(r"^\s*SCENE\s+(\d+)\s*$", re.I)
SCENE_META_RE = re.compile(r"^\s*SCENE\s+(HEADING|START|END)\s*:\s*(.*?)\s*$", re.I)
SCENE_TIME_RE = re.compile(r"^(\d{1,2}:\d{2}(?:\.\d{1,3})?)$")


def to_ms(value: str) -> int:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?", value.strip())
    if not m:
        raise ValueError(f"Unsupported timestamp: {value}")
    if int(m.group(2)) >= 60:
        raise ValueError(f"Invalid seconds component in timestamp: {value}")
    frac = (m.group(3) or "").ljust(3, "0")
    return (int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(frac or 0)


def fmt_ms(ms: int) -> str:
    centiseconds = ms % 10 == 0
    precision = 2 if centiseconds else 3
    fraction = (ms % 1000) // 10 if centiseconds else ms % 1000
    return f"{ms // 60000:02}:{(ms % 60000) // 1000:02}.{fraction:0{precision}d}"


def _event_label(raw: str) -> tuple[str, str, str]:
    label = re.sub(r"\s+", " ", raw.strip())
    if label.upper() in {"CAMERA", "ACTION", "SOUND FX", "SOUND EFFECT", "SOUND EFFECTS"}:
        return label.upper(), "", ""
    match = re.match(r"^(.*?)\s*\((.*?)\)\s*$", label)
    if match:
        return "DIALOGUE", match.group(1).strip(), match.group(2).strip()
    return "DIALOGUE", label, ""


def parse_episode(raw: str, source_name: str, series_id: str = "default") -> Episode:
    lines = raw.splitlines()
    episode_number = ""
    logline = ""
    notes: list[str] = []
    diagnostics: list[str] = []
    notes_started = False
    scenes: list[Scene] = []
    scene_counter = 0
    scene_id = "scene-unknown"
    current_scene: dict = {"scene_id": scene_id, "heading": "", "start_ms": None, "end_ms": None}

    def save_scene() -> None:
        try:
            scenes.append(Scene(**current_scene))
        except ValidationError as exc:
            diagnostics.append(f"Invalid {current_scene['scene_id']} metadata: {exc.errors(include_url=False)}")
            current_scene["end_ms"] = None
            scenes.append(Scene(**current_scene))

    for line in lines:
        match = re.match(r"\s*EPISODE\s+NUMBER\s*:?\s*(.*)", line, re.I)
        if match:
            episode_number = match.group(1).strip()
        match = re.match(r"\s*EPISODE\s+LOGLINE\s*:?\s*(.*)", line, re.I)
        if match:
            logline = match.group(1).strip()
        if re.match(r"\s*CONTINUITY\s+NOTES\s*:", line, re.I):
            note = line.split(":", 1)[1].strip()
            if note:
                notes.append(note)
            notes_started = True
            continue
        if notes_started:
            if re.match(r"\s*---\s*SCRIPT", line, re.I):
                notes_started = False
            elif line.strip():
                notes.append(line.strip())
        sm = SCENE_RE.match(line)
        if sm:
            if scene_counter:
                save_scene()
            scene_counter += 1
            scene_id = f"scene-{int(sm.group(1)):02d}"
            current_scene = {"scene_id": scene_id, "heading": "", "start_ms": None, "end_ms": None}
        sm = SCENE_META_RE.match(line)
        if sm:
            key, value = sm.group(1).lower(), sm.group(2)
            if key == "heading":
                current_scene["heading"] = value
            elif SCENE_TIME_RE.match(value):
                try:
                    current_scene[f"{key}_ms"] = to_ms(value)
                except ValueError as exc:
                    diagnostics.append(f"{current_scene['scene_id']} {key}: {exc}")

    events: list[Event] = []
    pending: dict | None = None
    active_label = ("DIALOGUE", "", "")
    active_scene_id = "scene-unknown"
    source_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        content = "\n".join(part.rstrip() for part in pending["content"] if part.strip()).strip()
        if not content:
            diagnostics.append(f"Line {pending['line']}: timed block had no content")
        else:
            try:
                events.append(Event(
                event_id=f"{source_hash[:10]}-{len(events)+1:04}",
                scene_id=pending["scene_id"],
                start_ms=pending["start"], end_ms=pending["end"],
                event_type=pending["type"], speaker=pending["speaker"], delivery=pending["delivery"], content=content,
                source_reference=f"{source_name}:{pending['line']}", source_order=pending["line"],
                ))
            except ValidationError as exc:
                diagnostics.append(f"{source_name}:{pending['line']}: invalid event: {exc.errors(include_url=False)}")
        pending = None

    for line_no, line in enumerate(lines, 1):
        scene_match = SCENE_RE.match(line)
        if scene_match:
            flush()
            active_scene_id = f"scene-{int(scene_match.group(1)):02d}"
            continue
        match = RANGE_RE.search(line)
        if match:
            flush()
            try:
                start, end = to_ms(match.group(1)), to_ms(match.group(2))
                if end < start:
                    diagnostics.append(f"Line {line_no}: event ends before it starts")
                    continue
                kind, speaker, delivery = active_label
                pending = {"start": start, "end": end, "type": kind, "speaker": speaker, "delivery": delivery,
                           "scene_id": active_scene_id, "content": [], "line": line_no}
            except ValueError as exc:
                diagnostics.append(f"Line {line_no}: {exc}")
            continue
        if re.match(r"\s*Start\s*:", line, re.I):
            diagnostics.append(f"Line {line_no}: malformed Start/End timestamp range: {line.strip()}")
        label = EVENT_LABEL_RE.match(line.strip())
        if label:
            if pending is not None:
                flush()
            active_label = _event_label(label.group(1))
            if label.group(2).strip():
                diagnostics.append(f"Line {line_no}: inline text on event label is not associated with a timestamp")
            continue
        if pending is None:
            continue
        if line.strip() and not re.match(r"\s*(?:SCENE\s+\d+|SCENE\s+(?:HEADING|START|END):|---\s*SCRIPT)", line, re.I):
            pending["content"].append(line)
    flush()
    if scene_counter:
        save_scene()
    events.sort(key=lambda e: (e.start_ms, e.source_order))
    if not events:
        diagnostics.append("No timestamped events recognized; expected Start: (MM:SS.xx) | End: (MM:SS.xx) blocks")
    duration = max([e.end_ms for e in events] + [s.end_ms or 0 for s in scenes], default=0)
    for event in events:
        scene = next((s for s in scenes if s.scene_id == event.scene_id), None)
        if scene and scene.start_ms is not None and event.start_ms < scene.start_ms:
            diagnostics.append(f"{event.source_reference}: starts before {scene.scene_id}")
        if scene and scene.end_ms is not None and event.end_ms > scene.end_ms:
            diagnostics.append(f"{event.source_reference}: ends after {scene.scene_id}")
    title = Path(source_name).stem
    return Episode(series_id=series_id, episode_id=f"ep-{source_hash[:10]}", episode_number=episode_number or title,
                   title=title, logline=logline, continuity_notes="\n".join(notes).strip(), duration_ms=duration,
                   scenes=scenes, source_name=source_name, source_hash=source_hash, events=events, diagnostics=diagnostics)


def load_episodes(input_path: Path, series_id: str = "default") -> tuple[list[Episode], list[str]]:
    sources: list[tuple[str, str]] = []
    failures: list[str] = []
    if input_path.is_file() and zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".txt") and not name.endswith("/"):
                    try:
                        sources.append((name, archive.read(name).decode("utf-8-sig", errors="replace")))
                    except (KeyError, RuntimeError) as exc:
                        failures.append(f"{name}: {exc}")
    elif input_path.is_dir():
        for path in sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"):
            try:
                sources.append((str(path), path.read_text(encoding="utf-8-sig", errors="replace")))
            except OSError as exc:
                failures.append(f"{path}: {exc}")
    elif input_path.is_file() and input_path.suffix.lower() == ".txt":
        sources.append((input_path.name, input_path.read_text(encoding="utf-8-sig", errors="replace")))
    else:
        raise ValueError("Input must be a .txt screenplay, a folder, or a ZIP containing .txt files.")
    if not sources:
        failures.append(f"No .txt screenplay files found in {input_path}")
    episodes = []
    for name, text in sources:
        try:
            episodes.append(parse_episode(text, name, series_id=series_id))
        except (ValidationError, ValueError) as exc:
            failures.append(f"{name}: parsing failed: {exc}")
    return episodes, failures


def chronology_key(episode: Episode) -> tuple[int, str]:
    """Sort key for story order: the number in EPISODE NUMBER, else in the file name."""
    match = re.search(r"(\d+)", episode.episode_number)
    if not match:
        match = re.search(r"episode[-_ ]?(\d+)", episode.source_name, re.I)
    return (int(match.group(1)) if match else 10**9, episode.source_name.lower())
