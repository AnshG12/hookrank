from hookrank.parsing import parse_episode


def test_supported_format_preserves_labels_overlap_and_scene_bounds():
    raw = """EPISODE NUMBER: Episode 6
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: Prior fact.
--- SCRIPT START ---
SCENE 01
SCENE HEADING: INT. ROOM – NIGHT
SCENE START: 00:00.00
SCENE END: 00:25.00
**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of the room.
**JULIANA (Quiet, Breathless):**
Start: (00:02.00) | End: (00:05.00)
"Who is there?"
--- SCRIPT END ---"""
    episode = parse_episode(raw, "ep6.txt", "demo")
    assert episode.episode_number == "Episode 6"
    assert episode.series_id == "demo"
    assert len(episode.events) == 2
    assert episode.events[0].event_type == "CAMERA"
    assert episode.events[1].speaker == "JULIANA"
    assert episode.events[1].delivery == "Quiet, Breathless"
    assert episode.events[0].start_ms == 0
    assert episode.events[1].start_ms == 2000
    assert episode.events[0].scene_id == "scene-01"
    assert episode.events[0].source_reference == "ep6.txt:10"
    assert episode.scenes[0].end_ms == 25000


def test_malformed_timestamp_is_reported_and_overlap_is_kept():
    raw = """SCENE 01
Scene start: 00:00.00
Scene end: 00:40.00
**ACTION:**
Start: (00:00.00) | End: (00:20.00)
First
**DIALOGUE (Urgent):**
Start: (00:01.00) | End: (00:22.00)
Second
Start: invalid
"""
    episode = parse_episode(raw, "synthetic.txt")
    assert len(episode.events) == 2
    assert episode.events[1].start_ms < episode.events[0].end_ms
    assert any("malformed Start/End" in warning for warning in episode.diagnostics)
