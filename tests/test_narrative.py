import json

from hookrank.narrative.annotation import parse_annotations
from hookrank.narrative.consensus import build_consensus
from hookrank.narrative.extraction import parse_story
from hookrank.narrative.render import render_episode
from hookrank.narrative.schemas import EpisodeAnnotations, EpisodeStory, CharacterRecord, ThreadRecord
from hookrank.narrative.story import build_series_story
from hookrank.narrative.text import name_in_text, quote_match


# ---- evidence matching ---------------------------------------------------------------------

def test_quote_match_exact_elided_and_rejected():
    source = "Macro close-up of her bloodied fist gripping the emerald ring against the frozen ground."
    assert quote_match("bloodied fist gripping the emerald ring", source) == "exact"
    assert quote_match("bloodied fist gripping the emerald ring as the episode ends", source) is None
    assert quote_match("her bloodied fist gripping the ring", source) == "elided"   # dropped a word
    assert quote_match("white‑knuckled", "a white-knuckled fist") == "exact"     # unicode hyphen
    assert quote_match("ring", "no match here") is None


def test_name_in_text_handles_possessives_and_requires_all_words():
    assert name_in_text("Juliana", "It was Juliana's mother's gown.")
    assert not name_in_text("Lady Catherine", "Catherine arrives.")
    assert name_in_text("Lady Catherine", "Lady Catherine arrives.")


# ---- story extraction verification --------------------------------------------------------

def test_story_parse_verifies_names_and_keeps_every_speaker(episode):
    rendered = render_episode(episode)
    raw = json.dumps({
        "summary": "s",
        "characters": [
            {"label": "LADY IN GOLD GOWN", "name": "Clarissa", "evidence": ["E002"]},   # stated in E002
            {"label": "GERALD", "name": "Duke Ravencroft", "evidence": ["E999"]},         # not in text, bad ID
        ],
        "threads": [{"key": "T1", "summary": "groom stolen", "kind": "mystery", "participants": ["Juliana"],
                     "status": "introduced", "evidence": ["E003"]}],
    })
    story = parse_story(raw, rendered, "m")
    by_label = {c.label: c for c in story.characters}
    assert by_label["LADY IN GOLD GOWN"].name == "Clarissa" and by_label["LADY IN GOLD GOWN"].name_verified
    assert by_label["GERALD"].name == "GERALD" and not by_label["GERALD"].name_verified
    assert "JULIANA" in by_label                                   # model skipped her; added back
    assert story.threads[0].kind == "other"                        # unknown kind normalised
    kinds = {i.kind for i in story.issues}
    assert {"unverified_name", "invalid_id", "missing_speaker", "unknown_thread_kind"} <= kinds


# ---- annotation verification and ID repair ------------------------------------------------

def _annotate(episode, items, model="m"):
    rendered = render_episode(episode)
    result = EpisodeAnnotations(episode_id=episode.episode_id, model=model)
    parse_annotations(json.dumps({"events": items}), rendered, model, None, result)
    return result


def test_annotation_repairs_neighbouring_id_and_drops_unsupported(episode):
    result = _annotate(episode, [
        {"id": "E003", "quote": "Juliana slaps Gerald", "tags": ["physical_action"]},  # quote is E004's
        {"id": "E005", "quote": "words nobody ever wrote", "tags": ["reveal"]},        # unsupported
        {"id": "E002", "quote": "he is marrying me", "tags": ["shocking_line", "made_up_tag"]},
        {"id": "E006", "quote": "", "tags": ["striking_visual"]},                       # no evidence
    ])
    by_short = {a.short_id: a for a in result.annotations}
    assert set(by_short) == {"E004", "E002"}
    assert by_short["E004"].repaired_from == "E003" and by_short["E004"].repair_distance == 1
    assert by_short["E002"].tags == ["shocking_line"]
    kinds = [i.kind for i in result.issues]
    assert {"id_repaired", "quote_not_found", "unknown_tag", "missing_quote"} <= set(kinds)


def test_tags_must_fit_the_event_type_including_after_repair(episode):
    result = _annotate(episode, [
        {"id": "E004", "quote": "Juliana slaps Gerald", "tags": ["physical_action"]},          # ACTION: kept
        {"id": "E005", "quote": "I never loved you", "tags": ["physical_action", "shocking_line"]},  # DIALOGUE: physical dropped
        {"id": "E008", "quote": "A cliffhanger chord swells", "tags": ["shocking_line", "striking_visual"]},  # CAMERA
        {"id": "E002", "quote": "how could you betray me", "tags": ["physical_action"]},  # repaired onto E003 (DIALOGUE)
    ])
    by_short = {a.short_id: a for a in result.annotations}
    assert by_short["E004"].tags == ["physical_action"]
    assert by_short["E005"].tags == ["shocking_line"]
    assert by_short["E008"].tags == ["striking_visual"]                    # a camera line is not a spoken line
    assert "E003" not in by_short                                          # repair does not carry the tag over
    assert sum(i.kind == "tag_not_applicable" for i in result.issues) == 3


# ---- consensus -------------------------------------------------------------------------------

def test_consensus_counts_coreferent_support_but_not_distant_events(episode):
    a = _annotate(episode, [{"id": "E004", "quote": "slaps Gerald", "tags": ["confrontation"]},
                            {"id": "E007", "quote": "who sent the letter", "tags": ["open_question"]}], model="A")
    b = _annotate(episode, [{"id": "E005", "quote": "I never loved you", "tags": ["confrontation"]},   # adjacent: 0s gap
                            {"id": "E002", "quote": "marrying me", "tags": ["open_question"]}], model="B")    # far away
    records = {r.short_id: r for r in build_consensus(episode, [a, b])}
    assert sorted(records["E004"].tags["confrontation"]) == ["A", "B"]
    assert records["E007"].tags["open_question"] == ["A"]


# ---- series identity and thread linking ----------------------------------------------------

def _story(ep, characters, threads=()):
    return EpisodeStory(episode_id=ep.episode_id, episode_number=ep.episode_number, model="m",
                        characters=[CharacterRecord(**c) for c in characters],
                        threads=[ThreadRecord(**t) for t in threads])


def test_identity_resolution_merges_aliases_and_refuses_conflicts(episode_factory):
    ep1, ep2 = episode_factory(1), episode_factory(2)
    stories = {
        ep1.episode_id: _story(ep1, [
            {"label": "LADY IN GOLD GOWN", "name": "Clarissa", "name_verified": True, "speaking": True},
            {"label": "JULIANA", "name": "JULIANA", "name_verified": False, "speaking": True}]),
        ep2.episode_id: _story(ep2, [   # a wrong claim that would fuse two named characters
            {"label": "LADY IN GOLD GOWN", "name": "Catherine", "name_verified": True, "speaking": True}]),
    }
    series, index = build_series_story("s", [ep1, ep2], stories, {})
    clarissa = index.resolve("LADY IN GOLD GOWN")
    assert clarissa == index.resolve("Clarissa")
    assert index.resolve("Catherine") != clarissa
    assert any(i.kind == "merge_refused" for i in series.issues)


def test_thread_linking_ignores_protagonist_only_overlap(episode_factory):
    ep1, ep2 = episode_factory(1), episode_factory(2)
    chars = [{"label": "JULIANA", "name": "JULIANA", "name_verified": False, "speaking": True},
             {"label": "GERALD", "name": "GERALD", "name_verified": False, "speaking": True},
             {"label": "LADY IN GOLD GOWN", "name": "Clarissa", "name_verified": True, "speaking": True}]
    stories = {
        ep1.episode_id: _story(ep1, chars, [
            {"key": "T1", "summary": "betrayal", "kind": "betrayal", "participants": ["Juliana", "Gerald", "Clarissa"], "status": "introduced"},
            {"key": "T2", "summary": "resolve", "kind": "revenge", "participants": ["Juliana"], "status": "introduced"}]),
        ep2.episode_id: _story(ep2, chars, [
            {"key": "T1", "summary": "betrayal again", "kind": "betrayal", "participants": ["Gerald", "Clarissa"], "status": "escalated"},
            {"key": "T2", "summary": "poison plot", "kind": "revenge", "participants": ["Juliana", "Clarissa"], "status": "introduced"}]),
    }
    series, _ = build_series_story("s", [ep1, ep2], stories, {})
    m = series.thread_map
    assert m[f"{ep1.episode_id}:T1"] == m[f"{ep2.episode_id}:T1"]          # 2 shared participants, same kind
    assert m[f"{ep1.episode_id}:T2"] != m[f"{ep2.episode_id}:T2"]          # one-person arc vs two-person plot


def test_truncated_annotation_is_split_into_windows_and_not_reported_as_failure(tmp_path, monkeypatch):
    from hookrank.cache import CallCache
    from hookrank.llm import ChatResult, ModelSpec, TruncatedResponse
    from hookrank.narrative.engine import NarrativeEngine
    from hookrank.parsing import parse_episode

    lines = "\n".join(f'**JULIANA (Calm):**\nStart: (00:{i:02d}.00) | End: (00:{i:02d}.90)\n"Line number {i}."'
                      for i in range(30))
    ep = parse_episode(f"EPISODE NUMBER: Episode 1\n{lines}\n", "episode-001.txt", "s")
    full_window_calls = []

    def fake_complete(self, prompt, min_output_tokens=800):
        if prompt.startswith("EVENT ANNOTATION") and "E001 " in prompt and "E030 " in prompt:
            full_window_calls.append(1)
            raise TruncatedResponse("hit output budget")
        return ChatResult(text='{"events": []}' if prompt.startswith("EVENT") else '{"summary": "s"}',
                          model=self.spec.model, usage={}, finish_reason="stop", latency_s=0, attempts=1,
                          waited_s=0, max_output_tokens=1000)

    monkeypatch.setattr("hookrank.llm.ChatClient.complete", fake_complete)
    spec = ModelSpec(model="m", base_url="u", api_key="k")
    cache = CallCache(tmp_path / "c.sqlite3")
    for _ in range(2):   # second run: the truncation is cached, so no full-window call is repeated
        result = NarrativeEngine(spec, [spec], cache, "s").run([ep])
        annotations = result.episodes[ep.episode_id].annotations[0]
        calls = result.episodes[ep.episode_id].calls
        assert annotations.ok and annotations.chunks > 1
        assert any(c.superseded for c in calls) and not any(c.failed for c in calls)
    assert len(full_window_calls) == 1
    cache.close()


def test_shared_story_and_annotator_model_runs_in_parallel_without_deadlock(tmp_path, monkeypatch, episode_factory):
    """One model doing story extraction AND annotation with several workers must neither deadlock
    nor be serialised (annotation is queued only after its story finishes)."""
    import threading
    import time
    from hookrank.cache import CallCache
    from hookrank.llm import ChatResult, ModelSpec
    from hookrank.narrative.engine import NarrativeEngine

    active, peak, lock = {}, {}, threading.Lock()

    def fake_complete(self, prompt, min_output_tokens=800):
        m = self.spec.model
        with lock:
            active[m] = active.get(m, 0) + 1
            peak[m] = max(peak.get(m, 0), active[m])
        time.sleep(0.05)
        with lock:
            active[m] -= 1
        text = '{"events": []}' if prompt.startswith("EVENT") else '{"summary": "s"}'
        return ChatResult(text=text, model=m, usage={}, finish_reason="stop", latency_s=0,
                          attempts=1, waited_s=0, max_output_tokens=1000)

    monkeypatch.setattr("hookrank.llm.ChatClient.complete", fake_complete)
    shared = ModelSpec(model="shared", base_url="u", api_key="k", concurrency=2)   # story + annotator, only 2 workers
    other = ModelSpec(model="other", base_url="u", api_key="k", concurrency=4)
    episodes = [episode_factory(n) for n in range(1, 9)]
    result = NarrativeEngine(shared, [other, shared], CallCache(tmp_path / "c.sqlite3"), "s").run(episodes)
    assert len(result.episodes) == 8 and all(len(n.annotations) == 2 and n.story.ok for n in result.episodes.values())
    assert peak["shared"] > 1 and peak["other"] > 1      # both models genuinely ran requests in parallel
