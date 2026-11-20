from hookrank.narrative.schemas import ConsensusEvent
from hookrank.narrative.story import build_series_story
from hookrank.promos.generate import EpisodeContext, evaluate, generate
from hookrank.promos.rank import rank_promos


def _context(episode, tags_by_short: dict[str, dict[str, list[str]]], rank: int = 0) -> EpisodeContext:
    """Episode context with hand-written consensus tags: {"E004": {"physical_action": ["A", "B"]}}."""
    events = episode.events
    evidence = {}
    for short, tags in tags_by_short.items():
        e = events[int(short[1:]) - 1]
        evidence[e.event_id] = ConsensusEvent(event_id=e.event_id, short_id=short, episode_id=episode.episode_id,
                                              start_ms=e.start_ms, end_ms=e.end_ms, event_type=e.event_type,
                                              tags=tags, direct_models=sorted({m for ms in tags.values() for m in ms}))
    _, index = build_series_story("s", [episode], {}, {})
    return EpisodeContext(episode=episode, evidence=evidence, index=index, rank=rank)


def test_cut_points_never_fall_inside_dialogue(episode):
    ctx = _context(episode, {})
    assert ctx.is_legal_cut(9_000)            # between a line and the slap
    assert not ctx.is_legal_cut(16_000)       # inside Gerald's line (14.0-20.0)
    assert all(ctx.is_legal_cut(t) for t in ctx.legal_starts + ctx.legal_ends)


def test_promo_opens_on_hook_and_cuts_after_unresolved_question(episode):
    ctx = _context(episode, {
        "E004": {"physical_action": ["A", "B"]},       # slap at 9.0s
        "E007": {"open_question": ["A", "B"]},         # "who sent the letter?" ends 36.0s
    })
    promos = [evaluate(c) for c in generate([ctx])]
    assert promos, "expected a 9.0s -> 36.0s candidate"
    best = max(promos, key=lambda p: p.scores.quality)
    seg = best.segments[0]
    assert (seg.start_ms, seg.end_ms) == (9_000, 36_000)
    assert best.scores.valid and best.scores.confidence_label == "high"
    assert best.hook.tags == ["physical_action"] and best.cliffhanger.tags == ["open_question"]
    assert 20 <= best.duration_seconds <= 60


def test_resolved_moment_is_never_a_cliffhanger(episode):
    ctx = _context(episode, {
        "E004": {"physical_action": ["A"]},
        "E009": {"reveal": ["A"], "resolution": ["B"]},   # "It was your father. Everything is settled."
    })
    assert all(p.cliffhanger.event_id != episode.events[8].event_id for p in map(evaluate, generate([ctx])))


def test_quality_ignores_agreement_but_confidence_reflects_it(episode):
    one = _context(episode, {"E004": {"physical_action": ["A"]}, "E007": {"open_question": ["A"]}})
    two = _context(episode, {"E004": {"physical_action": ["A", "B"]}, "E007": {"open_question": ["A", "B"]}})
    p1 = max(map(evaluate, generate([one])), key=lambda p: p.scores.quality)
    p2 = max(map(evaluate, generate([two])), key=lambda p: p.scores.quality)
    assert p1.scores.quality == p2.scores.quality
    assert p1.scores.confidence < p2.scores.confidence


def test_ranking_skips_near_duplicates(episode):
    ctx = _context(episode, {
        "E002": {"shocking_line": ["A", "B"]}, "E004": {"physical_action": ["A", "B"]},
        "E007": {"open_question": ["A", "B"]},
    })
    ranked = rank_promos([evaluate(c) for c in generate([ctx])], top_n=10)
    cliffs = [p.cliffhanger.event_id for p in ranked]
    assert len(cliffs) == len(set(cliffs))     # one promo per cliffhanger moment
    assert [p.rank for p in ranked] == list(range(1, len(ranked) + 1))


def test_cross_episode_distance_uses_episode_numbers_not_run_positions(episode_factory):
    from hookrank.promos import _episode_rank
    ep2, ep5 = episode_factory(2), episode_factory(5)
    assert _episode_rank(ep2, 0) == 2 and _episode_rank(ep5, 1) == 5   # adjacent in a subset run, 3 apart in the show


def _verdict(op_type="attack_or_danger", op_ids=None, question="Who sent the letter?", answering=(),
             en_type="question_or_demand_awaiting_answer", en_ids=None, missing=(), jump="none"):
    """Build a classify-and-extract verifier answer (prompts/clip_check.md)."""
    return {"opening": {"type": op_type, "what_happens": "w", "evidence_event_ids": op_ids or []},
            "central_question": {"text": question, "evidence_event_ids": []}, "answering_event_ids": list(answering),
            "ending": {"type": en_type, "what_happens": "w", "evidence_event_ids": en_ids or []},
            "missing_context": list(missing), "jump_problem": jump}


def _best_promo(episode, tags):
    ctx = _context(episode, tags)
    return max((p for p in map(evaluate, generate([ctx])) if p.scores.valid), key=lambda p: p.scores.quality)


def test_clip_check_enforces_windows_resolution_and_strictness(episode):
    import json
    from hookrank.promos.verify import parse_check, render_clip
    promo = _best_promo(episode, {"E004": {"physical_action": ["A", "B"]}, "E007": {"open_question": ["A", "B"]}})
    text, events, duration = render_clip(promo, {episode.episode_id: episode})
    first, last = list(events)[0], list(events)[-1]
    assert text.startswith(f"{first} [00:00.00-") and "Episode" not in text            # clip text only
    check = lambda **kw: parse_check(json.dumps(_verdict(**kw)), events, duration, "v")
    good = check(op_ids=[first], en_ids=[last], missing=["who is Gerald?", "why the letter?", "where are they?"])
    assert good.recommendation == "accept" and good.opening["hook_offset_seconds"] == 0.0  # missing context: advisory
    assert check(op_ids=[last], en_ids=[last]).opening["pass"] is False                 # hook not in first 3s
    assert check(op_type="routine_activity", op_ids=[first], en_ids=[last]).recommendation == "reject"
    settled = check(op_ids=[first], en_ids=[last], answering=[last])
    assert settled.ending["pass"] is False and settled.ending["answered_within_clip"] and settled.recommendation == "reject"
    assert check(op_ids=[first], en_ids=[last], en_type="reaction_after_resolution").recommendation == "reject"
    assert check(op_ids=[first], en_ids=[last], en_type="music_or_camera_only").ending["pass"] is False  # cue only
    assert check(op_ids=[first], en_ids=[last], question="What happens next?").ending["pass"] is False
    invented = check(op_ids=["C99"], en_ids=[last])
    assert invented.opening["pass"] is False and any("not in the clip" in i for i in invented.issues)
    assert check(op_type="exciting", op_ids=[first], en_ids=[last]).status == "unverified"   # unknown category
    assert parse_check("not json", events, duration, "v").status == "unverified"


def test_failed_checks_scale_only_their_component_and_keep_label_values(episode):
    import json
    from hookrank.promos.verify import apply_check, parse_check, render_clip
    promo = _best_promo(episode, {"E004": {"physical_action": ["A", "B"]}, "E007": {"open_question": ["A", "B"]}})
    text, events, duration = render_clip(promo, {episode.episode_id: episode})
    before = dict(promo.scores.components), promo.scores.quality
    apply_check(promo, parse_check(json.dumps(_verdict(op_type="routine_activity", op_ids=[list(events)[0]],
                                                       en_ids=[list(events)[-1]])), events, duration, "v"))
    comps = promo.scores.components
    assert comps["hook"] == round(before[0]["hook_label"] * 0.4, 3) and comps["cliffhanger"] == before[0]["cliffhanger"]
    assert comps["hook_label"] == before[0]["hook_label"] and promo.scores.label_quality == before[1] > promo.scores.quality


def test_rejected_candidates_are_replaced_and_never_shortlisted(episode):
    import json
    from hookrank.calls import CallRecord
    from hookrank.llm import ModelSpec
    from hookrank.promos.verify import select_verified
    ctx = _context(episode, {"E002": {"shocking_line": ["A", "B"]}, "E004": {"physical_action": ["A", "B"]},
                             "E005": {"shocking_line": ["A"]}, "E007": {"open_question": ["A", "B"]},
                             "E009": {"open_question": ["A"]}})
    evaluated = [evaluate(c) for c in generate([ctx])]
    first_pick = rank_promos([p.model_copy(deep=True) for p in evaluated], top_n=1)[0].promo_id
    seen = []

    class FakeCaller:
        def call(self, stage, spec, prompt, episode_id, label, min_output):
            pid = label.split()[-1]
            seen.append(pid)
            import re
            ids = re.findall(r"^(C\d{2}) ", prompt, flags=re.M)
            verdict = _verdict(op_ids=[ids[0]], en_ids=[ids[-1]], answering=[ids[-1]] if pid == first_pick else [])
            return json.dumps(verdict), CallRecord(stage=stage, model="v", cached=False, ok=True), False

    final, checks = select_verified(evaluated, {episode.episode_id: episode}, FakeCaller(),
                                    ModelSpec(model="v", base_url="u", api_key="k"), top_n=1, max_checks=5)
    assert checks[first_pick].recommendation == "reject" and first_pick not in [p.promo_id for p in final]
    assert final and final[0].clip_check["recommendation"] == "accept" and final[0].rank == 1   # alternative entered
    assert len(seen) <= 5                                                                         # budget respected


def test_variety_adjustment_never_changes_raw_quality(episode):
    ctx = _context(episode, {"E002": {"shocking_line": ["A", "B"]}, "E004": {"physical_action": ["A", "B"]},
                             "E005": {"shocking_line": ["A"]}, "E007": {"open_question": ["A", "B"]},
                             "E009": {"open_question": ["A"]}})
    promos = [evaluate(c) for c in generate([ctx])]
    raw = {p.promo_id: p.scores.quality for p in promos}
    ranked = rank_promos(promos, top_n=5)
    assert all(p.scores.quality == raw[p.promo_id] and p.scores.variety_adjusted is not None for p in ranked)
    assert [p.scores.quality for p in map(evaluate, generate([ctx]))] == [p.scores.quality for p in map(evaluate, generate([ctx]))]


def test_resolution_inside_final_window_fails_even_before_the_cliffhanger_event(episode):
    ctx = _context(episode, {"E004": {"physical_action": ["A"]}, "E008": {"resolution": ["B"]},  # resolves at 36s
                             "E009": {"open_question": ["A"]}})                                   # "question" at 38.5s
    promos = [p for p in map(evaluate, generate([ctx])) if p.segments[-1].end_ms == 42_000]   # cut after E009
    assert promos and all(p.scores.validity_checks["ending_unresolved"] is False and not p.scores.valid for p in promos)


def test_production_cues_are_noted_but_not_scored(episode):
    ctx = _context(episode, {"E004": {"physical_action": ["A"]}, "E008": {"escalation": ["A"]}})  # "cliffhanger chord"
    promo = next(p for p in map(evaluate, generate([ctx])) if p.segments[-1].end_ms == 38_000)
    assert promo.scores.components["cliffhanger"] == 0.75                                   # escalation weight, no bonus
    assert any("production cue; not scored" in n for n in promo.notes)


def test_explanations_quote_the_event_and_describe_sounds_as_sound_cues():
    from hookrank.parsing import parse_episode
    script = """EPISODE NUMBER: Episode 9
SCENE 01
Scene start: 00:00.00
Scene end: 00:40.00
**SOUND FX:**
Start: (00:00.00) | End: (00:03.00)
Heavy, frantic pounding echoes against the cottage door.
**MOTHER (Desperate):**
Start: (00:03.00) | End: (00:10.00)
"Please, my boy is burning with fever!"
**CAMERA:**
Start: (00:10.00) | End: (00:30.00)
Slow push-in on the healer weighing the risk.
**HEALER (Quiet):**
Start: (00:30.00) | End: (00:35.00)
"If this fails, they will hang me."
"""
    ep = parse_episode(script, "ep9.txt", "s")
    ctx = _context(ep, {"E001": {"physical_action": ["A"]}, "E004": {"threat": ["A"]}})
    promo = next(p for p in map(evaluate, generate([ctx])) if p.scores.valid)
    assert "a sound cue" in promo.why and "Heavy, frantic pounding echoes" in promo.why
    assert "they will hang me" in promo.why and "clash" not in promo.why
