import json
import re
from pathlib import Path

from hookrank.config import Settings
from hookrank.llm import ChatResult
from hookrank.run import run

from conftest import SCRIPT

STORY = {"summary": "Juliana learns Clarissa stole her groom.",
         "characters": [{"label": "LADY IN GOLD GOWN", "name": "Clarissa", "evidence": ["E002"]}],
         "threads": [{"key": "T1", "summary": "stolen groom", "kind": "betrayal",
                      "participants": ["Juliana", "Gerald", "Clarissa"], "status": "introduced", "evidence": ["E002"]}]}
ANNOTATIONS = {"events": [
    {"id": "E004", "quote": "Juliana slaps Gerald", "tags": ["physical_action"], "thread": "T1", "prior_context": "none"},
    {"id": "E007", "quote": "who sent the letter", "tags": ["open_question"], "question": "Who sent it?",
     "thread": "T1", "prior_context": "none"}]}


def test_end_to_end_run_with_stubbed_models(tmp_path: Path, monkeypatch):
    calls = []

    def fake_complete(self, prompt, min_output_tokens=800):
        calls.append(self.spec.model)
        if prompt.startswith("CLIP CHECK"):   # cite the first clip event for the opening, the last for the ending
            ids = re.findall(r"^(C\d{2}) ", prompt, flags=re.M)
            body = {"opening": {"type": "attack_or_danger", "what_happens": "a slap", "evidence_event_ids": [ids[0]]},
                    "central_question": {"text": "Who sent the letter?", "evidence_event_ids": [ids[-1]]},
                    "answering_event_ids": [],
                    "ending": {"type": "question_or_demand_awaiting_answer", "what_happens": "asks",
                               "evidence_event_ids": [ids[-1]]},
                    "missing_context": [], "jump_problem": "none"}
        else:
            body = STORY if prompt.startswith("NARRATIVE EXTRACTION") else ANNOTATIONS
        return ChatResult(text=json.dumps(body), model=self.spec.model, usage={"prompt_tokens": 10, "completion_tokens": 5},
                          finish_reason="stop", latency_s=0.0, attempts=1, waited_s=0.0, max_output_tokens=1000)

    monkeypatch.setattr("hookrank.llm.ChatClient.complete", fake_complete)
    source = tmp_path / "scripts"
    source.mkdir()
    for n in (1, 2):
        (source / f"episode-{n:03d}.txt").write_text(SCRIPT.format(n=n), encoding="utf-8")
    settings = Settings.from_mapping({"OPENAI_API_KEY": "test"}, story_model="story-m", annotator_models=("ann-a", "ann-b"),
                                     verifier_model="verify-m", spend_ledger=str(tmp_path / "ledger.sqlite3"))
    out = tmp_path / "out"

    outcome = run(source, out, "demo", settings, top_n=5, verify_top=3)
    m = outcome["manifest"]
    assert m["episodes"] == 2 and m["model_calls"]["failed"] == 0
    narrative_calls = [c for c in calls if c != "verify-m"]
    assert sorted(narrative_calls) == sorted(["story-m", "ann-a", "ann-b"] * 2)
    # clip check ran on the top candidates; identical clip text (the two test episodes share a script) is
    # answered from cache, so live calls can be fewer than checked promos
    assert 0 < calls.count("verify-m") <= m["clip_check"]["checked"] <= 3
    assert m["clip_check"]["recommendations"]["accept"] == m["clip_check"]["checked"]
    assert m["clip_check"]["shortlist_verified"] == m["promos"]["ranked"]
    for name in ("episodes.json", "narrative.json", "promos_ranked.json", "promos_ranked.csv", "promos_by_episode.json",
                 "promos_all.json", "promos_report.md", "failures.json", "run_manifest.json"):
        assert (out / name).exists(), name
    ranked = json.loads((out / "promos_ranked.json").read_text())
    assert ranked and ranked[0]["scores"]["valid"] and ranked[0]["scores"]["confidence_label"] == "high"
    assert ranked[0]["clip_check"]["recommendation"] == "accept"
    assert ranked[0]["clip_check"]["ending"]["unresolved_question"] == "Who sent the letter?"
    assert "Clarissa" in json.dumps(json.loads((out / "narrative.json").read_text())["series"]["characters"])

    calls.clear()                                    # second run: everything from cache, no model calls
    again = run(source, out, "demo", settings, top_n=5, verify_top=3)
    assert calls == [] and again["manifest"]["model_calls"]["cached"] == 6 + m["clip_check"]["checked"]
