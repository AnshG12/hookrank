"""Tests for the Streamlit adapter (hookrank/studio.py) and the Streamlit page (streamlit_app.py)."""
import io
import json
import re
import zipfile
from pathlib import Path

import pytest

from hookrank import studio
from hookrank.budget import Price
from hookrank.llm import ChatResult

from conftest import SCRIPT

ROOT = Path(__file__).resolve().parent.parent
FAKE_KEY = "nb-test-secret-key-0000"


def _zip(entries: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


# ---- upload validation ---------------------------------------------------------------------------

def test_single_txt_and_multi_episode_zip_are_accepted():
    single = studio.validate_uploads([("ep1.txt", SCRIPT.format(n=1).encode())])
    assert [n for n, _ in single] == ["ep1.txt"]
    many = studio.validate_uploads([("show.zip", _zip({f"show/episode-{n}.txt": SCRIPT.format(n=n) for n in (1, 2, 3)}
                                                      | {"__MACOSX/._episode-1.txt": "x", "show/notes.pdf": "x"}))])
    assert sorted(n for n, _ in many) == ["episode-1.txt", "episode-2.txt", "episode-3.txt"]   # metadata/non-txt skipped


@pytest.mark.parametrize("entries, message", [
    ({"../evil.txt": "x"}, "unsafe paths"),
    ({"/etc/passwd.txt": "x"}, "unsafe paths"),
    ({"bomb.txt": b"\0" * (1900 * 1024)}, "compression bomb"),
])
def test_unsafe_zips_are_rejected(entries, message):
    with pytest.raises(studio.UploadError, match=message):
        studio.validate_uploads([("a.zip", _zip(entries))])


def test_invalid_uploads_are_rejected():
    limits = studio.Limits(max_file_bytes=1000, max_txt_files=2)
    with pytest.raises(studio.UploadError, match="only .zip and .txt"):
        studio.validate_uploads([("script.pdf", b"x")])
    with pytest.raises(studio.UploadError, match="larger than"):
        studio.validate_uploads([("big.txt", b"x" * 2000)], limits)
    with pytest.raises(studio.UploadError, match="more than 2"):
        studio.validate_uploads([("a.zip", _zip({f"e{n}.txt": "x" for n in range(3)}))], limits)
    with pytest.raises(studio.UploadError, match="not a valid ZIP"):
        studio.validate_uploads([("a.zip", b"not a zip")])
    with pytest.raises(studio.UploadError, match="empty"):
        studio.validate_uploads([("e.txt", b"   ")])
    with pytest.raises(studio.UploadError, match="No timestamped events"):
        studio.preview([("e.txt", "EPISODE NUMBER: Episode 1\nno timings here")])
    with pytest.raises(studio.UploadError, match="limit per analysis is 1"):
        studio.preview([(f"e{n}.txt", SCRIPT.format(n=n)) for n in (1, 2)], studio.Limits(max_episodes=1))


def test_written_inputs_use_sanitized_unique_names(tmp_path):
    input_dir = studio.write_inputs([("ep 1?.txt", "a"), ("ep 1?.txt", "b"), ("x/../y.txt", "c")], tmp_path)
    names = sorted(p.name for p in input_dir.iterdir())
    assert names == ["ep 1_-dup2.txt", "ep 1_.txt", "y.txt"] and all(p.parent == input_dir for p in input_dir.iterdir())


# ---- end-to-end through the existing pipeline (stubbed models, no network) ---------------------------

STORY = {"summary": "s", "characters": [{"label": "LADY IN GOLD GOWN", "name": "Clarissa", "evidence": ["E002"]}],
         "threads": [{"key": "T1", "summary": "stolen groom", "kind": "betrayal",
                      "participants": ["Juliana", "Gerald", "Clarissa"], "status": "introduced", "evidence": ["E002"]}]}
ANNOTATIONS = {"events": [
    {"id": "E004", "quote": "Juliana slaps Gerald", "tags": ["physical_action", "confrontation"], "thread": "T1"},
    {"id": "E007", "quote": "who sent the letter", "tags": ["open_question"], "question": "Who sent it?", "thread": "T1"}]}


@pytest.fixture
def stub_models(monkeypatch):
    calls = []

    def fake_complete(self, prompt, min_output_tokens=800):
        calls.append(self.spec.model)
        if prompt.startswith("CLIP CHECK"):
            ids = re.findall(r"^(C\d{2}) ", prompt, flags=re.M)
            body = {"opening": {"type": "attack_or_danger", "what_happens": "slap", "evidence_event_ids": [ids[0]]},
                    "central_question": {"text": "Who sent the letter?", "evidence_event_ids": [ids[-1]]},
                    "answering_event_ids": [],
                    "ending": {"type": "question_or_demand_awaiting_answer", "what_happens": "q", "evidence_event_ids": [ids[-1]]},
                    "missing_context": [], "jump_problem": "none"}
        else:
            body = STORY if prompt.startswith("NARRATIVE EXTRACTION") else ANNOTATIONS
        return ChatResult(text=json.dumps(body), model=self.spec.model, usage={"prompt_tokens": 100, "completion_tokens": 50},
                          finish_reason="stop", latency_s=0.0, attempts=1, waited_s=0.0, max_output_tokens=1000)

    monkeypatch.setattr("hookrank.llm.ChatClient.complete", fake_complete)
    monkeypatch.setattr("hookrank.budget.fetch_prices", lambda base_url, api_key, timeout=30: {
        "openai/gpt-oss-120b": Price(1e-7, 1e-7), "Qwen/Qwen3-235B-A22B-Instruct-2507": Price(1e-7, 1e-7)})
    return calls


def test_analysis_runs_the_existing_pipeline_in_an_isolated_temp_dir(stub_models, monkeypatch):
    made = []
    real_mkdtemp = studio.tempfile.mkdtemp
    monkeypatch.setattr(studio.tempfile, "mkdtemp", lambda **kw: made.append(real_mkdtemp(**kw)) or made[-1])
    sources = studio.validate_uploads([("show.zip", _zip({f"episode-{n}.txt": SCRIPT.format(n=n) for n in (1, 2)}))])
    progress = studio.Progress(narrative_calls=2 * 3)
    first = studio.analyze(sources, {"NEBIUS_API_KEY": FAKE_KEY}, 0.5, progress)
    second = studio.analyze(sources, {"NEBIUS_API_KEY": FAKE_KEY}, 0.5)
    assert len(made) == 2 and made[0] != made[1] and not any(Path(p).exists() for p in made)   # isolated, then deleted
    assert {"openai/gpt-oss-120b", "Qwen/Qwen3-235B-A22B-Instruct-2507"} <= set(stub_models)   # real pipeline, Nebius roles
    assert progress.counts["story"] == 2 and progress.counts["annotation"] == 4 and progress.counts["clip_check"] >= 1
    assert 0 < len(first.promos) <= studio.TOP_N and first.episode_count == 2
    exported = json.loads(first.json_bytes)
    assert [p["promo_id"] for p in exported] == [p.promo_id for p in first.promos]
    csv_text = first.csv_bytes.decode()
    assert all(f"{s.start}-{s.end}" in csv_text for p in first.promos for s in p.segments)      # exact timestamps
    assert FAKE_KEY not in csv_text and FAKE_KEY.encode() not in first.json_bytes
    rows = studio.excerpt_rows(first.promos[0], first.episodes)
    assert {r["role"] for r in rows if not r["cut"]} >= {"hook", "cliffhanger"}
    assert second.promos[0].promo_id == first.promos[0].promo_id


def test_missing_server_key_is_reported_without_details(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                          # no local .env
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    sources = [("e.txt", SCRIPT.format(n=1))]
    with pytest.raises(studio.UploadError, match="missing its Nebius configuration"):
        studio.analyze(sources, {}, 0.5)


# ---- the Streamlit page: access control and no inference on re-render --------------------------

def _app(monkeypatch, tmp_path, secrets: dict):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(tmp_path)                          # the app must not pick up the developer's .env
    for key in ("NEBIUS_API_KEY",):
        monkeypatch.delenv(key, raising=False)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    for k, v in secrets.items():
        at.secrets[k] = v
    return at


def _page_text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "error", "warning", "info", "title", "text_input", "button"):
        for el in getattr(at, kind):
            parts.append(str(getattr(el, "value", "")) + str(getattr(el, "label", "")))
    return " ".join(parts)


def test_app_is_open_without_sign_in_and_never_shows_the_key(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, {"NEBIUS_API_KEY": FAKE_KEY}).run()
    assert not at.text_input                                                 # no password box any more
    analyze = [b for b in at.button if b.label == "Analyze"]
    assert analyze and analyze[0].disabled                                   # enabled only after a valid upload
    assert FAKE_KEY not in _page_text(at) and not at.exception


def test_rerendering_results_never_calls_the_pipeline(stub_models, monkeypatch, tmp_path):
    sources = studio.validate_uploads([("e.txt", SCRIPT.format(n=1).encode())])
    analysis = studio.analyze(sources, {"NEBIUS_API_KEY": FAKE_KEY}, 0.5)
    at = _app(monkeypatch, tmp_path, {"NEBIUS_API_KEY": FAKE_KEY})
    monkeypatch.setattr(studio, "analyze", lambda *a, **k: pytest.fail("inference must not run on re-render"))
    at.session_state["analysis"] = analysis
    at.run()
    at.run()                                                                 # a promo selection is a plain rerun
    assert any("Top promos" in s.value for s in at.subheader)
    assert len(at.get("download_button")) == 2
    assert FAKE_KEY not in _page_text(at)
