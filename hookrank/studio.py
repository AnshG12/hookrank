"""Thin adapter between the Streamlit app (streamlit_app.py) and the existing HookRank pipeline.

It adds no AI logic. It validates uploads, isolates each analysis in its own temporary
directory, estimates and caps spend, reports progress, calls ``hookrank.run.run`` (the same
entry point as ``python main.py run``) and prepares display rows. No Streamlit import, so it
is unit-testable.

Security model (docs/decisions.md D15):
  * the Nebius key is read server-side from Streamlit secrets or environment variables and is
    only ever used inside the existing model client; it is never returned to the UI;
  * every run gets a fresh temp directory (inputs, cache, outputs, spend ledger), deleted when
    the run ends, so sessions never share files and nothing is kept after a run;
  * each run has its own spend ledger, so the per-run cap is exact even though hosted storage
    is ephemeral; an app-wide daily counter is a second, soft layer (it resets on restart, so
    the provider-side spending limit remains the real backstop).
"""
from __future__ import annotations

import io
import json
import logging
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping

from .budget import fetch_prices
from .config import Settings, split_model
from .llm import CHARS_PER_TOKEN
from .parsing import chronology_key, load_episodes
from .promos.schemas import Promo
from .run import run
from .schemas import Episode, Event

log = logging.getLogger("hookrank.studio")

# Model roles chosen by experiments S4/S5/S7 (D11, D12, D14). Overridable through secrets.
STUDIO_DEFAULTS = {
    "HOOKRANK_STORY_MODEL": "nebius:openai/gpt-oss-120b",
    "HOOKRANK_ANNOTATOR_MODELS": "nebius:Qwen/Qwen3-235B-A22B-Instruct-2507,nebius:openai/gpt-oss-120b",
    "HOOKRANK_VERIFIER_MODEL": "nebius:openai/gpt-oss-120b",
}
TOP_N = 10
VERIFY_TOP = 40          # clip-check budget for the top-10 shortlist and alternatives (30 left 1 of 10 unchecked in a live 3-episode run)
EPISODE_TRIES = 0        # the app shows only the top 10, so per-episode picks are not checked


class UploadError(ValueError):
    """The upload was rejected; the message is safe to show to the user."""


@dataclass(frozen=True)
class Limits:
    max_upload_bytes: int = 20 * 1024 * 1024        # all uploaded files together
    max_files: int = 100                            # individual .txt uploads
    max_zip_entries: int = 400                      # all entries listed in a ZIP
    max_txt_files: int = 100                        # .txt screenplays in a ZIP
    max_file_bytes: int = 2 * 1024 * 1024           # one screenplay
    max_total_text_bytes: int = 30 * 1024 * 1024    # extracted text in total
    max_compression_ratio: int = 100                # zip-bomb guard
    max_episodes: int = 60
    max_total_chars: int = 1_500_000                # bounds prompt tokens per run


# ---- uploads ---------------------------------------------------------------------------------

def _safe_member_name(name: str) -> str | None:
    """Return a safe basename for a ZIP member, or None if the entry must be skipped/rejected."""
    if "\x00" in name:
        raise UploadError("The ZIP contains an invalid file name.")
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", name):
        raise UploadError("The ZIP contains unsafe paths (absolute or '..'); it was rejected.")
    if name.endswith("/") or any(p.startswith(".") or p == "__MACOSX" for p in path.parts):
        return None                                   # directories, hidden files, macOS metadata
    return path.name if path.suffix.lower() == ".txt" else None


def _sanitize(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).stem).strip() or "episode"
    return f"{stem[:80]}.txt"


def _decode(raw: bytes, name: str) -> str:
    text = raw.decode("utf-8-sig", errors="replace")
    if not text.strip():
        raise UploadError(f"{name} is empty.")
    return text


def _read_zip(data: bytes, limits: Limits) -> list[tuple[str, str]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UploadError("The file is not a valid ZIP archive.") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > limits.max_zip_entries:
            raise UploadError(f"The ZIP has {len(infos)} entries; the limit is {limits.max_zip_entries}.")
        sources: list[tuple[str, str]] = []
        total = 0
        for info in infos:
            name = _safe_member_name(info.filename)
            if name is None:
                continue
            if info.flag_bits & 0x1:
                raise UploadError("Encrypted ZIP entries are not supported.")
            if info.file_size > limits.max_file_bytes:
                raise UploadError(f"{name} is larger than {limits.max_file_bytes // 1024 // 1024} MB.")
            if info.compress_size and info.file_size / info.compress_size > limits.max_compression_ratio:
                raise UploadError("The ZIP looks like a compression bomb and was rejected.")
            if len(sources) >= limits.max_txt_files:
                raise UploadError(f"The ZIP contains more than {limits.max_txt_files} screenplay files.")
            with archive.open(info) as fp:                 # never trust the declared size
                raw = fp.read(limits.max_file_bytes + 1)
            if len(raw) > limits.max_file_bytes:
                raise UploadError(f"{name} is larger than {limits.max_file_bytes // 1024 // 1024} MB.")
            total += len(raw)
            if total > limits.max_total_text_bytes:
                raise UploadError("The ZIP's extracted text is too large.")
            sources.append((name, _decode(raw, name)))
        return sources


def validate_uploads(files: list[tuple[str, bytes]], limits: Limits = Limits()) -> list[tuple[str, str]]:
    """Turn uploaded (name, bytes) pairs into safe (filename, text) screenplays, or raise UploadError."""
    if not files:
        raise UploadError("Upload a ZIP of screenplay episodes or one or more .txt files.")
    if len(files) > limits.max_files:
        raise UploadError(f"Too many files; the limit is {limits.max_files}.")
    if sum(len(data) for _, data in files) > limits.max_upload_bytes:
        raise UploadError(f"The upload is larger than {limits.max_upload_bytes // 1024 // 1024} MB.")
    sources: list[tuple[str, str]] = []
    for name, data in files:
        lower = name.lower()
        if lower.endswith(".zip"):
            sources += _read_zip(data, limits)
        elif lower.endswith(".txt"):
            if len(data) > limits.max_file_bytes:
                raise UploadError(f"{name} is larger than {limits.max_file_bytes // 1024 // 1024} MB.")
            sources.append((Path(name).name, _decode(data, name)))
        else:
            raise UploadError(f"{name}: only .zip and .txt files are accepted.")
    if not sources:
        raise UploadError("No .txt screenplay files were found in the upload.")
    if sum(len(text) for _, text in sources) > limits.max_total_chars:
        raise UploadError("The screenplays are too long for one analysis; split them into smaller uploads.")
    return sources


def write_inputs(sources: list[tuple[str, str]], root: Path) -> Path:
    """Write screenplays into ``root/input`` under sanitized, unique names (no paths from the upload)."""
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    used: set[str] = set()
    for name, text in sources:
        safe = _sanitize(name)
        stem, n = safe[:-4], 2
        while safe.lower() in used:
            safe, n = f"{stem}-dup{n}.txt", n + 1
        used.add(safe.lower())
        (input_dir / safe).write_text(text, encoding="utf-8")
    return input_dir


@dataclass
class Preview:
    episodes: list[Episode]
    unreadable: list[str]

    @property
    def usable(self) -> list[Episode]:
        return [e for e in self.episodes if e.events]

    @property
    def events(self) -> int:
        return sum(len(e.events) for e in self.episodes)


def preview(sources: list[tuple[str, str]], limits: Limits = Limits()) -> Preview:
    """Parse locally (no model calls) to show the detected episodes before anything is sent anywhere."""
    with tempfile.TemporaryDirectory(prefix="hookrank-preview-") as tmp:
        episodes, failures = load_episodes(write_inputs(sources, Path(tmp)), series_id="preview")
    episodes.sort(key=chronology_key)
    result = Preview(episodes=episodes, unreadable=failures)
    if not result.usable:
        raise UploadError("No timestamped events were recognised; check the screenplay format "
                          "(Start: (MM:SS.cc) | End: (MM:SS.cc) blocks).")
    if len(result.usable) > limits.max_episodes:
        raise UploadError(f"{len(result.usable)} episodes found; the limit per analysis is {limits.max_episodes}.")
    return result


# ---- settings, access and spend ----------------------------------------------------------------

def studio_settings(secrets: Mapping[str, object], run_ledger: Path, run_budget_usd: float) -> Settings:
    """Settings for one analysis: Nebius roles, the server-side key, and a per-run ledger and cap."""
    get = lambda key, default="": str(secrets.get(key) or default)
    base = Settings.from_env() if not get("NEBIUS_API_KEY") else Settings()
    roles = {k: get(k, v) for k, v in STUDIO_DEFAULTS.items()}
    return replace(base,
                   nebius_api_key=get("NEBIUS_API_KEY", base.nebius_api_key),
                   nebius_concurrency=int(get("NEBIUS_CONCURRENCY", base.nebius_concurrency)),
                   story_model=roles["HOOKRANK_STORY_MODEL"],
                   annotator_models=tuple(m.strip() for m in roles["HOOKRANK_ANNOTATOR_MODELS"].split(",") if m.strip()),
                   verifier_model=roles["HOOKRANK_VERIFIER_MODEL"],
                   budget_usd=run_budget_usd, spend_ledger=str(run_ledger))


class DailyBudget:
    """App-wide spend per calendar day, shared by all sessions in this process (thread-safe).
    Soft limit: it resets when the hosted app restarts, so a provider-side limit is still needed."""

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        self._day = date.today()
        self._spent = 0.0

    def remaining(self) -> float:
        with self._lock:
            if date.today() != self._day:
                self._day, self._spent = date.today(), 0.0
            return self.cap_usd - self._spent

    def record(self, usd: float) -> None:
        with self._lock:
            self._spent += usd


@dataclass
class Estimate:
    calls: int
    prompt_tokens: int
    output_tokens: int
    usd: float


def estimate(episodes: list[Episode], settings: Settings, prices: dict | None = None) -> Estimate:
    """Upper-bound-ish estimate of one run's model usage and cost, from script size and roles."""
    refs = [settings.story_model, *settings.annotator_models]
    per_episode_chars = [sum(len(e.content) + 40 for e in ep.events) + 3000 for ep in episodes]
    prompt = sum(int(c / CHARS_PER_TOKEN) for c in per_episode_chars) * len(refs) + VERIFY_TOP * 3000
    output = 2000 * len(episodes) * len(refs) + VERIFY_TOP * 700
    usd = 0.0
    if prices:
        models = [split_model(r)[1] for r in refs]
        priced = [prices[m] for m in models if m in prices]
        if priced:
            avg_in = sum(p.prompt_per_token for p in priced) / len(priced)
            avg_out = sum(p.completion_per_token for p in priced) / len(priced)
            usd = prompt * avg_in + output * avg_out
    return Estimate(calls=len(episodes) * len(refs) + VERIFY_TOP, prompt_tokens=prompt, output_tokens=output,
                    usd=round(usd, 4))


def nebius_prices(settings: Settings) -> dict:
    endpoint = settings.endpoint("nebius")
    return fetch_prices(endpoint.base_url, endpoint.api_key) if endpoint.api_key else {}


# ---- progress ----------------------------------------------------------------------------------

class Progress(logging.Handler):
    """Counts finished model calls from the pipeline's own log records ("<label> <stage> <model> ok|failed").
    Log records carry episode numbers and token counts only, never screenplay text or credentials.
    The app allows one analysis at a time, so the counts belong to that analysis."""

    def __init__(self, narrative_calls: int) -> None:
        super().__init__(level=logging.INFO)
        self.narrative_calls = max(1, narrative_calls)
        self.counts = {"story": 0, "annotation": 0, "clip_check": 0}
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        for stage in self.counts:
            if f" {stage} " in message:
                with self._lock:
                    self.counts[stage] += 1
                return

    def snapshot(self) -> tuple[float, str]:
        with self._lock:
            done = self.counts["story"] + self.counts["annotation"]
            checks = self.counts["clip_check"]
        if done < self.narrative_calls:
            return 0.85 * done / self.narrative_calls, f"Reading episodes and labelling events ({done}/{self.narrative_calls} model calls)"
        return min(0.98, 0.85 + 0.13 * checks / VERIFY_TOP), f"Building and checking promo cuts ({checks} clips checked)"


# ---- analysis ----------------------------------------------------------------------------------

@dataclass
class Analysis:
    promos: list[Promo]
    episodes: dict[str, Episode]
    csv_bytes: bytes
    json_bytes: bytes
    episode_count: int
    candidates: int
    spend_usd: float
    seconds: float
    warnings: list[str] = field(default_factory=list)


def analyze(sources: list[tuple[str, str]], secrets: Mapping[str, object], run_budget_usd: float,
            progress: Progress | None = None, on_settings: Callable[[Settings], None] | None = None) -> Analysis:
    """Run the existing pipeline on the uploaded screenplays in an isolated temp dir, then delete it."""
    work = Path(tempfile.mkdtemp(prefix=f"hookrank-{uuid.uuid4().hex[:8]}-"))
    calls_logger = logging.getLogger("hookrank.calls")
    previous_level = calls_logger.level
    try:
        input_dir = write_inputs(sources, work)
        settings = studio_settings(secrets, work / "spend.sqlite3", run_budget_usd)
        if settings.missing_keys():
            raise UploadError("The server is missing its Nebius configuration; contact the app owner.")
        if on_settings:
            on_settings(settings)
        if progress:
            calls_logger.setLevel(logging.INFO)
            calls_logger.addHandler(progress)
        outcome = run(input_dir, work / "output", f"upload-{uuid.uuid4().hex[:8]}", settings,
                      top_n=TOP_N, verify_top=VERIFY_TOP, episode_tries=EPISODE_TRIES)
        manifest = outcome["manifest"]
        out = work / "output"
        warnings = []
        if manifest["model_calls"]["failed"]:
            warnings.append(f"{manifest['model_calls']['failed']} model calls failed; results use the remaining evidence.")
        unverified = sum(1 for p in outcome["ranked"] if not (p.clip_check and p.clip_check.get("status") == "verified"))
        if unverified:
            warnings.append(f"{unverified} of the top promos could not be clip-checked within the run's limits.")
        from datetime import datetime
        seconds = (datetime.fromisoformat(manifest["finished_utc"]) - datetime.fromisoformat(manifest["started_utc"])).total_seconds()
        return Analysis(
            promos=outcome["ranked"], episodes={eid: n.episode for eid, n in outcome["narrative"].episodes.items()},
            csv_bytes=(out / "promos_ranked.csv").read_bytes(), json_bytes=(out / "promos_ranked.json").read_bytes(),
            episode_count=manifest["episodes"], candidates=manifest["promos"]["valid_candidates"],
            spend_usd=float(manifest["spend"]["this_run_usd"]), seconds=round(seconds, 1), warnings=warnings)
    finally:
        if progress:
            calls_logger.removeHandler(progress)
            calls_logger.setLevel(previous_level)
        shutil.rmtree(work, ignore_errors=True)      # inputs, cache, outputs and ledger never outlive the run


# ---- display -----------------------------------------------------------------------------------

def fmt(ms: int) -> str:
    return f"{ms // 60000:02d}:{(ms % 60000) // 1000:02d}.{(ms % 1000) // 10:02d}"


def excerpt_rows(promo: Promo, episodes: dict[str, Episode], opening_ms: int = 3000, ending_ms: int = 6000) -> list[dict]:
    """Timestamped excerpt of the promo, with the hook and cliffhanger evidence marked."""
    rows: list[dict] = []
    offset = 0
    total = int(promo.duration_seconds * 1000)
    for i, seg in enumerate(promo.segments):
        if i:
            rows.append({"cut": True, "text": f"Cut to {seg.episode} {seg.start}"})
        for e in episodes[seg.episode_id].events:
            if not (e.start_ms < seg.end_ms and e.end_ms > seg.start_ms):
                continue
            rel_start = offset + max(0, e.start_ms - seg.start_ms)
            rel_end = offset + min(seg.end_ms, e.end_ms) - seg.start_ms
            role = ("hook" if e.event_id == promo.hook.event_id else
                    "cliffhanger" if e.event_id == promo.cliffhanger.event_id else
                    "opening" if rel_start < opening_ms else "ending" if rel_end >= total - ending_ms else "")
            rows.append({"cut": False, "episode": seg.episode, "script_time": f"{fmt(e.start_ms)}-{fmt(e.end_ms)}",
                         "promo_time": f"{rel_start / 1000:4.1f}s", "label": _label(e),
                         "text": " ".join(e.content.split()), "role": role})
        offset += seg.end_ms - seg.start_ms
    return rows


def _label(e: Event) -> str:
    if e.event_type == "DIALOGUE":
        return (e.speaker or "Speaker").title() + (f" ({e.delivery})" if e.delivery else "")
    return e.event_type.title()


def summary_rows(promos: list[Promo]) -> list[dict]:
    return [{"Rank": p.rank, "Episode(s)": ", ".join(dict.fromkeys(s.episode.replace("Episode ", "Ep ") for s in p.segments)),
             "Timestamps": " + ".join(f"{s.start}-{s.end}" for s in p.segments),
             "Duration": f"{p.duration_seconds:.0f}s", "Quality": round(p.scores.quality),
             "Confidence": p.scores.confidence_label} for p in promos]


def parse_json_export(data: bytes) -> list[dict]:
    return json.loads(data.decode("utf-8"))
