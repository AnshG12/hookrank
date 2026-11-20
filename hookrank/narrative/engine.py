"""Narrative engine orchestration: story extraction -> two-model annotation -> consensus -> series model.

Scheduling. Each distinct model gets its own queue, sized by the model's concurrency (1 on
free tiers, where parallel requests only collide with the quota; more on paid endpoints). An
episode's annotation tasks are submitted only when its story is finished (annotators receive that
episode's who-is-who context), so no worker ever blocks waiting on another task. That makes the
schedule deadlock-free for any worker count, including one model doing both jobs (D12).

Failure isolation. A failed story call degrades that episode's annotation (no context) but
does not stop it; a failed annotator leaves single-model evidence (lower confidence). Failed
calls are not cached, so re-running resumes exactly where quota or network stopped.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from ..budget import BudgetGuard
from ..cache import CallCache
from ..calls import CachedCaller
from ..llm import ModelSpec, estimate_tokens
from ..schemas import Episode
from .annotation import build_annotation_prompt, parse_annotations
from .consensus import build_consensus
from .extraction import build_story_prompt, parse_story, story_context
from .render import RenderedEpisode, render_episode
from .schemas import CallRecord, ConsensusEvent, EpisodeAnnotations, EpisodeStory, Issue
from .story import CharacterIndex, SeriesStory, build_series_story

log = logging.getLogger("hookrank.narrative")

STORY_MIN_OUTPUT = 1000
ANNOTATION_MIN_OUTPUT = 1500
MIN_WINDOW_EVENTS = 12
MAX_SPLITS = 2          # on a truncated answer, halve the event windows at most twice


@dataclass
class EpisodeNarrative:
    episode: Episode
    story: EpisodeStory | None = None
    annotations: list[EpisodeAnnotations] = field(default_factory=list)
    consensus: list[ConsensusEvent] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)


@dataclass
class NarrativeResult:
    series_id: str
    episodes: dict[str, EpisodeNarrative]
    series: SeriesStory
    index: CharacterIndex
    story_model: str
    annotator_models: list[str]

    def to_json(self) -> dict:
        return {
            "series_id": self.series_id,
            "story_model": self.story_model,
            "annotator_models": self.annotator_models,
            "series": self.series.model_dump(),
            "episodes": [{
                "episode_id": n.episode.episode_id,
                "episode_number": n.episode.episode_number,
                "story": n.story.model_dump() if n.story else None,
                "annotations": [a.model_dump() for a in n.annotations],
                "consensus": [c.model_dump() for c in n.consensus],
                "calls": [c.model_dump() for c in n.calls],
            } for n in self.episodes.values()],
        }


class NarrativeEngine:
    def __init__(self, story_spec: ModelSpec, annotator_specs: list[ModelSpec], cache: CallCache,
                 series_id: str, cache_only: bool = False, budget: BudgetGuard | None = None) -> None:
        if not annotator_specs:
            raise ValueError("At least one annotator model is required.")
        self.story_spec = story_spec
        self.annotator_specs = annotator_specs
        self.cache = cache
        self.series_id = series_id
        self.cache_only = cache_only
        self.budget = budget
        self._caller = CachedCaller(cache, series_id, cache_only=cache_only, budget=budget)
        self._lock = threading.Lock()

    # ---- model calls -----------------------------------------------------------------------

    def _call(self, stage: str, spec: ModelSpec, prompt: str, episode: Episode,
              min_output: int) -> tuple[str | None, CallRecord, bool]:
        """Return (raw text or None, call record, truncated?) via the shared cached caller."""
        return self._caller.call(stage, spec, prompt, episode.episode_id, episode.episode_number, min_output)

    # ---- stages ----------------------------------------------------------------------------

    def _story(self, rendered: RenderedEpisode, record: EpisodeNarrative) -> EpisodeStory:
        spec, episode = self.story_spec, rendered.episode
        prompt = build_story_prompt(rendered)
        if self._client_budget(spec, prompt) < STORY_MIN_OUTPUT:
            prompt = build_story_prompt(rendered, skip_camera=True)  # degraded but still useful
            log.info("%s story prompt too large; dropping camera lines", episode.episode_number)
        raw, call, _ = self._call("story", spec, prompt, episode, STORY_MIN_OUTPUT)
        with self._lock:
            record.calls.append(call)
        if raw is None:
            story = EpisodeStory(episode_id=episode.episode_id, episode_number=episode.episode_number,
                                 model=spec.model, ok=False)
            story.issues.append(Issue(stage="story", model=spec.model, kind="call_failed", detail=call.error))
            return story
        return parse_story(raw, rendered, spec.model)

    def _client_budget(self, spec: ModelSpec, prompt: str) -> int:
        return spec.request_token_limit - estimate_tokens(prompt) - 150

    def _annotate(self, rendered: RenderedEpisode, spec: ModelSpec, record: EpisodeNarrative,
                  story_future: Future) -> EpisodeAnnotations:
        episode = rendered.episode
        story = story_future.result()
        context = story_context(story)
        windows = self._windows(rendered, spec, context)
        for split in range(MAX_SPLITS + 1):
            result = EpisodeAnnotations(episode_id=episode.episode_id, model=spec.model, chunks=len(windows))
            truncated, any_ok = False, False
            for start, end in windows:
                prompt = build_annotation_prompt(rendered, context, start, end)
                raw, call, cut_off = self._call("annotation", spec, prompt, episode, ANNOTATION_MIN_OUTPUT)
                with self._lock:
                    record.calls.append(call)
                if cut_off and split < MAX_SPLITS:
                    truncated = True  # answer too long for the budget: retry with smaller windows
                    call.superseded = True
                    break
                if raw is None:
                    result.issues.append(Issue(stage="annotation", model=spec.model, kind="call_failed",
                                               detail=f"events {start}-{end}: {call.error}"))
                    continue
                any_ok = True
                parse_annotations(raw, rendered, spec.model, story, result)
            if not truncated:
                result.ok = result.ok and any_ok
                return result
            size = max(MIN_WINDOW_EVENTS, max(e - s for s, e in windows) // 2)
            log.info("%s annotation %s truncated; splitting into windows of %d events",
                     episode.episode_number, spec.short_name, size)
            windows = rendered.chunks(size)
        return result

    def _windows(self, rendered: RenderedEpisode, spec: ModelSpec, context: str) -> list[tuple[int, int]]:
        """Smallest number of event windows whose prompts leave enough output budget.
        Microdrama episodes normally fit in one; long or dense scripts are split."""
        size = len(rendered.lines)
        while True:
            windows = rendered.chunks(size)
            if all(self._client_budget(spec, build_annotation_prompt(rendered, context, s, e)) >= ANNOTATION_MIN_OUTPUT
                   for s, e in windows) or size <= MIN_WINDOW_EVENTS:
                return windows  # if still too large, the call fails with a clear RequestTooLarge
            size = max(MIN_WINDOW_EVENTS, size // 2)

    # ---- orchestration ---------------------------------------------------------------------

    @staticmethod
    def _after(story_future: Future, executor: ThreadPoolExecutor, fn, *args) -> Future:
        """A future for ``fn(*args, story_future)``, submitted to ``executor`` only once the story is done.

        Annotation tasks are never queued while their story is pending, so no worker ever sits blocked
        waiting for another task. That makes any number of workers per model deadlock-free, even when
        one model both extracts stories and annotates (the earlier design had to force such a model
        to a single worker).
        """
        out: Future = Future()

        def relay(inner: Future) -> None:
            exc = inner.exception()
            out.set_exception(exc) if exc is not None else out.set_result(inner.result())

        def start(done: Future) -> None:
            if done.exception() is not None:
                out.set_exception(done.exception())
                return
            try:
                executor.submit(fn, *args, done).add_done_callback(relay)
            except RuntimeError as exc:  # executor already shut down (only on an aborted run)
                out.set_exception(exc)

        story_future.add_done_callback(start)
        return out

    def run(self, episodes: list[Episode]) -> NarrativeResult:
        specs = [self.story_spec, *self.annotator_specs]
        # One queue per model, each allowed the model's own concurrency (free tiers 1, Nebius more).
        workers = {s.model: max(1, s.concurrency) for s in specs}
        executors = {m: ThreadPoolExecutor(max_workers=n, thread_name_prefix=m.split("/")[-1]) for m, n in workers.items()}
        records: dict[str, EpisodeNarrative] = {}
        pending: list[tuple[EpisodeNarrative, Future, list[Future]]] = []
        try:
            for episode in episodes:
                rendered = render_episode(episode)
                record = records[episode.episode_id] = EpisodeNarrative(episode=episode)
                story_future = executors[self.story_spec.model].submit(self._story, rendered, record)
                annotation_futures = [self._after(story_future, executors[s.model], self._annotate, rendered, s, record)
                                      for s in self.annotator_specs]
                pending.append((record, story_future, annotation_futures))
            for record, story_future, annotation_futures in pending:
                record.story = story_future.result()
                record.annotations = [f.result() for f in annotation_futures]
                usable = [a for a in record.annotations if a.ok and a.annotations]
                record.consensus = build_consensus(record.episode, usable)
        finally:
            for executor in executors.values():
                executor.shutdown(wait=True)
        stories = {eid: r.story for eid, r in records.items() if r.story}
        consensus = {eid: r.consensus for eid, r in records.items()}
        series, index = build_series_story(self.series_id, episodes, stories, consensus)
        return NarrativeResult(series_id=self.series_id, episodes=records, series=series, index=index,
                               story_model=self.story_spec.model,
                               annotator_models=[s.model for s in self.annotator_specs])
