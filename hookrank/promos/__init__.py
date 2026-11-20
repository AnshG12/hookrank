"""Promo construction, evidence-based scoring and variety-aware ranking (Python only)."""
from __future__ import annotations

from ..parsing import chronology_key
from ..narrative import NarrativeResult
from .generate import EpisodeContext, evaluate, generate, summarize_rejections
from .rank import rank_promos
from .schemas import Promo


def _episode_rank(episode, position: int) -> int:
    number = chronology_key(episode)[0]
    return number if number < 10**9 else position


def build_promos(result: NarrativeResult, top_n: int = 25) -> tuple[list[Promo], list[Promo], dict]:
    """Return (ranked promos, all evaluated candidates, stats) from a narrative result."""
    ordered = sorted((n.episode for n in result.episodes.values()), key=chronology_key)
    # rank = the episode's number in the series (not its position in this run), so cross-episode
    # distance stays correct when only a subset of episodes is processed.
    contexts = [EpisodeContext(episode=ep, evidence={c.event_id: c for c in result.episodes[ep.episode_id].consensus},
                               index=result.index, rank=_episode_rank(ep, i)) for i, ep in enumerate(ordered)]
    unique: dict[str, Promo] = {}
    for candidate in generate(contexts):
        promo = evaluate(candidate)
        kept = unique.get(promo.promo_id)
        if kept is None or promo.scores.quality > kept.scores.quality:
            unique[promo.promo_id] = promo
    evaluated = list(unique.values())
    ranked = rank_promos(evaluated, top_n=top_n)
    valid = [p for p in evaluated if p.scores.valid]
    stats = {
        "candidates_evaluated": len(evaluated),
        "valid_candidates": len(valid),
        "invalid_by_check": summarize_rejections(evaluated),
        "valid_by_kind": {k: sum(p.kind == k for p in valid) for k in ("single", "two_segment", "cross_episode")},
        "episodes_with_valid_promo": len({s.episode_id for p in valid for s in p.segments}),
        "episodes_total": len(ordered),
        "ranked": len(ranked),
    }
    return ranked, evaluated, stats


__all__ = ["Promo", "build_promos", "rank_promos"]
