"""Ranking: quality first, then variety. Confidence is reported, never used as a substitute.

Greedy selection: at each step take the valid promo with the highest *adjusted* quality,
where adjusted = quality x decay for every already-chosen promo from the same episode or on
the same series thread. Near-duplicates (same hook event, same cliffhanger event, or >50%
time overlap in an episode) are skipped outright. This answers "which would you post
first" with a list that is strong and does not repeat one scene ten times.
"""
from __future__ import annotations

from . import scoring as cfg
from .schemas import Promo


def _overlap(a: Promo, b: Promo) -> float:
    """Largest time-overlap ratio between any two same-episode segments (vs the shorter one)."""
    best = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode_id != sb.episode_id:
                continue
            inter = min(sa.end_ms, sb.end_ms) - max(sa.start_ms, sb.start_ms)
            if inter > 0:
                best = max(best, inter / min(sa.duration_ms, sb.duration_ms))
    return best


def _near_duplicate(p: Promo, chosen: list[Promo]) -> Promo | None:
    for q in chosen:
        if (p.hook.event_id == q.hook.event_id or p.cliffhanger.event_id == q.cliffhanger.event_id
                or _overlap(p, q) > cfg.MAX_TIME_OVERLAP):
            return q
    return None


def _adjusted(p: Promo, chosen: list[Promo]) -> tuple[float, int, int]:
    episodes = {s.episode_id for s in p.segments}
    same_ep = sum(1 for q in chosen if episodes & {s.episode_id for s in q.segments})
    same_thread = sum(1 for q in chosen if set(p.threads) & set(q.threads))
    factor = (1 - cfg.SAME_EPISODE_DECAY) ** same_ep * (1 - cfg.SAME_THREAD_DECAY) ** same_thread
    return p.scores.quality * factor, same_ep, same_thread


def rank_promos(promos: list[Promo], top_n: int = 25) -> list[Promo]:
    pool = sorted((p for p in promos if p.scores.valid), key=lambda p: (-p.scores.quality, -p.scores.confidence))
    chosen: list[Promo] = []
    while pool and len(chosen) < top_n:
        pool = [p for p in pool if _near_duplicate(p, chosen) is None]
        if not pool:
            break
        best = max(pool, key=lambda p: (_adjusted(p, chosen)[0], p.scores.confidence))
        adjusted, same_ep, same_thread = _adjusted(best, chosen)
        best.scores.variety_adjusted = round(adjusted, 1)
        best.rank = len(chosen) + 1
        best.rank_reason = _reason(best, adjusted, same_ep, same_thread)
        chosen.append(best)
        pool.remove(best)
    return chosen


def _reason(p: Promo, adjusted: float, same_ep: int, same_thread: int) -> str:
    comp = p.scores.components
    strongest = max(("hook", "cliffhanger", "standalone"), key=comp.__getitem__)
    weakest = min(("hook", "cliffhanger", "standalone"), key=comp.__getitem__)
    parts = [f"quality {p.scores.quality:.0f}/100, strongest on {strongest} ({comp[strongest]:.2f})"]
    if comp[weakest] < 0.6:
        parts.append(f"weakest on {weakest} ({comp[weakest]:.2f})")
    parts.append(f"{p.scores.confidence_label} confidence")
    if same_ep or same_thread:
        parts.append(f"variety-adjusted to {adjusted:.0f} (shares episode with {same_ep}, storyline with {same_thread} higher picks)")
    return "; ".join(parts) + "."
