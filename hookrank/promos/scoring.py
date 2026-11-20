"""Single source of truth for promo scoring: quality estimate, confidence and validity rules.

Everything here is a stated, *provisional* assumption (docs/decisions.md D5, D7): the
weights encode editorial reasoning from the brief, not values fitted to data, because no
labelled data exists (D2). The quality checker's sensitivity analysis (QC-D3) is where
their effect on rankings will be measured.

Three dimensions are kept separate on purpose:
  validity   - hard constraints; invalid promos are never ranked
  quality    - strength of narrative evidence (union of both annotators' tags)
  confidence - reliability of that evidence (agreement, repairs, diagnostics)
Model agreement is NOT an input to quality.
"""
from __future__ import annotations

import math

# ---- hard constraints (validity) ----------------------------------------------------------
MIN_DURATION_MS = 20_000
MAX_DURATION_MS = 60_000
HOOK_WINDOW_MS = 3_000          # brief: hook within the first 3 seconds
HOOK_LEAD_MAX_MS = 2_000        # hook event must start <=2s in, so >=1s of it lands in the window
CLIFF_WINDOW_MS = 6_000         # cliffhanger evidence must sit in the final 6 seconds
CLIFF_TAIL_MAX_MS = 1_500       # cut at most 1.5s after the unresolved moment ends
MIN_SEGMENT_MS = 4_000          # shorter fragments read as glitches, not cuts
MIN_SEGMENT_GAP_MS = 5_000      # closer segments should simply be one continuous clip
MAX_SEGMENTS = 2
CROSS_EPISODE_MAX_DISTANCE = 2  # join an episode only with the next one or two

# ---- quality: what each tag contributes as an opening hook or a closing cliffhanger --------
# physical_action lowered from 1.00 to 0.80 (D14/S8): tightening its definition did not reduce false
# positives (Qwen 191 -> 255 tags, 28 on sound effects), and physical movement is not automatically the
# strongest hook. It now ranks with humiliation, below a shocking line or betrayal.
HOOK_WEIGHTS = {
    "physical_action": 0.80, "shocking_line": 0.90, "betrayal": 0.90, "reveal": 0.85, "threat": 0.85,
    "humiliation": 0.80, "confrontation": 0.75, "reversal": 0.70, "striking_visual": 0.60,
    "new_conflict": 0.60, "emotional_shift": 0.55, "escalation": 0.50, "open_question": 0.50,
}
CLIFF_WEIGHTS = {
    "open_question": 1.00, "threat": 0.90, "new_conflict": 0.80, "escalation": 0.75, "betrayal": 0.75,
    "reversal": 0.60, "reveal": 0.60, "confrontation": 0.60, "shocking_line": 0.60,
    "physical_action": 0.55, "humiliation": 0.50, "emotional_shift": 0.45, "striking_visual": 0.40,
}
HOOK_MIN = 0.50                 # weakest opening that may anchor a candidate
CLIFF_MIN = 0.55                # weakest ending that may anchor a candidate
DEVELOPMENT_TAGS = {"reveal", "reversal", "escalation", "new_conflict", "betrayal"}
ENGAGEMENT_TAGS = {"confrontation", "physical_action", "humiliation", "emotional_shift", "shocking_line", "threat"}

QUALITY_WEIGHTS = {"hook": 0.30, "cliffhanger": 0.30, "standalone": 0.20, "progression": 0.12, "engagement": 0.08}
EXTRA_HOOK_TAG_BONUS = 0.05     # per additional distinct hook tag on the hook event itself, capped
EXTRA_HOOK_TAG_CAP = 0.15
LATE_HOOK_MS = 1_500            # strongest hook starting later than this is discounted
LATE_HOOK_FACTOR = 0.90
# Production cues (a "cliffhanger" stage direction, the episode's closing beat, music, a held close-up)
# may reinforce suspense but must not create it (D14): they are reported as notes, never scored.
RESOLUTION_AFTER_CLIFF_FACTOR = 0.40
SATURATION_SCALE = 3.0         # progression/engagement: 1 - exp(-count / scale); 3 events -> 0.63

# standalone comprehension proxy
PRIOR_CONTEXT_WEIGHT = {"none": 0.0, "some": 0.5, "heavy": 1.0}
CONTEXT_PENALTY_SCALE = 0.60
CROWD_FREE_CHARACTERS = 3       # characters beyond this many each cost a little clarity
CROWD_PENALTY = 0.07
UNSEEN_REFERENCE_PENALTY = 0.05 # a named character who never appears in the promo
UNSEEN_REFERENCE_CAP = 0.15
EXTRA_SEGMENT_PENALTY = 0.05
CROSS_EPISODE_PENALTY = 0.10

# ---- confidence -----------------------------------------------------------------------------
AGREEMENT_SCORE = {1: 0.5, 2: 1.0}
REPAIR_PENALTY = 0.10
CONTEXT_DISAGREEMENT_PENALTY = 0.05
DIAGNOSTIC_PENALTY = 0.10
WEAK_THREAD_LINK_PENALTY = 0.05
CONFIDENCE_LABELS = ((0.80, "high"), (0.55, "medium"), (0.0, "low"))

# ---- ranking diversity ---------------------------------------------------------------------
SAME_EPISODE_DECAY = 0.10       # each already-selected promo from the same episode
SAME_THREAD_DECAY = 0.04        # each already-selected promo on the same series thread (main
                                # storylines span many episodes, so this compounds quickly)
MAX_TIME_OVERLAP = 0.50         # overlap ratio above which a promo is a near-duplicate

WRITER_CUES = ("cliffhanger", "cliff-hanger", "cut to black", "smash cut", "freeze frame", "freezes")


def saturate(count: int) -> float:
    """Diminishing returns without a hard ceiling (a 3-event cap made every clip score 1.0)."""
    return round(1 - math.exp(-count / SATURATION_SCALE), 3)


def confidence_label(value: float) -> str:
    return next(label for threshold, label in CONFIDENCE_LABELS if value >= threshold)


def quality_total(components: dict[str, float]) -> float:
    return round(100 * sum(QUALITY_WEIGHTS[k] * components[k] for k in QUALITY_WEIGHTS), 1)
