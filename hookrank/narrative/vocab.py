"""Fixed annotation vocabulary: the only labels a model may attach to an event.

Single source of truth for both the prompt (definitions are rendered from here) and the
Python that consumes the labels. Models describe *what an event does*, never how good it
is: S1 showed self-scored strength merely restates rank (docs/decisions.md D5).
"""
from __future__ import annotations

# Narrative properties (D5, Layer B): what the event does to the story.
NARRATIVE_TAGS: dict[str, str] = {
    "betrayal": "reveals or enacts a betrayal of trust",
    "new_conflict": "introduces a conflict that was not present before",
    "escalation": "raises the stakes of a conflict already under way",
    "reversal": "changes a character's situation, status or power position",
    "reveal": "reveals information the audience did not know before this event",
    "open_question": ("cutting immediately after this event leaves a specific, consequential outcome unresolved, "
                      "such as imminent danger, an important answer, a decision, or a pending revelation; exclude "
                      "routine questions, low-stakes uncertainty, and generic curiosity about what happens next"),
    "resolution": "answers an earlier question or releases built-up tension",
}

# Attention properties: what could make a scrolling stranger stop.
ATTENTION_TAGS: dict[str, str] = {
    "physical_action": ("a conspicuous physical action that creates, reveals, or materially changes immediate "
                        "dramatic stakes, such as an attack, forceful restraint, sudden collapse, dangerous "
                        "intervention, or consequential physical gesture; exclude routine movement, object handling, "
                        "ordinary touches, and gestures whose dramatic significance is not evident; a sound alone "
                        "is not a visible physical action"),
    "shocking_line": "a line that shocks, wounds, accuses or insults",
    "threat": "a threat, ultimatum or imminent danger",
    "confrontation": ("a direct, meaningful challenge or clash between characters, such as an accusation, "
                      "defiance, or challenge to authority; exclude ordinary disagreement or conversation between rivals"),
    "humiliation": "public shaming or degradation of a character",
    "emotional_shift": "a sharp emotional change: tears, rage, despair, sudden joy",
    "striking_visual": ("a visually distinctive, explicitly described image or reveal that conveys significant "
                        "danger, surprise, emotion, or curiosity; an ordinary close-up or scene transition does not "
                        "qualify by itself"),
}

TAGS: dict[str, str] = {**NARRATIVE_TAGS, **ATTENTION_TAGS}

# Tags whose definition restricts the kind of event they can describe (enforced in annotation parsing,
# D14/S8): a physical action or a striking image must be visible, and a shocking *line* must be spoken.
# Observed before enforcement: physical_action on sound effects ("echoing thud") and on dialogue.
TAG_EVENT_TYPES: dict[str, set[str]] = {
    "physical_action": {"ACTION", "CAMERA"},
    "striking_visual": {"ACTION", "CAMERA"},
    "shocking_line": {"DIALOGUE"},
}

PRIOR_CONTEXT_LEVELS = ("none", "some", "heavy")
PRIOR_CONTEXT_HELP = ("how much a first-time viewer needs EARLIER episodes to understand this event: "
                      "none = clear from this episode alone; some = partly relies on earlier events; "
                      "heavy = meaningless without earlier episodes")

THREAD_KINDS = ("betrayal", "rivalry", "romance", "revenge", "secret", "power", "survival", "family", "justice", "other")
THREAD_STATUSES = ("introduced", "escalated", "ongoing", "resolved")


def render_tag_definitions() -> str:
    lines = ["Narrative properties:"]
    lines += [f"- {tag}: {text}" for tag, text in NARRATIVE_TAGS.items()]
    lines.append("Attention properties:")
    lines += [f"- {tag}: {text}" for tag, text in ATTENTION_TAGS.items()]
    return "\n".join(lines)
