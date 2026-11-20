EVENT ANNOTATION

You are annotating the events of ONE microdrama episode with narrative properties. Do NOT judge quality and do NOT rate anything: only state which properties each event has.

Properties (use only these names):
{{TAG_DEFINITIONS}}

Story context for this episode:
{{STORY_CONTEXT}}

For EVERY event that has at least one property, return an object:
{"id": the event ID exactly as listed, "quote": up to 12 words copied exactly from THAT event's text, "tags": [property names], "question": the unanswered question if "open_question" is tagged, otherwise "", "thread": a thread key from the story context or "", "prior_context": "none" | "some" | "heavy"}

prior_context means {{PRIOR_CONTEXT_HELP}}.

Rules:
- Annotate only events that clearly support one or more listed properties. Some episodes contain many qualifying events, others very few; do not target a fixed count. Prefer leaving an event untagged over tagging it on weak or speculative evidence.
- Story context identifies characters, relationships and ongoing storylines. It must not create properties the event's own dialogue, action, camera or sound description does not support. Do not infer an attack, confrontation, revelation or threat solely from earlier story knowledge.
- Do not label routine actions (walking, turning, reaching, handling objects, working, preparing things) as physical_action because a character is upset or a conflict exists elsewhere in the episode. A sound alone is not a visible physical action.
- Cut points matter as much as openings. Tag "open_question" when cutting immediately after the event leaves a specific, meaningful outcome unresolved: imminent danger with an unknown result, a revelation about to occur, a consequential decision awaiting an answer, a confrontation approaching decisive action, a meaningful threat, or an emotionally significant unanswered question.
- Do not tag "open_question" solely because a character asks an ordinary question, a conversation has not ended, the episode ends, music plays, the camera holds on a face, or the wider story has unresolved plotlines. The question must be about a concrete outcome, relationship, conflict, decision or danger; never a generic "what happens next?". Write it in 12 words or fewer, without inventing stakes. Some episodes have stronger resolution beats than cliffhangers.
- Strong moments are not only fights: betrayal, confession, humiliation, proposals, identity reveals, rejection, power reversals, vulnerability and discoveries all count when the text shows them.
- Action, camera, dialogue and sound descriptions are all valid evidence when explicitly present.
- The quote must come from the event whose ID you give.
- Return only {"events": [...]} as minified JSON: a single line, no indentation, no line breaks.

{{EPISODE}}
