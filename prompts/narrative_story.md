NARRATIVE EXTRACTION

You are building a story model of ONE episode of a vertical microdrama, so promo editors can tell who is who and which conflicts are in play. Use ONLY this episode's text below. Never use knowledge of other episodes.

Return ONE minified JSON object (a single line: no indentation, no line breaks) with exactly these keys:
- "summary": 2-3 sentences on what happens in this episode.
- "characters": one entry for EVERY speaker label, plus important named characters who do not speak. Each entry:
  {"label": the speaker label exactly as listed (for a non-speaker, the name as written), "name": the character's personal name ONLY if this episode's text states it (logline, notes, dialogue or action) - otherwise repeat the label, "description": who they are, at most 10 words, "evidence": [event IDs that establish the name or role]}
- "relationships": [{"a": name, "b": name, "relation": e.g. "sisters", "fiance", "mother and son", "change": how it changes in this episode or "", "evidence": [event IDs]}]
- "threads": the conflict or story threads in play, at most 4 (merge overlapping ones):
  [{"key": "T1", "summary": at most 15 words, "kind": one of {{THREAD_KINDS}}, "participants": [names], "status": one of {{THREAD_STATUSES}}, "evidence": [event IDs]}]
- "revelations": facts revealed in this episode, at most 3: [{"text": at most 15 words, "evidence": [event IDs]}]
- "open_questions": questions left unanswered at the end of this episode, at most 3, each at most 15 words.

Rules:
- Copy event IDs exactly as listed (e.g. "E014"). Every claim needs at least one event ID, except facts stated only in the LOGLINE or CONTINUITY NOTES: give those an empty evidence list [] instead of inventing an event ID.
- Spell names exactly as the text does. Descriptive labels (e.g. "LADY IN GOLD GOWN") get a personal name only when this episode's text links them to it; if unsure, keep the label.
- Distinguish confirmed events from characters' beliefs, suspicions, lies, and claims. A character making an accusation does not mean the accusation is true.
- Story thread keys T1-T4 are local to this episode. Describe each thread consistently using the involved characters, conflict, and current status so that an external system can link related threads across episodes.
- Return JSON only.

{{EPISODE}}
