CLIP CHECK

You are describing ONE candidate promo clip cut from a vertical microdrama. You see ONLY this clip, exactly as a first-time viewer would; you know nothing else about the show. Do not judge whether it is a good promo: describe what the clip contains, precisely and honestly.

Answer:
1. "opening": what a viewer sees or hears in the first 3 seconds (events starting before 00:03.00).
   - "type": exactly one of: attack_or_danger, confrontation_or_accusation, shocking_statement, revelation, emotional_breakdown, intimacy_or_romance, humiliation, proposal_or_decision, mystery_or_discovery, routine_activity, scene_setting, reply_to_unseen_question, other
   - "what_happens": one sentence; "evidence_event_ids": the opening events you describe
2. "central_question": the main question or outcome the clip builds toward, in at most 15 words, with "evidence_event_ids" of the events that raise it.
3. "answering_event_ids": IDs of every event INSIDE the clip that answers or settles that central question (an acceptance, refusal, confession, completed rescue, threat carried out or averted, decision announced). Use [] only if nothing in the clip settles it.
4. "ending": what is happening in the last 5 seconds.
   - "type": exactly one of: action_about_to_happen, question_or_demand_awaiting_answer, threat_hanging, revelation_pending, decision_pending, reaction_after_resolution, calm_or_closure, music_or_camera_only, other
   - "what_happens": one sentence; "evidence_event_ids": the final events you describe
5. "missing_context": things a new viewer would need to know to follow the immediate situation ([] if it is clear; a little mystery is fine).
6. "jump_problem": for clips with a [CUT], any lost setup or impossible transition at the jump; otherwise "none".

Use only event IDs listed in the clip. Return only one minified JSON object:
{"opening":{"type":"...","what_happens":"...","evidence_event_ids":["C01"]},"central_question":{"text":"...","evidence_event_ids":["C05"]},"answering_event_ids":[],"ending":{"type":"...","what_happens":"...","evidence_event_ids":["C12"]},"missing_context":[],"jump_problem":"none"}

CLIP (times are from the start of the promo; [CUT] marks a jump to a later moment):
{{CLIP}}
