# Quality checker — task backlog

**Status:** deferred (D5). Main features are built first. This file is the single source of truth for what the quality checker must do; tick items as they land and link the commit or run that proves them.

**Ground rules (apply to every task):**
- Every number must come from a real run; record run ID, models, prompt hashes and input hashes.
- Label-free by design (D2): we measure validity, consistency, structural strength and agreement, **never** "accuracy" or "engagement".
- Every metric is reported against a chance or baseline comparison where one exists.
- Model agreement is a *confidence* signal, never proof of quality (D5).
- The checker never changes rankings; it reports.

---

## QC-0 Infrastructure
- [ ] `hookrank check --run <output-dir>` CLI entry; exits non-zero on any Layer A failure.
- [ ] `quality_report.json` and a human-readable `quality_report.md` per run.
- [ ] Run registry: every run writes `run_id`, timestamp, models, request parameters, prompt template hashes and input source hashes into the manifest; the checker refuses to compare runs whose inputs differ.
- [ ] Fixtures: 3–4 tiny synthetic screenplays with known answers (one obvious hook, one mid-line cut trap, one resolution-after-question trap, one cross-episode pair) for unit tests of the checker itself.

## QC-A Layer A — deterministic validity (hard fail)
Generation already enforces these; the checker re-verifies independently from the output files alone, so a generator bug cannot hide.
- [ ] A1 Duration: total of all segments is 20–60s; each segment is at least the minimum segment length.
- [ ] A2 Timestamp validity: every segment start/end equals a parsed event boundary of that episode, inside episode bounds.
- [ ] A3 Dialogue integrity: no segment boundary falls strictly inside a DIALOGUE event.
- [ ] A4 Hook timing: at least one hook-tagged event starts within the first 3s of the promo, and its quote matches that event's text.
- [ ] A5 Cliffhanger position: at least one cliffhanger-tagged event ends within the final 6s, and no `resolution`-tagged event follows it inside the promo.
- [ ] A6 Source integrity: every cited event ID exists; every quote is verbatim (after normalisation) in its cited event.
- [ ] A7 Segment order: segments are chronological and non-overlapping; cross-episode segments are in episode order.
- [ ] A8 Multi-segment coherence: segments share a conflict thread or at least one principal character.
- [ ] A9 Repetition: no two ranked promos share the same hook event *and* the same cliffhanger event; report time-overlap ratios above 0.5 within an episode.
- [ ] A10 Output schema: every promo has episode, segment timestamps, hook, cliffhanger, one-line reason, three scores and evidence IDs.

## QC-B Layer B — narrative evidence integrity
- [ ] B1 Annotation validity per model: complete responses, invalid or invented IDs, verbatim-quote rate, ID-repair rate (S1 baseline: gpt-oss 8% repairs, qwen 0%).
- [ ] B2 Tag distribution per model and per episode; flag tag saturation (e.g. more than 60% of annotated events carrying the same tag).
- [ ] B3 Inter-model agreement per tag (event-level, ±1s co-reference), with chance level computed from tag base rates (e.g. Cohen's kappa).
- [ ] B3a Confidence-signal looseness: confidence counts agreement on any cliffhanger-class tag (CLIFF_MIN 0.55 admits about 90% of tagged events, observed in S3). Measure how often 'agreement' is just both models tagging something near the end, and test a stricter version (strong cliffhanger tags only) before trusting confidence labels.
- [ ] B4 Story-model integrity: every alias merge has evidence; flag a character merged with more than one distinct name; flag thread links resting on a single shared participant.
- [ ] B5 Structural silver labels:
  - [ ] Episode-ending detection: share of episodes whose final 10s carry an `open_question`/cliffhanger tag, versus the chance rate. Run once **with** writer cues ("cliffhanger", "cut to black") redacted, to measure model ability without the leak.
  - [ ] Logline coverage: share of episodes where the logline's key event is covered by at least one annotated event (word-overlap or entity match), versus random events.
  - [ ] Writer-cue agreement: share of writer-marked cliffhanger cues that fall inside a model-tagged cliffhanger moment.

## QC-C Layer C — structural proxies, reported per promo and in aggregate
Proxies, not ground truth; the report must say so next to each.
- [ ] C1 Strong opening: hook-tag class and how soon it starts (ms from promo start).
- [ ] C2 Narrative progression: number of distinct development events (reveal, reversal, escalation, new conflict, betrayal).
- [ ] C3 Emotional engagement: presence of conflict, confrontation, reversal, emotional shift or humiliation.
- [ ] C4 Curiosity: open question at the end, not followed by a resolution inside the promo.
- [ ] C5 Standalone comprehension: principal characters in the promo who are introduced (speak, are named or are described) within the promo; share of events needing prior context.
- [ ] C6 Promo variety across the top N: distinct episodes, threads and principal characters; concentration index (e.g. HHI) versus the series' own distribution.

## QC-D Stability and robustness
- [ ] D1 Re-run stability: annotate the same episodes twice with the same model; overlap of top-N promos (Jaccard / rank-biased overlap) and of tags.
- [ ] D2 Model-swap stability: rankings using annotator A only, B only, and consensus; report how many top-10 promos survive each.
- [ ] D3 Weight sensitivity: perturb quality weights ±20% and report rank changes in the top 10 (ranking robustness to unvalidated weights).
- [ ] D4 Cut tolerance: shift every boundary to the nearest alternative legal cut point; report how many promos stay valid.

## QC-E Failure analysis ("where does the tool pick badly?")
Automated report answering the brief's last question, with concrete examples from the run.
- [ ] E1 Evidence-weak promos: high quality, low confidence (one model only, or repaired IDs).
- [ ] E2 Context-dependent promos that still ranked high (standalone proxy below threshold).
- [ ] E3 Episodes with no valid promo, and why (no hook in range, no legal cut, all candidates over 60s, …).
- [ ] E4 Hooks that are camera/visual-only (text cannot confirm their on-screen impact).
- [ ] E5 Concentration: one storyline or character dominating the top N.
- [ ] E6 Known systemic limits, always printed: script time ≠ final video time (R1), text cannot judge performance or visuals, weights unvalidated.

## QC-F Optional bounded cold-viewer experiment (not part of ranking)
- [ ] F1 Top 10–15 promos only; two independent open-source models; clip text only, no story context.
- [ ] F2 Record each evaluator's verdict and their disagreement; never use as ground truth or a ranking input.
- [ ] F3 Report rank correlation between evidence-based ranking and each evaluator; if they disagree substantially, report the uncertainty, not a score.
- [ ] F4 Position-bias control for any pairwise comparison (both orders).

## QC-G Documentation
- [ ] Add measured results and their limits to `docs/decisions.md` and `docs/evaluation.md`.
- [ ] Replace stale numbers in `docs/solution.md` with numbers from a checked run.
