# HookRank — Decisions, Assumptions and Tradeoffs

A running log. Every entry states what was decided, the alternatives considered, why, and what evidence would make us revisit it. Entries are never silently edited; superseded entries are marked as such and linked to the replacement.

Status values: **Decided** · **Proposed** (awaiting agreement) · **Superseded**

---

## H0 — The original keyword-baseline prototype (historical; removed 2026-11-19, see D8)

The first prototype ranked clips with keyword lists and punctuation counts. It sent at most 10 clips to an optional "cold-viewer" LLM and planned to measure quality with human editor ratings. Recorded results (heuristic mode, 50/50 episodes, 3,727 timed events):
- 393 candidates; 74 removed as duplicates. **0 passed** the provisional gates (hook ≥ 3.5, cliffhanger ≥ 3.5, clarity ≥ 3.0), so its ranked export was empty. The best clip (Episode 46, `00:42.80–01:40.40`, 76.8/100) still failed the cliffhanger gate at 3.2.
- 382 of 393 candidates had no hook keyword at all in their opening (A1). The keyword signal was mostly punctuation and length.
- Clip boundaries could only be dialogue or action lines, so episode-ending camera and sound cues could never be cut points (A1).

**Observations still relevant to the current system:**
- **Overlapping-line cut.** An Episode 5 candidate `00:16.00–01:02.50` ended while an overlapping dialogue line ran to 64.60s. → The current builder's legal-cut rule (D7) makes such clips impossible to generate.
- **Scene-boundary overlap.** In Episode 22 a sound effect ends at 42.80s, after Scene 1's declared end of 42.50s; the parser reports it as a diagnostic. Audio legitimately crosses scene cuts, so this is a *diagnostic*, not an error. The current confidence score treats it as a small penalty (D7).

**Approaches abandoned, and why:** keyword scoring (no signal, genre-specific word lists); per-clip LLM scoring (does not fit free quotas, and the keyword prefilter decided what the model saw); a mandatory LLM cold-viewer judge (D5); human-rating metrics and a review dashboard (no labels will exist, D2).

## D1 — LLM access is limited to free APIs · Decided (2026-11-19)

**Decision.** Only free-tier or free-model APIs are used (e.g. Gemini free tier, Groq, OpenRouter free models, Hugging Face inference).

**Consequences.**
- Request volume, not compute, is the scaling bottleneck. The design must keep model calls proportional to *episodes*, not to *candidate clips*.
- Free tiers have changing rate limits, model availability and data-use terms. The pipeline must survive rate-limit errors (throttling, resumable runs, caching) and record exactly which model produced each output.
- **Privacy tradeoff:** free tiers' terms may allow the provider to use submitted content. Sending screenplays to a free hosted API is assumed acceptable for this assignment pack; for proprietary production use this must be re-checked or replaced with a local model.

**Revisit if:** a free quota proves too small for a full 50-episode series, or the scripts are declared confidential.

## D2 — No human labelling · Decided (2026-11-19)

**Decision.** No human-labelled gold set will be produced.

**Consequences.**
- The existing human-review metrics (Precision@10, pairwise agreement against editor ratings) will always report "unavailable". They cannot be the quality checker.
- Quality must be established from label-free evidence: hard validity checks, labels implied by the scripts' own structure, and independent-model judging. **Honest limit:** this demonstrates validity, consistency and agreement between independent judges, *not* proven editorial or audience performance. Reports must say so.

**Revisit if:** an editor or a public dataset of real promo picks becomes available.

## D3 — The LLM is a component, not the whole solution · Decided (2026-11-19)

**Decision.** Do not send whole scripts to an LLM and ask for promos. Python owns timing, cut-point validity, clip construction, constraint checking, ranking and verification. A model is used only for judgements that text rules cannot make (what is shocking, what is unresolved, what needs prior context), and its output is tied to parsed event IDs and verified.

**Why.**
- An LLM cannot be trusted with exact timestamps or the 20–60s and 3s/6s windows.
- Asking it for promos directly produces outputs we can't verify.
- Per-clip LLM calls do not fit free quotas.

---

## D4 — Groq free tier, two model families on one key · Decided (2026-11-19)

**Decision.** Use Groq's free tier through the existing OpenAI-compatible provider.
- **Primary labeller:** `openai/gpt-oss-120b`, the largest model available on the account (131k context).
- **Independent judge:** `qwen/qwen3.8-27b`, from a different model family (Alibaba).

Both health-checked OK on 2026-11-19, responding in under 1s.

> **Partly superseded by S1 change 1.** Both models now label every episode, and their consensus is used. "Primary" and "judge" no longer mean different jobs.

**Why.** One free key gives two unrelated model families, which the cross-model agreement check (P1 §5) needs. Gemini is not configured.

**Tradeoffs.**
- Groq's free-tier limits and model list can change without notice. Model IDs are configuration, never hard-coded in logic.
- The judge is smaller than the labeller (27B vs 120B). Where they disagree, that may reflect capability rather than the clip, and must be read with that in mind.

**Fix made along the way.** Groq (behind Cloudflare) rejected urllib's default `Python-urllib` user agent with HTTP 403 / code 1010. This was verified with curl: the same request returns 403 with that agent and 200 with a named one. `hookrank/providers.py` now always sends `User-Agent: hookrank/0.1`. This affects every hosted provider, not just Groq.

## A1 — Scalability audit of the existing prototype (2026-11-19)

Measured on the supplied pack (50 episodes): parsing takes 0.07s, and candidate generation takes 0.24s for 393 clips. Episodes are 70–201s long (median 118s), with 37–121 events (median 73) and about 2.6k–8k characters of content each. Compute is not a constraint; a whole episode fits comfortably in a single model request.

| Component | Scales to new shows weekly? | Evidence / reason |
|---|---|---|
| Screenplay parser | **Conditionally.** Assumes the studio's template | Handles this template with 0 failures. Assumes MM:SS times (no hours) and `**LABEL:**` blocks; a different template yields zero events. Assumed to be a stable house format. |
| Keyword hook/cliffhanger scoring | **No** | 382 of 393 candidates have *zero* hook keywords in their hook window, so the signal is mostly punctuation and length. The word lists (`duke`, `wife`, `gun`…) are genre-specific. |
| Candidate boundaries | **No** | Only DIALOGUE/ACTION events can start or end a clip, yet CAMERA is 47% of events (1,754/3,727), and every sampled episode *ends* on a CAMERA event. The episode's built-in cliffhanger can never be a cut point. |
| Single continuous clip only | **Incomplete** | The brief allows joining several moments, within one episode or across episodes. Not supported. |
| Per-candidate LLM scoring (2 calls/clip) | **No** on free quotas | A full series would need about 800 calls. The cap of 10 clips means 2.5% of candidates ever see a model, and the keyword prefilter decides which ones. |
| Story-aware LLM call | **No** (wasted) | Costs one call per clip and has no effect on any score. |
| Distinctiveness | **No** | Fixed at 3.0 for every clip. |
| Human-review evaluation | **No** under D2 | Needs labels that will not exist. |
| SQLite series memory + change invalidation | **Yes** (prototype scale) | Series isolation and content-hash invalidation are sound. Single-writer SQLite is fine for one machine. |
| Provider abstraction | **Partly** | The interface is sound. Missing: rate-limit throttling, resumable runs, and caching keyed to prompt content (today a prompt edit does not invalidate the cache). |
| Deduplication, validation checks, manifest | **Yes** | Generic, deterministic, show-independent. |

---

## P1 — Proposed architecture · Partly superseded (§4–5 replaced by D5; the rest adopted)

1. **Deterministic timeline (Python).** Parse events and compute *legal cut points*: moments where no dialogue line is in progress. Every event type, CAMERA included, can bound a clip.
2. **LLM event annotation (one call per episode, cached by content hash).** The model labels parsed event IDs with hook strength and type (slap, reveal, shocking line…), unresolved tension at that point, and context dependency (does a cold viewer need earlier information?). Output is validated against the parsed IDs; unknown IDs are rejected.
   - Calls per show = number of episodes (about 50).
   - Re-runs on unchanged episodes cost zero calls.
3. **Clip construction and ranking (Python).** Search single-segment and multi-segment promos (hook moment + build + cliffhanger, within or across episodes) under the hard constraints. Score from annotations plus structural features, enforce diversity, and rank globally.
4. **Blind cold-viewer check (bounded).** Only the top-K finalists per show go to a *different* model family, which sees just the clip and is asked to judge whether a stranger would understand it and want more.
5. **Label-free quality checker.**
   - **Hard checks** on every output: timestamps, windows, no cut mid-line, quoted evidence matches the source.
   - **Structural silver labels:** episode endings are cliffhangers by design. We measure whether the cliffhanger detector ranks them above random cut points. Loglines name each episode's key event, and we measure whether promos cover them.
   - **Agreement between two independent models**, with answer order swapped to control position bias.
   - **Stability across re-runs.**
   - **An automatically generated report** of where the tool picks badly.

### P1 risks and real-world assumptions (2026-11-19)

**Intended use.** An editor's *shortlist* tool (cutting "watch every episode" down to "review the top ~15 with reasons"). It is not an autopilot that posts promos. Nothing buildable without performance or editor data could justify autonomous posting.

| # | Risk / assumption | Severity | Mitigation in the prototype | What fixes it for real |
|---|---|---|---|---|
| R1 | Script timestamps equal timestamps in the final edited video | High | Stated assumption; outputs labelled "script time" | Align picks to the actual video (e.g. subtitles or speech recognition) |
| R2 | The LLM's "hook" and "cliffhanger" labels reflect editorial taste. In melodrama, absolute scores tend to bunch at "strong". | High | Ask for relative rankings within each episode instead of absolute scores; test before building (S1) | Editor picks and posted-promo performance as feedback |
| R3 | Label-free checks show validity and consistency, not goodness. Models agreeing can reflect shared bias. | High | Report wording limited to what is measured | Real editor or performance labels |
| R4 | Free API tiers: changing quotas, model retirement, data-use terms | Medium | Provider is a config setting; throttling, caching and resumable runs | A paid or local model in production (~50 calls per show) |
| R5 | One script template per studio | Medium | Fail loudly on zero parsed events; never output empty results silently | An adapter for each new template |
| R6 | Multi-segment and cross-episode promos multiply in number and can confuse cold viewers | Medium | Bound the search to nearby episodes or the same storyline; stricter standalone check | Editor feedback on multi-segment picks |

### S1 — Validation spike before building P1 · Done (2026-11-19) → **GO, with design changes**

**Setup.**
- Script: `experiments/s1_label_spike.py`. Raw outputs: `outputs/spikes/s1/` (git-ignored, contains screenplay text).
- Episodes 1, 13, 25, 37, 50. Each model ranks its top 6 hook events and top 6 cliffhanger events, by event ID, with a verbatim quote for each.
- Runs per episode: gpt-oss-120b twice (stability) and qwen3.8-27b once (agreement). Final run: 15/15 calls succeeded.
- **Leakage control:** 22 of 50 scripts contain the literal word "cliffhanger" in stage directions; it is removed from model input.

**Problems found on the way (all real, all fixed in the spike):**
1. Groq blocked urllib's default user agent (see D4).
2. gpt-oss-120b with Groq defaults failed 4 of 10 calls with HTTP 400 `json_validate_failed`, with empty output. In a fifth call it silently omitted the whole cliffhanger list. Cause: its hidden reasoning exhausted the output budget (about 3.9k reasoning tokens at medium effort). Fix: always set `reasoning_effort` and `max_completion_tokens` explicitly.
3. Groq free tier is **8,000 tokens/min and 1,000 requests/day per model** (from response headers). A daily token cap was not shown in the headers, so it is unknown. A single request over 8k is rejected outright (HTTP 413), so **medium reasoning effort cannot fit**: prompt 3.3k + reasoning about 5.7k. Low effort is used.

**Results (N = 5 episodes: small; treat as directional, not conclusive).**

| Measure | gpt-oss-120b (low) | qwen3.8-27b | Chance |
|---|---|---|---|
| Complete lists | 5/5 | 5/5 | — |
| Invented event IDs | 0/60 | 0/60 | — |
| Quote verbatim at cited ID | 50/60 | 60/60 | — |
| …of which the ID is wrong (quote belongs to a neighbouring event) | **5/60 (8%)** | 0/60 | — |
| Episode's final 10s among its cliffhanger picks | 5/5 (run A), 4/5 (run B) | 4/5 | mean 0.47 per episode |
| Top-1 cliffhanger in final 10s | 2/5 | 4/5 | — |

- **Overlap of top-6 sets.** Counts are matches out of 6 per episode; "tolerant" means within ±3s. The chance exact overlap is about 0.4–0.9 per episode.
  - Same model, two runs (stability): hooks exact `[1,5,5,6,4]`; cliffhangers exact `[2,4,1,5,2]`.
  - gpt-oss vs qwen (agreement): hooks tolerant `[4,3,4,3,5]`; cliffhangers tolerant `[1,3,2,3,3]`.
  - Top-1 is the same pick in only 2 of 5 episodes, for both stability and agreement.
- **Self-reported strength just restates rank.** Qwen's strengths were non-increasing with rank in 10 of 10 lists, with only 4 distinct patterns (e.g. `5,5,4,4,3,3`).
- **"Context needed" disagrees between models.** gpt-oss marked 31 of 60 picks `none`, qwen 20 of 60.

**What this shows.**
- Labels are mechanically trustworthy once IDs are verified.
- The models agree on *which moments* matter far above chance, more for hooks than for cliffhangers.
- The *order* within the top set is noisy. Self-scored strength carries no extra information.
- Ending detection beats chance, but positional knowledge ("episodes end on cliffhangers") may explain part of it.
- Most cliffhanger picks are **mid-episode**, which is what promos need, since each episode has only one ending.

**Design changes to P1 adopted from S1:**
1. **Use consensus votes, not self-reported scores.** Each episode is labelled by both model families (2 calls). A moment's strength is how many independent label sets include it, within ±3s. Rank order inside a list is a weak tie-breaker at most; the `strength` field is dropped.
2. **Quote-anchored ID repair (mandatory).** If a quote does not match its cited event, search neighbouring events (±2). If it is found there, move the ID to that event and record the repair. Otherwise drop the pick. Matching normalises Unicode punctuation and allows quotes trimmed from the source line.
3. **Both models run in parallel.** Each has its own quota, so two label sets per episode cost no extra time. Throughput: about 5k tokens per call at 8k tokens/min gives about 1.5 episodes/min per model, so **about 35 min for a 50-episode show**. 1,000 requests/day per model is not the binding limit.
4. **Runs must resume.** Results are cached per episode content hash. If a quota stops a run, rerunning continues where it stopped.
5. **"Context needed" is not trusted from one model.** The standalone check uses the blind cold-viewer step (P1 §4), not these labels.

**Still unproven (carried into R2/R3):** whether consensus moments match editorial taste. N = 5 is small; the full 50-episode run will re-measure these numbers.

---

## D5 — Revised architecture: evidence-based, no LLM judge · Decided (2026-11-19)

**Supersedes** P1 §4 (mandatory cold-viewer model check) and P1 §5 (two models judging finalists). The rest of P1 and the S1 design changes stand.

**Decision.** No LLM is ever asked "is this a good promo?". Models annotate *events* with narrative properties. Python builds, validates and ranks promos from that evidence. The pipeline has six stages:

1. **Ingestion and parsing** (Python): timestamps, dialogue, action, camera, sound.
2. **Narrative intelligence**: characters (with alias resolution), relationships, conflict threads and story progression. Each episode is extracted independently, and the series model is assembled deterministically in Python.
3. **Two-model event annotation** (gpt-oss + qwen): tags from a fixed vocabulary, each with a verbatim quote and an event ID. No numeric scores, since S1 showed self-scored strength just restates rank. Python verifies and repairs IDs, and records consensus.
4. **Promo generation and ranking** (Python): legal cut points, single-segment and two-segment candidates, hard constraints, and diversity.
5. **Evidence-based evaluation**, in three layers:
   - A: deterministic validity checks.
   - B: structured narrative evidence.
   - C: measurable structural proxies.
6. **Output**: ranked recommendations, each with evidence, three separate scores and stated limitations.

**Three separate dimensions per promo (never collapsed into one number):**

| Dimension | Question it answers | Source |
|---|---|---|
| **Validity** | Does it satisfy every hard constraint? | Deterministic checks. Only valid promos are ranked. |
| **Quality estimate** | How strong is the narrative evidence? | Annotation tags (union of both models) and structure. **Model agreement is not an input.** |
| **Confidence** | How reliable is that evidence? | Model agreement, ID repairs, single-source facts, parse diagnostics. |

Ranking is by quality estimate. Confidence is reported alongside and never used as a substitute for quality. Example: a promo whose hook only one model found, but which has a stronger reversal and open ending, can outrank one that both models agree is mediocre.

**Cold-viewer model evaluation** becomes an optional, bounded experiment in the quality-checker backlog (`docs/quality_checker_todo.md`). It is not part of ranking.

**Honest limit (unchanged).** Without editor ratings or audience data we cannot establish that the top promos are the most engaging. We can establish that they are valid, structurally compelling by stated criteria, consistent, and traceable to the screenplay.

**Build order.** Main features first (stages 1–4, then 6). The standalone quality checker (layer C reporting, stability and coverage analysis, failure report) is deferred and tracked in `docs/quality_checker_todo.md`. Layer A validity is built into generation, because invalid promos must never be produced.

## D6 — Narrative engine design choices · Decided (2026-11-19)

Each choice below is backed by a real run on episodes 1, 13 and 25 (dev cache `outputs/dev/`).

1. **Independent per-episode extraction, series model assembled in Python.**
   - Alternative: sequential extraction that feeds the growing series model into each call.
   - Chosen because calls run in parallel, are cached per episode, and a changed episode invalidates only its own calls.
   - Cost: cross-episode identity and thread linking are heuristic (see 4 and 5), not model-judged.

2. **Story model = gpt-oss-120b; annotators = gpt-oss-120b + qwen3.8-27b.**
   - Qwen was chosen first. On the original prompt it produced the only fully accurate summaries of the three models: gpt-oss-120b said arsenic was the *treatment*, gpt-oss-20b said Catherine *poisoned* Juliana.
   - Reversed after measurement. Groq enforces an undocumented **output-tokens-per-minute limit of 1,000 on qwen** (it appears only in 429 bodies, not in headers). With qwen doing story plus annotation, nearly every episode hit it, at a 65s wait each: **8m50s for 3 episodes**, projecting to about 2.5 h per show.
   - With the revised prompt, gpt-oss-120b's summaries were accurate on all three episodes.
   - Projected throughput with one call per model per episode: about 50 min per 50-episode show.
   - **Revisit if** a paid tier lifts the qwen limit; qwen remains the more careful extractor on this small sample.

3. **Minified JSON output.** Qwen's pretty-printed JSON was 30–40% whitespace. Prompting for minified output cut qwen output by about 40% (story 1,300 → 700 tokens; annotation 1,400 → 900), which matters directly under the output limit.

4. **Characters present in an event are computed in Python**: the speaker plus verified personal names mentioned in the text. They are never taken from a model's "characters involved" list, which was unverifiable and cost output tokens.

5. **Identity resolution** uses a conflict-aware union-find over speaker labels and verified names, with honorific stripping only onto names that already exist on their own. On the dev episodes it gives Clarissa = {CLARISSA, LADY IN GOLD GOWN}, Gerald = {GERALD, THE FORMER GROOM}, Juliana = {JULIANA, LADY JULIANA}, while LADY/LORD ASHCROFT stay separate by construction.

6. **Thread linking needs at least two shared participants** for multi-character threads. Observed failure before this rule: Juliana's personal "resolve" arc (Ep 1) was linked to the arsenic-poisoning plot (Ep 25), only because both involve the protagonist. One-person arcs link only to the same person's arc of the same kind.

7. **Quote-anchored ID repair stays mandatory**, even though the new compact rendering (short IDs, no timestamps) produced **0 repairs in 104 gpt-oss annotations** (two dev runs: 59 + 45), against 8% in S1. The safeguard is cheap, and the S1 failure mode is real.

8. **Writer cues stay in the model input.** Stage directions such as "cliffhanger chord" or "cut to black" are the screenplay author's own signals and legitimate evidence for the product. They were redacted only in S1, to test model ability. The quality checker's ending-detection test must redact them again (QC-B5).

9. **Truncated answers trigger smaller windows, not retries.** gpt-oss annotation output varies a lot: hidden reasoning tokens count against the output budget. Its answers were cut off on Episodes 3 and 6 of the full run, so about 30% of early episodes. The client rejects any `finish_reason=length` answer. The engine then halves the event windows (at most twice) and annotates each window separately. A truncated answer is never parsed. Only the fact that the prompt truncates is cached, so re-runs go straight to smaller windows. Calls recovered this way are reported as superseded, not failed.

## D7 — Promo construction and scoring · Decided (2026-11-19)

All logic is in Python (`hookrank/promos/`); all weights are in `hookrank/promos/scoring.py`. Every weight is a **provisional assumption**, not fitted to data (D2). Their effect on rankings is to be measured by QC-D3 (weight sensitivity).

**Construction.**
- A promo starts at the latest legal cut at most 2s before a hook-tagged event, so the hook lands inside the first 3s with at least 1s of it visible.
- It ends at the earliest legal cut at most 1.5s after a cliffhanger-tagged event.
- **A legal cut is any moment not inside a dialogue line.** Cutting mid-shot (camera) is allowed. Measured: 95% of event end-times are legal cuts, so the rule costs little coverage.
- Two-segment promos exist only to (a) join a strong hook to a cliffhanger more than 60s away in the same episode, (b) skip over a `resolution` that a continuous clip would carry, or (c) bridge to the next one or two episodes.
  - Within an episode, the two segments must share a storyline or a character. Across episodes, they must share a series storyline.
  - Segments are at least 4s long and at least 5s apart (closer segments should simply be one clip).
  - At most 2 segments: more would read as a montage, not a story moment.

**Hard rules added after observing failures on real output (episodes 1–7):**
1. *An event any model tagged `resolution` can never be a cliffhanger.* Before this rule, the #2 promo ended on "Juliana and the Bearded Man walk hand-in-hand… into the sunlight", which counted as a cliffhanger via `reversal`.
2. *The displayed hook and cliffhanger are the events the scores came from.* Before this, an explanation said "cuts on an unanswered question" while showing a different event, tagged `reveal`.

**Quality estimate.** Weights: 0.30 hook, 0.30 cliffhanger, 0.20 standalone, 0.12 progression, 0.08 engagement. Hook, cliffhanger and "makes sense on its own" carry 80%, because they are the brief's three criteria.
- Tag strengths are an ordinal editorial judgement. A physical clash or a shocking line is the strongest opening; an open question or a threat is the strongest ending.
- Progression and engagement use `1 - exp(-n/3)`. **Changed after observation:** a hard cap at 3 events gave nearly every 30s melodrama clip 1.0, and two promos scored 100/100.
- Standalone proxy: penalties for annotated events needing prior context, more than 3 characters, names mentioned but never shown, and extra segments or episode jumps.
- No duration preference: nothing in the brief or data says shorter is better within 20–60s.

**Confidence.** Agreement means: did each model mark *some* hook in the opening window, and *some* unresolved moment in the final window? **Changed after observation:** exact same-tag, same-event agreement understated agreement, because models label neighbouring events of one beat (the line vs the reaction shot). Penalties: repaired evidence IDs, models disagreeing on context need, parser diagnostics, heuristic cross-episode storyline links.

**Ranking.** By quality, then greedy variety selection:
- ×0.90 per already-chosen promo from the same episode, ×0.96 per already-chosen promo on the same storyline.
- Near-duplicates are skipped: same hook event, same cliffhanger event, or more than 50% time overlap.
- **Changed after observation:** a storyline decay of 0.92 compounded so fast (the main betrayal storyline spans many episodes) that a quality-97 promo fell below a quality-78 one.
- Confidence is a tie-breaker only.

**Outputs.**
- A ranked "post first" list.
- The best valid promo per episode (coverage, since the brief asks for promos "across all episodes").
- All evaluated candidates with their validity checks.
- A readable report and a manifest (run ID, models and parameters, prompt hashes, input hashes, scoring config snapshot).

## D8 — Legacy architecture removed · Decided (2026-11-19)

**Decision.** The narrative-aware pipeline (D5–D7) is the only active architecture. The original keyword-baseline prototype was deleted rather than kept alongside it, because two pipelines would double maintenance and confuse which outputs are current. Its findings are preserved in H0.

**Removed:**
- **Code:** the keyword pipeline and rubric (`pipeline.py`, `scoring.py`), clip validation and deduplication (`validation.py`, `ranking.py`), lexical memory retrieval (`retrieval.py`), human-rating metrics (`evaluation.py`), the SQLite `StoryMemory` (`memory.py`).
- **Provider-specific clients** (heuristic, Ollama-native, Gemini-native). The pipeline uses one OpenAI-compatible client.
- **Prompts:** `cold_viewer.md`, `story_discovery.md`, `episode_memory.md`.
- **The Streamlit review dashboard** (`app.py`) and the `streamlit` dependency.
- **Commands:** `evaluate` and `providers`.
- **Docs and tests:** the old walkthrough, solution and evaluation docs, and 14 tests that exercised only removed code.

**Kept and reused:**
- the parser (with `chronology_key` moved into it);
- `ProviderError`, `extract_json` and the user-agent fix (`providers.py`);
- the prompt renderer, the episode schemas, the console helpers;
- `configure`, rewritten to set the endpoint, key and model roles.

**Recovery.** None of the legacy code was ever committed. A tarball of every deleted or rewritten file is kept locally at `outputs/archive/legacy-architecture-2026-11-19.tar.gz` (git-ignored).

**Not changed.** Outputs and caches from earlier runs stay in `outputs/`, including the legacy run's files in `outputs/broken-soldier/` (`candidates.json`, `ranked_promos.*`, `evaluation.json`, …). They are stale but were not deleted.

## R4 update — free-tier daily token limit (observed 2026-11-19)

During the full 50-episode run, after a day of development calls, gpt-oss-120b answered HTTP 429 with a retry wait of about 26 minutes. That wait is far beyond the per-minute window, so it is consistent with a daily token limit. The run stalled at 11 of 50 episodes and was stopped. All completed calls stayed cached and will be resumed.
- The error text was not logged, so the exact limit is unknown.
- **Implication:** on the free tier a 50-episode show may take more than one day. The pipeline is built to resume, but throughput is quota-bound, not compute-bound.
- **Follow-up:** log the provider's 429 message on long waits, so the limit can be measured.

## S2 — Gemini flash-lite trial · Done (2026-11-19) → works; not yet adopted (decision pending)

**Setup.**
- Model: `gemini-3.5-flash-lite` through Gemini's OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai`), using HookRank's client unchanged.
- Gemini did both story extraction and annotation on episodes 1, 2 and 4, and its output was compared with the cached Groq results for the same episodes.
- Trial cache: `outputs/gemini-trial/` (git-ignored).

**Results.**
- **Speed.** 6 calls in 31s, about 10x faster than the Groq free tier, which needs about 1 minute per episode. One HTTP 503 was retried automatically. No rate-limit headers are returned, so its free-tier quotas are unknown.
- **Validity.** 38/38 kept quotes exact, 1 unsupported label dropped, 0 ID repairs, 0 truncations.
- **Story extraction.** Aliases are correct (LADY IN GOLD GOWN = Clarissa, THE FORMER GROOM = Gerald, LADY JULIANA = Juliana); the scarred warrior stays unnamed in Episode 4, which is correct for that episode. Summaries are accurate. 5 relationship items used wrong field names (`"c"` for `relation`) and were correctly rejected.
- **Hook annotation.** Its hook-class picks almost always coincide with another model's: 44 of 47 matched gpt-oss and 44 of 47 matched qwen (within ±1s).
- **Weakness: cliffhangers.**
  - Gemini annotates sparsely: 9–18 events per episode, against 13–25 for the Groq models.
  - It **never used `open_question`**, and in Episode 4 tagged no cliffhanger-class event at all. Used alone, it would find no promos there.

**Finding that affects the current pipeline, not only Gemini.** On the same episodes **qwen also tagged `open_question` 0 times**; gpt-oss tagged it 11 times. Cliffhanger evidence currently rests largely on one model, and cliffhanger confidence is low partly for this reason. A likely cause is the prompt: `open_question` also asks the model to write the question. This is unverified; testing it needs a prompt change and re-annotation.

**Options (not decided):**
- (a) Gemini lite as the **story model**. This halves gpt-oss's tokens per episode, roughly doubling the episodes per day possible under the gpt-oss daily limit (R4 update).
- (b) Gemini lite **replacing qwen** as the second annotator.
- (c) Revise the annotation prompt so cliffhanger properties are tagged by every model.

Both (a) and (b) need per-model endpoints: today all models share one `OPENAI_BASE_URL`. Both also send screenplay text to a second provider, Google, whose free tier may use submitted content (D1).

## S3 — Cliffhanger tagging prompt · Done (2026-11-19) → adopted

**Question.** S2 found that qwen and Gemini flash-lite almost never tag `open_question`, so cliffhanger evidence rested on gpt-oss alone. Is the prompt the cause?

**Setup.** `experiments/s3_cliffhanger_prompt.py` annotated episodes 1, 2 and 4 with gpt-oss-120b, qwen3.8-27b and gemini-3.5-flash-lite under the old and the revised prompt. The story model was the same in both runs (Gemini flash-lite), so story context was identical and only the annotation prompt differed. "Strong cliffhanger" means `open_question` (weight 1.0) or `threat` (0.9). A first metric that counted every tag at the anchor threshold (0.55) included about 90% of events and was discarded as uninformative. Raw results: `outputs/spikes/s3/`.

**Change.**
- `open_question` is redefined as: cutting right after this event would leave the viewer needing to know what happens next (a hanging question, or a pending threat, decision or reveal), not only a literal question.
- A new rule says cut points matter as much as openings, and that most episodes have several such moments, usually including the final beat.

**Results** (3 episodes, one run per prompt version):

| | gpt-oss | qwen | gemini-lite |
|---|---|---|---|
| `open_question` tags, before → after | 9 → 8 | 2 → 4 | **0 → 22** |
| strong-cliffhanger events | 19 → 14 | 5 → 7 | **2 → 24** |
| episode's final 10s tagged (of 3) | 0 → 2 | 0 → 1 | 0 → 1 |
| share of annotated events tagged `open_question` (after) | 16–27% | 0–9% | 17–48% |

- **Before the change, no model tagged any episode's built-in ending as a cliffhanger.**
- Every `open_question` after the change came with its question text.
- gpt-oss returned fewer events after the change (74 → 42) with no failures or truncations. With one run per version, that cannot be separated from run-to-run variance, so it is **not** attributed to the prompt.
- Over-tagging stayed bounded: at most 48% of an episode's annotated events.
- **Limit:** N = 3 episodes, one run each. This is directional evidence, not a measured effect size.

## D9 — Model roles: Gemini flash-lite for story and second annotator · Decided (2026-11-19)

**Decision.**
- Story model: `gemini:gemini-3.5-flash-lite`.
- Annotators: `openai/gpt-oss-120b` (Groq) + `gemini:gemini-3.5-flash-lite`. Qwen is no longer a default.
- **Supersedes** D6 item 2 (story model = gpt-oss-120b; annotators gpt-oss + qwen).

**Evidence.**
- S2: Gemini's story extraction was accurate, with correct aliases and summaries, and it ran about 10x faster than the Groq free tier.
- S3: after the prompt change, Gemini gave about 3x more strong-cliffhanger evidence than qwen (24 vs 7) on a similar number of annotated events (68 vs 70).

**Why it scales better.**
- gpt-oss now makes one call per episode instead of two. Groq's daily token limit (R4 update) therefore covers about twice as many episodes per day.
- Qwen's undocumented 1,000 output-tokens/min limit is gone from the critical path.
- **Implementation:** models are addressed as `[endpoint:]model`, each endpoint with its own key and request limits (`hookrank/config.py`). Gemini's story and annotation calls share one queue, in FIFO order: story(N) before annotate(N), so it cannot deadlock (D6 scheduling).

**Trade-offs.**
- **Confidence is somewhat lower:** gpt-oss and Gemini agree less than gpt-oss and qwen did. On episodes 1, 2 and 4: 40% of valid candidates had high confidence (was 50%), 10% low (was 5%). *Correction (same day):* an earlier version of this entry attributed part of the old agreement to both models rarely tagging cliffhangers. That was wrong. Confidence counts any cliffhanger-class tag (weight >= 0.55: shocking lines, reveals, confrontations…) in the final 6s, not `open_question` specifically. Qwen tagged many such events, so the old agreement came from that broad class. Because this agreement check is loose, the 50% -> 40% change says more about which broad tags each model pair shares than about which partner is better.
- **Privacy:** screenplays now also go to Google, whose free tier may use submitted content (D1).
- **Unknown quota:** Gemini returns no rate-limit headers, so its free-tier quotas are unknown until a full run. Transient HTTP 503s occur and are retried.
- **Fewer storylines:** Gemini extracts fewer threads than gpt-oss (Episode 2: 1 thread), which means fewer cross-episode links.

**Revisit if:** a full run shows Gemini daily quotas binding, or confidence collapsing on later episodes.

## S4 — Nebius annotator selection · Done (2026-11-19) → Qwen3-235B (see results below and D11)

**Purpose.** Replace Groq-hosted gpt-oss (rate-walled: about 1 call per 40s, plus a daily limit) with a paid Nebius Token Factory model, under a hard $20 total cap (D10).

**Candidates** (all non-Google, so the partner annotator stays independent of Gemini). The list price for a 50-episode show is shown in brackets:
- openai/gpt-oss-120b ($0.07)
- Qwen/Qwen3-235B-A22B-Instruct-2507 ($0.08)
- deepseek-ai/DeepSeek-V4-Pro-0813 ($0.53)
- zai-org/GLM-5.3 ($0.58)
- moonshotai/Kimi-K3.1-Pro ($1.10)

All passed a JSON smoke test, at a total cost of $0.0019.

**Protocol.**
- Episodes 1, 3 (the densest: 118 events), 25, 37 and 50.
- Story context: Gemini flash-lite, identical for every candidate. Partner: Gemini flash-lite annotations.
- Each candidate annotates every episode **twice** (runs A and B; B is forced past the cache), to measure stability.

**Selection rule, fixed before any result was seen:**
1. *Reliability gate*: at least 95% of calls succeed, 0 invented event IDs, at least 95% of quotes verbatim (after repair), no unrecovered truncation.
2. Among models that pass, rank by:
   - (a) cliffhanger coverage: episodes whose final 10s carry a strong cliffhanger tag, plus strong-cliffhanger events per episode, with `open_question` on at most 50% of annotated events (saturation guard);
   - (b) run-to-run stability (A vs B overlap of hook and strong-cliffhanger moments);
   - (c) agreement with Gemini above chance, a weaker signal because agreement is not quality (D5).
3. Cost and latency are tie-breakers only. Every candidate is far below the cap.

### S4 interim (2026-11-19)
- **Reasoning models eliminated by the reliability gate.** On Episode 1 their hidden reasoning used the full 8,000-token output budget:
  - Kimi-K3.1-Pro: truncated 3 times, $0.255;
  - GLM-5.3: truncated 3 times plus 2 HTTP 529 "overloaded", $0.116;
  - DeepSeek-V4-Pro: truncated 2 times, $0.071.

  The run was stopped to stop the spend. **Untested:** whether they would succeed with a far larger output budget or a reasoning-effort control. That was not pursued, because each attempt costs 10–40x more per call for a job that cheaper models complete.
- **Finalists:** gpt-oss-120b and Qwen3-235B. Qwen3-235B latency on Nebius was 10s to about 3 min per call (gpt-oss: about 4s), which is disqualifying given the goal (runtime).

## S5 — Screening of fast/low-cost Nebius models · Done (2026-11-19)

The user asked for an option with quality, speed and low cost. Each model got **one real Episode 1 annotation call** (the production prompt, with Gemini story context), with no splitting on truncation. Total cost: $0.034. Raw results: `outputs/spikes/s5_screen.json`.

| Model | Latency | Outcome |
|---|---|---|
| openai/gpt-oss-120b (benchmark) | 3.7s | valid; 12 events; 3 strong cliffhangers incl. the ending; $0.0010 |
| MiniMaxAI/MiniMax-M3.1-Flash | 7.4s | valid; 64 of 95 events annotated; `open_question` on 61% (fails the ≤ 50% saturation guard) |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | 29.7s | `open_question` on 32 of 32 annotations: label meaningless |
| nvidia/Nemotron-3_5-Lightning | 33.6s | truncated at 8,000 output tokens (reasoning) |
| zai-org/GLM-5.3-Flash | 48.9s | truncated |
| thinkingmachines/Inkling-2-Small | 46.0s | truncated |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | 62.8s | truncated |

**Reading.**
- Several "flash/lightning" models are reasoning models: under this budget they are slow and never finish.
- Of the alternatives, MiniMax is the only one that is fast and valid. But its over-tagging would flatten cliffhanger scoring, because when most events are "cliffhangers" the label no longer discriminates.
- **gpt-oss-120b remains the best balance of quality, speed and cost.** Its sparser labelling is complemented by Gemini's broader coverage.
- **Limit:** one call per model on one episode. This is a screen, not a benchmark.


### S4 final results (finalists; 5 episodes × 2 runs; `outputs/spikes/s4/report.json`; cost $0.045)

| Rule criterion | gpt-oss-120b | Qwen3-235B-A22B-Instruct-2507 |
|---|---|---|
| Reliability gate | pass: 10/10 calls, 0 truncations, 98.1% verbatim | pass: 10/10 calls, 0 truncations, 96.0% verbatim |
| (a) episode endings tagged | 4/5 | **5/5** |
| (a) strong-cliffhanger events per episode | 6, 9, 3, 5, 2 | **11, 14, 16, 14, 5** |
| (a) max `open_question` share | 31% | 31% |
| (b) A-vs-B stability, hooks / cliffhangers | 0.90 / 0.66 | **0.96 / 0.82** |
| (c) agreement with Gemini, hooks / cliffhangers (chance 0.35) | 0.83 / 0.55 | 0.73 / 0.58 |
| tie-breaker: mean latency | **5.2s** | 54.2s (observed range: 10s to about 3 min) |
| tie-breaker: measured cost, 10 calls | $0.0137 | $0.0154 |

**By the rule fixed in advance, Qwen3-235B wins** on (a) and (b), and is about level on (c). Latency was declared a tie-breaker only, and it was not used to overturn the result. Instead, latency is addressed by running Qwen requests in parallel (D11).

## D10 — Paid endpoint with a hard $20 cap · Decided (2026-11-19)

**Partly supersedes D1** (free APIs only). At the user's request, Nebius Token Factory (paid per token) may be used because free-tier rate limits made a full show take about a day. Free endpoints remain the code defaults; the paid choice lives in the user's `.env`.

**Guard (`hookrank/budget.py`):**
- Prices come from Nebius's own catalogue (`/models?verbose=true`). A model without a price is refused (fail closed).
- Each call reserves its worst case (estimated prompt plus full output allowance) before it is sent, and is refused if spent + in-flight + worst case would exceed the cap.
- Actual cost, from reported usage including reasoning tokens, goes to a ledger shared by all runs (`outputs/spend_ledger.sqlite3`), so the cap is cumulative.
- `python main.py spend` shows the ledger. `doctor` and experiment calls are counted too.
- Tested: refusal before sending, cumulative across processes, free endpoints never blocked.

**Spend to date** (after S4 and S5): **$0.50**, mostly reasoning models in the S4 interim ($0.44).

## D11 — Annotators: Nebius Qwen3-235B + Gemini flash-lite, with parallel requests · Decided (2026-11-19)

**Decision.**
- Story: Gemini flash-lite.
- Annotators: `nebius:Qwen/Qwen3-235B-A22B-Instruct-2507` + `gemini:gemini-3.5-flash-lite`.
- **Supersedes** D9's gpt-oss (Groq) annotator for this user's setup. Two families remain (Alibaba, Google).

**Why.** S4 (the advance rule) picked Qwen3-235B. S5 found no faster or cheaper model that is also valid and not over-tagging.

**Speed.** One request at a time per model was a free-tier design (quota-bound). Nebius publishes 400k–2M tokens/min per model, so models now carry a `concurrency` setting (Nebius default 6; free tiers 1).
- **Safety rule:** a model that also extracts stories stays at 1 worker. Its annotation tasks wait on story futures queued on the same executor, so more than 1 worker could fill every slot with waiting tasks and deadlock.
- Tested: parallel requests confirmed, and no deadlock with a shared story model.

**Trade-offs.**
- Qwen latency is erratic (10s to about 3 min).
- Gemini (free, sequential, 2 calls per episode) may now set the pace, and its quotas are still unmeasured.
- Screenplays go to Google and Nebius.

## D12 — All-Nebius roles; deadlock-free parallel scheduling · Decided (2026-11-19)

**Problem.** In the D11 setup Gemini flash-lite was the bottleneck: free-tier HTTP 503 "overloaded" responses, with one call taking over 3 minutes through retries; 2 calls per episode; and a single worker. It had to stay single because it did story extraction as well, and the old scheduler queued annotation tasks that blocked waiting for their story.

**Decision.**
- Story: `nebius:openai/gpt-oss-120b`.
- Annotators: `nebius:Qwen/Qwen3-235B-A22B-Instruct-2507` + `nebius:openai/gpt-oss-120b`. Two families remain (Alibaba, OpenAI).
- **Supersedes** D11's Gemini roles for this user's setup. Code defaults stay on free tiers.

**Why these models.**
- gpt-oss-120b is the only other candidate that passed S4/S5. It was fully reliable, about 5s per call, labelled selectively, and was the story model with accurate summaries in D6.
- MiniMax over-tagged, and the remaining fast models truncated.

**Scheduler change.**
- An episode's annotation tasks are now submitted only when its story finishes, via a completion callback (`NarrativeEngine._after`). No worker ever blocks on another task, so any worker count is deadlock-free, including one model doing both jobs.
- The D11 rule "a story model stays at 1 worker" is removed.
- Tested: a model doing both jobs with only 2 workers across 8 episodes completes, with parallel requests observed for both models.

**Known limitation.** Budget reservations are held per process. Two runs at once can each reserve independently, so the cap can be exceeded by at most each process's in-flight worst case (about $0.05 per call). Recorded spend in the shared ledger is always exact.

**Runs.**
- The D11 (Gemini) configuration run continues in `outputs/broken-soldier/`.
- The D12 run is in `outputs/broken-soldier-nebius/`.
- Both are kept for comparison.

**Results of both full runs (2026-11-19, 50 episodes each).**

| | D11 Gemini + Qwen | D12 all-Nebius |
|---|---|---|
| Wall time | 27 min 35 s (21 Gemini 503/429 retries) | **2 min 21 s** |
| Calls / failed | 150 / 0 | 150 / 0 |
| Episodes with a valid promo | 50/50 | 50/50 |
| Top-25 confidence | 13 high, 5 medium, 7 low | 20 high, 5 medium |
| Valid candidates with quality ≥ 95 | 15% | 17% |

- **Per-run spend bug found and fixed.** D11's manifest said $0.27, but it computed "this run" as a ledger-total delta while D12 ran at the same time, so it counted D12's spend too. Recomputed from each run's own calls × catalogue prices: **D11 $0.0746, D12 $0.1943** (D12's manifest said $0.1977, also inflated by the overlap). Ledger rows are now tagged with the run ID, and per-run spend sums only those rows (tested). The cap still uses the full ledger.
- **The rankings barely agree:** 1 identical promo in the two top-25 lists, 18 of 25 from the same episodes.
- **Hook over-tagging observed.** Qwen applied `physical_action` to ordinary movement ("turns abruptly", "takes a step forward", "lifts her chin"), annotating 41 events per episode against gpt-oss's 22. With quality computed from the union of tags (D5), this inflates hook scores and bunches quality at the top: 17% of valid candidates score ≥ 95. That makes the top of the ranking weakly discriminating, which is consistent with the low overlap above.
- **Proposed fix (pending user decision):** tighten the tag definitions, and fix the success criteria before re-running.

## D13 — Prompt precision revision and clip-level check · Decided (2026-11-19, user-specified)

The user specified these changes after reviewing the prompts and the D11/D12 outputs. **Principle:** an event can be dramatic without being a strong promo opening or ending. Keep event labels honest, and judge the final cut separately.

**1. `vocab.py`.** Four definitions are tightened; tag names are unchanged, so scoring code is unaffected:
- `open_question`: a specific, consequential unresolved outcome; excludes routine questions and generic curiosity.
- `physical_action`: a conspicuous action that creates or changes immediate stakes; excludes routine movements, touches and gestures.
- `confrontation`: a meaningful challenge or clash; excludes ordinary disagreement.
- `striking_visual`: a distinctive image conveying significant danger, surprise, emotion or curiosity; excludes ordinary close-ups and transitions.

All other tags are unchanged for this experiment.

**2. `narrative_annotate.md`.**
- (A) No target annotation count. Prefer leaving an event untagged to tagging it on weak evidence.
- (B) `open_question` needs meaningful stakes and a specific question. An episode's final event is not automatically one. No invented stakes.
- (C) Story context identifies characters and threads only. Every tag must be supported by the event itself.

**3. `narrative_story.md`.**
- Separate confirmed events from beliefs, accusations and lies. For example, an accusation of poisoning must not become a fact in series memory.
- Thread keys are episode-local and must be described consistently. The code already treats them as local (`thread_map` keys are `"<episode_id>:T<n>"`; tested).
- Facts stated only in the logline or continuity notes get an empty evidence list instead of an invented event ID.

**4. Clip check** (new; **amends D5**). For the top candidates only, one model sees **only the clip window** (no story memory, no logline). It answers three specific questions, each pass or fail with a reason and a supporting event ID that Python verifies is inside the clip:
- *opening*: do the first 3s grab attention and make sense without dialogue from before the clip?
- *ending*: is a concrete, consequential outcome unresolved at the cut?
- *standalone*: is the central conflict understandable from the clip alone?

**How D5 is respected.** The check never changes the quality score and is not a "good promo?" rating. It only demotes failed candidates below passing ones, and the reasons are reported. Unverifiable answers (bad JSON, an evidence ID outside the window) count as "unverified", not pass. Scoring weights are unchanged (`scoring.py`): the tightened definitions are tested first.

## S6 — Before/after test of D13 · Success criteria fixed before running (2026-11-19)

- **Setup.** Same input, models (all-Nebius D12), candidate generation and scoring weights. *Before* = the D12 run (`outputs/broken-soldier-nebius/`). *After* = D13 prompts, plus the clip check on the top 40.
- **Annotation-level criteria (prompt effect):**
  1. Qwen `physical_action` share of annotated events falls clearly below 15%.
  2. The share of valid candidates with quality ≥ 95 falls well below 17%.
  3. Episode-ending cliffhanger detection is not lost: endings still tagged in at least 40 of 50 episodes, counting either annotator.
- **Regression cases** (diagnostic, not a test set):
  - Ep 1 keeps a strong betrayal hook.
  - Ep 9 herbal preparation is not a powerful hook.
  - Ep 22 "Where is the West Wing?" is not a strong cliffhanger.
  - Ep 27 an unresolved poisoning revelation is distinguished from a resolved accusation.
  - Ep 32 the branding-iron danger is kept, without overrating an unclear opening.
- **Top-list criteria:** no top-10 hook is routine movement or a filler line; no top-10 cliffhanger is a routine logistical question.
- **Sample.** A random sample of other clips is reported for review, so that improving five known scenes cannot hide regressions elsewhere.

### S6 results (2026-11-19) → D13 kept; two problems remain

The run (`outputs/broken-soldier-nebius-d13/`): 50 episodes in 110s, 190 calls (150 narrative + 40 clip checks), 0 failed, $0.186. The *before* run is D12 (`outputs/broken-soldier-nebius/`).

| Fixed criterion | Before | After | Met? |
|---|---|---|---|
| 1. Qwen `physical_action` share clearly < 15% | 15% | 14% | **no** |
| 2. Valid candidates with quality ≥ 95 well below 17% | 17% | **3%** | yes |
| 3. Endings carry a strong cliffhanger in ≥ 40/50 episodes | 47 | 43 | yes |
| Top-10 hooks: no routine movement or filler | — | 3 of 10 routine (#1 raising a hand, #8 crushing herbs, #10 turning toward the pews) | **no** |
| Top-10 cliffhangers: no routine logistics | — | none observed | yes |

- **Annotation volume fell sharply.** Qwen went from 2,039 to 1,366 annotated events, gpt-oss from 1,105 to 624. Top-25 quality now spans 90–99 (it was 96–100), so the ranking discriminates.
- **Criterion 1, honestly.** Qwen's `physical_action` *count* fell 38% (about 306 → 191), but it dropped other weak tags proportionally, so the *share* stayed flat; gpt-oss's share rose from 16% to 24%. Random samples show precision is still poor: about 3/10 genuine for Qwen ("clutches his cane", "unlatching the door handle", "snips his hair" are tagged), about 5/10 for gpt-oss. **The models largely ignore the "exclude routine movements" clause.**
- **Regression cases.**
  - Ep 1: **regressed.** The best promo now opens on "Juliana turns abruptly toward the pews", a routine movement tagged `physical_action`.
  - Ep 9: fixed (the hook is now frantic pounding on the door).
  - Ep 22: mostly fixed (gpt-oss no longer tags "Where is the West Wing?"; Qwen still does; the best promo is different).
  - Ep 27: plausibly better (ends on unresolved danger).
  - Ep 32: clearly better (Catherine raising the branding poker → the glowing tip thrusting toward Juliana).
- **Random sample of 6 other episodes, read by the author (not an editor):** 2 strong (Ep 15, 24), 2 decent (Ep 11, 19 → 20 cross-episode), 1 weak hook (Ep 17, nailing a fence post), 1 mixed (Ep 50: exposition hook, strong reveal cliffhanger).
- **The clip check is too lenient.** gpt-oss passed 37/40, with **0 opening and 0 ending failures**, including #8 and #10 despite their routine openings. In its current form it does not catch the failure it was built for.

**Next options (not decided):**
- (a) Add concrete exclusions to the `physical_action` definition ("walking, turning, reaching, opening doors, holding or carrying objects, kneeling, placing hands").
- (b) If errors persist, recalibrate weights as the user's plan allows, for example full hook weight for `physical_action` only when the same event also carries a stakes tag (threat, confrontation, humiliation, betrayal).
- (c) Make the clip check stricter (for example, require it to state what a viewer knows by second 3), or use a different verifier model.

## D14 — Root-cause fixes for false "physical clash" hooks, resolved endings and a lenient clip check · Decided (2026-11-19, user-specified audit)

**Root causes,** traced stage by stage on the user's diagnostic cases (D13 run). Event-ID repair was involved in none of them.

| Symptom | Root cause | Where | Fix |
|---|---|---|---|
| Raised hand, herb crushing, fence-nailing, turning toward a father → "physical clash" | (1) Annotators tagged routine actions `physical_action` (gpt-oss in Ep 47, 21, 17, 1; Qwen on a **sound effect** in Ep 9) | model output | `physical_action` definition per the user, plus "a sound alone is not a visible physical action"; prompt rules against routine actions |
| Every `physical_action` described as a "clash" | (2) The report template mapped the label to the words "a physical clash" for any event | `PHRASES` in `promos/generate.py` | Neutral mechanism words; sounds become "a sound cue"; explanations quote the actual hook and cliffhanger text |
| Weak hooks scored 1.0 | (3) The hook score is the maximum label weight in the opening, so one `physical_action` from one model gives 1.0; the multi-tag bonus counted tags across different events | `evaluate()` | The bonus now counts only tags on the hook event. The label score is treated as *potential*: the clip check scales it when the opening fails (`hook_label` stays visible) |
| Ep 47 "resolved cliffhanger" scored 99 | (4) The ending check only looked for resolutions *after* the cliffhanger event started; "I accept, Your Grace." (128.5s) sat before the bowing reaction shot (131.5s) that carried `open_question` | `_validity` | Any resolution beat in the final 6s fails the check |
| Endings inflated | (5) +0.1 for the episode's closing beat and +0.1 for "cliffhanger"-type stage directions credited production cues as suspense | `evaluate()` / `scoring.py` | Removed; kept as notes ("production cue; not scored") |
| Clip check passed 37/40 with no opening or ending failures | (6) The verifier was asked for pass/fail verdicts; models answer agreeably | `promos/verify.py` | Classify and extract, Python decides (S7) |

**The clip check is redesigned** (`prompts/clip_check.md`, `promos/verify.py`):
- The verifier sees only the clip. It classifies the first-3s event type and the final-5s beat type from fixed lists, and extracts the central question plus **every event inside the clip that settles it**, the missing context, and any jump problem.
- Python decides:
  - **opening:** fails on routine activity, scene setting, or a reply to an unseen question, or without evidence in the first 3s;
  - **ending:** fails if anything settles the question, on a non-suspense final beat (a reaction after resolution, closure, or only music or camera), or on a missing or generic question;
  - **cut:** fails on a jump problem.
- Recommendation: reject if any of the three fails, otherwise accept. Failed checks scale the hook (×0.4) or cliffhanger (×0.4) component; the label values are kept.
- **Standalone is advisory.**
- **Selection runs in rounds:** rank, verify the unverified shortlist, drop rejects, re-rank. Alternative cuts of the same story beat can therefore enter, within a budget (`--verify-top`, default 80 checks).
- The report shows raw quality, labels-alone quality and the variety-adjusted value, the clip-check result, the unresolved question, evidence IDs, and a section of rejected candidates with their reasons.
- **Amends D5.** Model output now influences ranking through *extracted facts* (event types, settling events), never through a quality verdict. Scoring weights are otherwise unchanged.

## S7 — Clip-check calibration on the user's 12 diagnostic cuts · Done (2026-11-19)

`experiments/s7_verifier_calibration.py`, run on the baseline cuts; cost $0.037 over both rounds. The user's assessments were treated as hypotheses, not ground truth.

**Round 1: verdict prompt** ("does it pass?"). gpt-oss accepted 11 of 12 and Qwen 12 of 12. Both invented a new question for Ep 47 ("Will Catherine truly submit to servitude?") right after the clip shows "I accept, Your Grace.", although the prompt forbids it. **Verdict-style checking does not discriminate, whichever model is used.**

**Round 2: classify and extract.**
- gpt-oss: **all 6 strong cuts passed opening and ending.** Of the 4 clear weak cuts it failed 3: Ep 47 (ending classified "reaction after resolution", settled inside the clip), Ep 21 (opening classified "scene setting") and Ep 1 01:19 (ending classified "music or camera only"). It missed Ep 17. Ep 9 failed on a calm ending.
- Qwen also caught the weak cuts, but **claimed the question was answered in 2 strong cuts** (Ep 18, Ep 34).
- **gpt-oss is kept as the verifier.**
- Both models listed 3–5 "missing context" items for nearly every clip, strong ones included, so **standalone was made advisory.**

**Limit:** 12 hand-picked cases, one run each. This is calibration, not a measured accuracy.

### D14 follow-ups and S8 results (2026-11-19)

**S8: the annotation fix alone failed and was then backstopped.** With the tightened `physical_action` definition and rules, Qwen's `physical_action` tags *rose* from 191 to 255, and those on sound effects from 14 to 28. gpt-oss stayed at 149 → 148, and tagged a spoken line ("Put out your hands... You are under arrest"). The longer definition, with "dangerous intervention" and "consequential gesture", appears to have broadened Qwen's reading. Two model-independent fixes were added:
1. **Tag/event-type applicability is enforced in parsing** (`vocab.TAG_EVENT_TYPES`): `physical_action` and `striking_visual` require an ACTION or CAMERA event, and `shocking_line` requires DIALOGUE. Applicability is checked *after* any ID repair, so a repair never carries a strong tag onto an event it cannot describe (tested). It dropped 35 Qwen and 5 gpt-oss tags, re-parsed from cache at no cost.
2. **`physical_action` hook weight lowered from 1.00 to 0.80**, as the user's plan allowed once definition tightening failed.

Also added: **the best promo per episode is clip-checked too** (up to 3 candidates per episode).

**Final run** (`outputs/broken-soldier-nebius-d14/`; the interim first D14 run is kept in `-d14-interim/`):
- 50 episodes; 11,398 candidates (11,224 valid); 90 clip checks (45 accept, 44 reject, 1 unverified).
- Top 25 all verified; best-per-episode picks accepted in 44/50 episodes.
- Cost $0.196 across both D14 runs; total spend $1.18 of $20.
- Valid candidates with quality ≥ 95: 1.3% (17% in D12, 3.2% in D13). Top-25 range 91–98.
- "physical clash" in explanations: 0 (was 15 of the D13 top 25).
- 37 tests pass.

**Diagnostic cases.**
- The weak D13 cuts left the shortlist. Ep 47's resolved cut went from 99 (#1) to 89, and the episode's verified pick now ends *before* "I accept" (#8). Ep 21's herb cut went from 97 to 91, and the pick is "He shut the door in my face…" (#5). Ep 17 fell from 96 to 94, with a different cut picked. Ep 1's turn cut is no longer generated; its pick is "I changed my mind." → "And I am what?" (#3).
- Positives kept: Ep 32 branding iron (#4), Ep 43 accusation (#6, the same cut), Ep 3 proposal (#16), Ep 34 courtroom (#20). **Ep 18 lost:** no candidate was accepted within 3 tries.

**AI editorial review (by Claude, not a human editor; 31 clips read in full).** Verdicts: ACCEPT = could go to an editor; REVISE = promising material, weak cut; REJECT.

| | D13 top 10 (before) | D14 top 10 (after) |
|---|---|---|
| ACCEPT / REVISE / REJECT | 6 / 4 / 0 | **7 / 3 / 0** |
| Weak or routine hooks | 2 (herb prep, turning toward father) | **0** |
| False or weak cliffhangers | 3 (resolved Ep 47, music-only Ep 24 and Ep 1) | **1** (Ep 13: ends after the ring is found) |
| Remaining REVISE reasons | resolved ending, routine openings, music-only ending | resolved search (Ep 13), quiet opening plus a jump that skips the cure (Ep 21), talky scheming with low immediate stakes (Ep 48) |

- **Clip-check errors in the reviewed sample.**
  - Of 6 rejected candidates: 2 rejections agreed with (Ep 39, a haircut opening); 2 clearly wrong (Ep 11, a planted watch found, which is a twist rather than a resolution; Ep 44, whose identical "Her?!" ending was *accepted* in another cut, so the checker is inconsistent); 2 borderline (Ep 4, a proposal decision still pending).
  - Of 5 mid-ranked accepts, 2 should have been rejected: Ep 6 (quiet arrival, low stakes) and Ep 20 (an incoherent jump between unrelated scenes).
  - **Conclusion:** the checker now discriminates, but it over-classifies endings as "reaction after resolution", varies between near-identical cuts, and misses some weak cuts and jumps.
- **Metrics unavailable** (no human labels): editorial acceptance rate, Precision@10, ranking agreement with human review, annotation precision.
- **Measured instead** (AI review, 10 clips per list, directional only): top-10 acceptance 6 → 7; routine-hook rate 2 → 0; false-cliffhanger rate 3 → 1.

## D15 — Minimal Streamlit app for Community Cloud · Decided (2026-11-19, user-specified)

**Scope.** One page: upload → Analyze → top 10 → details → CSV/JSON. No sidebar, sliders or dashboards.
- `streamlit_app.py` contains UI only.
- `hookrank/studio.py` is a thin adapter that calls the **same `hookrank.run.run`** as the CLI (no second implementation). Fixed settings: top 10, clip-check budget 40, per-episode picks off (not shown in the app).
- Model roles default to the tested all-Nebius setup (D12/D14), overridable through secrets.

**Security and privacy decisions.**
- **Key handling:** read server-side from `st.secrets` or environment variables; never rendered, logged or exported. Tested: absent from served HTML and JS bundles, from rendered elements (AppTest) and from both exports. `client.showErrorDetails = "none"` keeps tracebacks out of the browser; unexpected errors show a generic message and log only the exception type.
- **Access:** Streamlit's built-in OIDC sign-in with an `ALLOWED_EMAILS` allowlist when configured. Otherwise a server-side password, compared in constant time, with a lockout after 5 failures. **No access control configured means analysis disabled** (fail closed). Checked before the uploader is even shown.
- **Spend:**
  - each run gets a **fresh ledger** with cap `HOOKRANK_RUN_BUDGET_USD`, so the cap is exact despite ephemeral hosting;
  - the cost estimate is checked before running;
  - `DailyBudget` is a soft app-wide daily cap that resets on restart;
  - one analysis at a time (process-wide semaphore), which also makes the log-based progress counter exact;
  - the provider-side spending limit is the stated real backstop.
- **Uploads:**
  - limits on size, file count, entry count, per-file size, extracted size, episodes and characters;
  - ZIP members are read in memory with a size cap (declared sizes are not trusted); absolute or `..` paths are rejected; encrypted entries and compression bombs are refused;
  - names are sanitized and de-duplicated.
- **Isolation:**
  - every analysis runs in its own temp directory (inputs, cache, outputs, ledger), removed in `finally`;
  - results live only in `st.session_state`;
  - no `st.cache_data` is used for user data, and `st.cache_resource` holds only the semaphore, daily counter and public prices;
  - the trade-off is no cross-run caching, so re-analysing the same upload costs again.
- **Inference only on click:** the pipeline runs only inside the Analyze branch, so selecting a promo is a plain rerun over stored results (tested with AppTest and a pipeline that fails if called).

**Verified** (real runs through the adapter, Nebius):

| Upload | Time | Cost | Result |
|---|---|---|---|
| Single TXT | 39s | $0.0095 | 4 promos, all clip-checked |
| 3-episode ZIP | 42s | $0.030 | 10 promos; 1 unchecked at budget 30, so raised to 40 |
| 50-episode pack | 111s | $0.183 | 10 promos, all clip-checked |

In every run the progress counts matched the real calls, no temp workspaces remained, and the key did not appear in exports. The local `streamlit run` served its health check. 49 tests pass, 12 of them new.

**Known limits:**
- Community Cloud sleeps after about 12 hours idle; resources are modest; hosting is US-only.
- The soft daily cap resets on restart.
- Clip-check variance between runs remains (S8): the Ep 39 haircut cut was rejected in D14 but accepted in this run.
- Not deployed yet: deployment needs the owner's GitHub repo and Streamlit account.

## D16 — Open access: password and sign-in removed · Decided (2026-11-19, user decision)

**Decision.** The deployed app has no sign-in. Anyone with the link can upload and run an analysis. This supersedes the *Access* bullet of D15. `authorized()`, `check_password`, `email_allowed`, their tests and the Authlib dependency were removed.

**Why.** The user wants reviewers to open the link and use the app with no friction; a shared password adds a step and gives only weak protection.

**What this changes.**
- The **key stays secret**: it is still server-side only and never rendered, logged or exported (tested).
- **Spend is now exposed** to anyone with the link. The remaining bounds:
  - a hard cap per analysis (`HOOKRANK_RUN_BUDGET_USD`, default $0.50) and a cost estimate checked first;
  - an app-wide daily cap (`HOOKRANK_APP_DAILY_BUDGET_USD`, default $3). It is held in memory, so it **resets when the app restarts** (sleep, reboot, redeploy), and a determined user could exceed it across restarts;
  - one analysis at a time across the app;
  - the upload and episode limits.
- Nebius Token Factory documents **no per-key spending cap**, and a balance can go negative when a card is attached. The real backstop is therefore operational: a separate key on an account with a small prepaid balance, auto top-up off, and the key rotated if abused.

**Alternatives kept in reserve.** If abuse appears: re-add a shared password (about 30 lines; it was removed, not redesigned), Streamlit OIDC sign-in, or a persistent daily cap in an external store (needs a database service).

**Verified.** The tests now include `test_app_is_open_without_sign_in_and_never_shows_the_key`: no password box, Analyze disabled until a valid upload, key absent from the page. 47 tests pass.

## D17 — Swappable model providers, configured in secrets only · Decided (2026-11-20, user request)

**Why.** The brief notes that stronger models may be connected through APIs the team provides, so swapping the model must be easy. The user also wants Nebius to stay the main provider, with other providers used only when their key is supplied, and only through secrets (never typed into the app).

**Decision.**
- **One table of providers** (`hookrank/config.py: PROVIDERS`): `nebius`, `anthropic`, `openai`, `gemini`, `groq`, `custom` (any OpenAI-compatible gateway).
  - Each provider is configured by variables named after it: `<NAME>_API_KEY`, `_BASE_URL`, `_CONCURRENCY`, `_REQUEST_TOKEN_LIMIT`, `_MAX_OUTPUT_TOKENS`.
  - Roles are `provider:model` references. Adding a provider means adding one row.
- **Two wire protocols, one client.**
  - OpenAI, Gemini, Nebius, Groq and gateways use Chat Completions over HTTP. Google provides an official OpenAI-compatible endpoint for Gemini.
  - Claude uses its native Messages API through the official Anthropic SDK, not a compatibility shim.
  - Both paths share the retry rules, pacing, truncation detection, caching and spending guard.
  - Claude requests send no temperature (current models reject sampling parameters) and get server-side refusal fallbacks on the models that support them.
  - A refusal is reported as a failed call.
- **Model-agnostic parameter handling.** A 400 that names a parameter we sent (`temperature`, `reasoning_effort`, `response_format`, `max_completion_tokens`) is retried once without it, instead of hard-coding per-model rules.
- **Nebius stays the default.** In the web app, a role whose provider has no key in secrets falls back to its Nebius default. A fallback never reduces two annotators to one, and it is logged. The page shows the models that will actually run.
- **Spending guard covers every provider.**
  - In the web app every provider is paid unless secrets list it in `HOOKRANK_FREE_ENDPOINTS`.
  - Price sources, lowest to highest precedence: built-in Claude list prices (as of 2026-11-03), provider catalogues (Nebius), then `HOOKRANK_PRICES`.
  - A paid model with no price is refused before the run starts, with a message naming the setting to add.
  - The cost estimate prices each role separately.
- **Breaking change for the command line.** `openai:` and unprefixed models now mean the real OpenAI API (it used to point at Groq), and the free-tier defaults now use the explicit `groq:` prefix (`GROQ_API_KEY`).
  - Older `.env` files that keep the Groq key in `OPENAI_API_KEY` with `OPENAI_BASE_URL` pointing at Groq still work, and that endpoint stays free.

**Verified.**
- 61 tests pass, 14 of them new: routing for each provider; Nebius default and fallback; Claude through a stubbed SDK (billing, effort, fallbacks, truncation, refusal, retry on 529); GPT retried without temperature; Gemini request shape; price parsing and refusal; the page shows the models in use and no key.
- Replaying the final 50-episode run from its cache with the new code gives the identical ranking (240/240 calls cached). The Nebius requests did not change.
- Live runs on Episode 1 through the app's own code:
  - default Nebius: $0.0105, 26 s, 4 promos;
  - story model swapped to Gemini in secrets: $0.0175, 54 s, 3 promos, same #1 promo;
  - Gemini without a price: refused before any call.

**Not verified live:** Claude and GPT calls, because no Anthropic or OpenAI key is available here. Their request paths are covered by tests with stubbed responses, and a single `python main.py doctor` with a real key would confirm them.

**Risks.**
- Stronger models are much more expensive: a 50-episode run on a frontier model can cost several dollars, so the per-run cap must be raised in secrets.
- Thinking models spend part of the output budget on reasoning. Truncated answers are already handled by splitting into smaller windows.
- Built-in Claude prices can go stale; `HOOKRANK_PRICES` overrides them.
- On a Claude refusal fallback, the call is priced at the configured model's rate.
- Quality on other models is unmeasured: S4–S8 calibrated prompts and checks on gpt-oss and Qwen.

## D18 — Any OpenAI-compatible API, added by configuration · Decided (2026-11-20, user request)

**Why.** The user asked for the app to work with "mostly every API possible". Most LLM APIs now accept the OpenAI Chat Completions format, so a fixed list of providers is unnecessary.

**Decision.**
- **More built-in providers.** OpenRouter, Together, Fireworks, DeepSeek, Mistral, xAI, Cerebras, Hugging Face and a local Ollama join the D17 set.
  - Each base URL was checked on 2026-11-20: `GET /models` answers 200, or 401/403 without a key.
  - Perplexity was left out because it has no `/models` endpoint; it can still be added by name.
- **Providers added by name.** Setting `<NAME>_BASE_URL` (plus `<NAME>_API_KEY`) makes `name:model` usable.
  - A prefix only counts as a provider if it is built in or declared this way, so model IDs containing a colon stay intact.
  - The web app reads every secret that matches the provider-variable pattern, not a fixed list.
- **Other ways of sending the key.** `<NAME>_AUTH_HEADER` sends the key in a named header (Azure OpenAI uses `api-key`) instead of `Authorization: Bearer`.
- **Local models.** A base URL on localhost needs no key and costs nothing.
- **Automatic prices from OpenRouter.** Its public `/models` catalogue publishes per-token prices in the same format as Nebius, so one OpenRouter key gives about 470 models priced automatically.
  - Entries with variable prices (negative values, such as `openrouter/auto`) are not trusted, so those models need a configured price.

**Verified.**
- 71 tests pass, 10 of them new: built-in routing, a provider added by name with a custom key header (checked on the actual HTTP request), colon model IDs, local keyless models, the OpenRouter catalogue, untrusted variable prices, and the app picking up a provider added by name from secrets without showing its key.
- Live: a provider defined only by `MYGATEWAY_BASE_URL`/`_API_KEY` (pointing at Nebius) answered a real call, billed at the configured price ($0.000026).
- Live: the OpenRouter catalogue returned 469 priced models.

**Not supported directly:** APIs that use neither the OpenAI nor the Claude format and have no OpenAI-compatible endpoint (for example, request signing with AWS SigV4). These need a small adapter, or a gateway such as LiteLLM in front of them.

