# HookRank

HookRank reads timestamped microdrama screenplays and finds promo clips. A promo here is a 20–60 second moment, continuous or joined from two segments, that **opens on a hook** in the first 3 seconds, **cuts on a cliffhanger**, and **makes sense to a stranger**.

For each promo it gives:
- the episode, and the start and end of every segment;
- the hook and the cliffhanger;
- one line on why it works.

It also ranks the promos: which to post first, and why.

It is an editor's **shortlist tool**: it narrows "watch every episode" down to a ranked list with evidence. It does not post anything by itself. Design decisions, the evidence behind them and their trade-offs are recorded in [docs/decisions.md](docs/decisions.md).

## Architecture

```
screenplays ─► 1 Parse ─► 2 Story extraction ─► 3 Two-model annotation ─► 4 Verify & consensus ─► 5 Series story model ─► 6 Promo builder ─► 7 Ranking ─► outputs
               Python      1 LLM call/episode    2 LLM calls/episode       Python                 Python                  Python            Python
```

1. **Screenplay ingestion and parsing** (`hookrank/parsing.py`). Reads the studio template (episode headers, scenes, speaker and delivery labels, `Start/End` timings) into timed events. It keeps source line references and records anything unrecognised as a diagnostic, never silently dropping it. Input can be a `.txt` file, a folder or a ZIP.

2. **Narrative and story extraction** (`hookrank/narrative/extraction.py`). One call per episode builds that episode's story model: characters (with the personal name behind descriptive labels such as `LADY IN GOLD GOWN`), relationships, conflict threads, revelations and open questions.
   - Episodes are extracted independently, so calls are parallel and cached, and a changed episode re-runs only itself.
   - A claimed name is accepted only if it appears in that episode's own text.

3. **Two-model event annotation** (`hookrank/narrative/annotation.py`). Two different model families each label events with properties from a fixed vocabulary (`hookrank/narrative/vocab.py`), for example `shocking_line`, `physical_action`, `reveal`, `open_question`, `resolution`. They also say how much earlier-episode context each event needs.
   - Models state **facts about events, never scores or opinions about promos**.

4. **Quote verification and ID repair.** Every label must quote its event verbatim:
   - A quote that belongs to a neighbouring event moves the label to that event, and the repair is recorded.
   - A quote found nowhere drops the label, and the drop is recorded.

   **Consensus** (`consensus.py`) counts how many models support each label, on the event or on an overlapping one. Agreement is a **confidence signal, not a quality score**.

5. **Series story model and cross-episode linking** (`hookrank/narrative/story.py`).
   - Python merges per-episode extractions: `LADY IN GOLD GOWN` = `CLARISSA`; `LADY ASHCROFT` ≠ `LORD ASHCROFT`.
   - Merges are conflict-aware: one bad claim cannot fuse two differently named characters.
   - Episode threads link into series storylines only when they share at least two participants. A shared protagonist alone never links two plots.
   - Who appears in each event is computed from the speaker and the names mentioned in the text.

6. **Python-based promo construction** (`hookrank/promos/generate.py`). Candidates start at a hook event and are cut right after a cliffhanger event.
   - Hard rules: no cut inside a dialogue line; hook within the first 2s; cut within 1.5s of the unresolved moment; 20–60s in total.
   - A moment tagged as a `resolution` can never be the cliffhanger.
   - Two-segment promos join a hook to a distant cliffhanger, skip over a resolution, or bridge to one of the next two episodes on a shared storyline.

   Each promo gets **three separate scores** (weights in `hookrank/promos/scoring.py`):
   - **Validity**: pass/fail on every hard rule. Only valid promos are ranked.
   - **Quality estimate** (0–100): hook, cliffhanger, standalone clarity (context need, cast size, unseen names), story progression and engagement. Computed from label evidence only.
   - **Confidence** (0–1): did both models see a hook in the opening and an unresolved ending? Repaired IDs, context disagreements, parse diagnostics and cross-episode links lower it.

7. **Ranking and diversity** (`hookrank/promos/rank.py`).
   - Promos are ranked by quality, with a variety adjustment: each already-chosen promo from the same episode or storyline reduces a candidate's adjusted score.
   - Near-duplicates (same hook, same cliffhanger, or more than 50% time overlap) are skipped.
   - Confidence is shown beside quality and only breaks ties.

8. **Clip check** (`hookrank/promos/verify.py`). The shortlist and each episode's best promo are checked by a model that sees **only the clip**. It is never asked "is this good?" (verdict prompts accepted nearly everything, S7). It **classifies and extracts**: the type of the first-3s event, the type of the final beat, the central question, and any event inside the clip that settles it.
   - Python rejects routine or context-dependent openings, endings whose question is already answered, endings that are a reaction, closure or music only, and broken jumps.
   - Rejected candidates make room for alternative cuts. Missing-context notes are advisory.
   - Label rules that follow from event types are enforced in Python: physical actions and striking images must be visible; a shocking line must be spoken.
   - The report quotes the actual events, so a label is never shown without its evidence.

The model client (`hookrank/llm.py`) sends explicit request parameters and paces to provider rate limits read from response headers, including an output-token limit it learns from the first rejection. It rejects cut-off answers; the engine then re-annotates in smaller event windows. Every raw response is cached (`hookrank/cache.py`) under a key covering the model, parameters and full prompt.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add free keys: Groq as OPENAI_API_KEY, Gemini as GEMINI_API_KEY (or: python main.py configure)
python main.py doctor             # health-checks every model the pipeline will call
```

Default model roles (free tiers; see D9 in the decision log):
- `gemini:gemini-3.5-flash-lite` extracts the story model and annotates.
- `openai/gpt-oss-120b` on Groq annotates independently, from a different model family.

Models are addressed as `[endpoint:]model`:
- no prefix: the default OpenAI-compatible endpoint (`OPENAI_BASE_URL`, Groq by default);
- `gemini:` Google's OpenAI-compatible endpoint;
- `nebius:` Nebius Token Factory.

Set roles with `HOOKRANK_STORY_MODEL` and `HOOKRANK_ANNOTATOR_MODELS`.

**Faster paid option (D10, D11).** With `NEBIUS_API_KEY` set, the recommended annotators are `nebius:Qwen/Qwen3-235B-A22B-Instruct-2507,gemini:gemini-3.5-flash-lite`. Experiment S4 chose Qwen3-235B. Nebius models run 6 requests in parallel (`NEBIUS_CONCURRENCY`).
- Every paid call is checked against a **hard spending cap** (`HOOKRANK_BUDGET_USD`, default $20) before it is sent.
- Each call is recorded in `outputs/spend_ledger.sqlite3`, cumulatively across all runs.
- `python main.py spend` shows the total.

## Usage

```bash
python main.py run --input data --series-id broken-soldier --output-dir outputs/broken-soldier
python main.py run --input data --series-id broken-soldier --output-dir outputs/broken-soldier --episodes 1,2,3 --top 10
python main.py run --input scripts.zip --series-id new-show --output-dir outputs/new-show --cache-only
python main.py run ... --verify-top 80 --episode-tries 3   # clip-check budget (shortlist + alternatives; per-episode picks)
python main.py narrative --input data --series-id broken-soldier --output-dir outputs/broken-soldier  # engine only
python main.py configure          # endpoints, keys (hidden input), model roles
python main.py spend              # spend at paid endpoints against the budget cap
python main.py                    # interactive setup screen
pytest -q
```

Because every response is cached, re-runs are free. An interrupted run (quota, network) resumes where it stopped. An edited episode or prompt re-runs only the affected calls. `--cache-only` makes no API calls.

## Web app (Streamlit)

`streamlit_app.py` is a one-page front end: **upload → Analyze → review → download**.
1. Upload a ZIP of episodes or `.txt` screenplays; the app shows the detected episode count.
2. **Analyze** runs the existing pipeline through `hookrank/studio.py`, which calls the same `run()` as the CLI, on Nebius Token Factory, with live progress.
3. The top 10 promos appear in a table. Clicking a row shows the hook, cliffhanger, unanswered question, why it works, the timestamped excerpt with the hook and ending highlighted, the score breakdown and clip-check warnings.
4. **Download CSV / JSON** gives the real ranked results of that analysis.

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in NEBIUS_API_KEY (git-ignored)
streamlit run streamlit_app.py
```

**Deploy to Streamlit Community Cloud**
1. Push this repository to GitHub. `data/`, `outputs/`, `.env` and `.streamlit/secrets.toml` are git-ignored, so no screenplays or keys are published.
2. At [share.streamlit.io](https://share.streamlit.io): *Create app* → choose the repo and branch, main file `streamlit_app.py`. Under *Advanced settings*, pick Python 3.13 (the version the tests run on).
3. In *Advanced settings → Secrets*, paste the contents of `.streamlit/secrets.toml.example` with real values:
   `NEBIUS_API_KEY` should be a **separate key** on an account with a small prepaid balance and auto top-up off.
4. Deploy, try a small upload, then share the link. There is no sign-in: anyone with the link can use the app (D16).

**Protections:**
- The key is read only on the server, never displayed, logged or exported.
- The app is open to anyone with the link, so spend is bounded by the limits below and by the key's prepaid balance.
- Uploads are limited to 20 MB, 100 files, 2 MB per screenplay and 60 episodes. ZIP traversal, ZIP bombs and encrypted entries are rejected.
- Each analysis runs in its own temporary folder, deleted when it ends. Results live only in the user's session.
- One analysis runs at a time across the app.
- Each run has its own hard spending cap (`HOOKRANK_RUN_BUDGET_USD`, default $0.50) and a cost estimate is checked first. There is also a soft daily cap for the whole app (`HOOKRANK_APP_DAILY_BUDGET_USD`, default $3).

**Deployment limits:**
- Community Cloud apps **sleep after about 12 h without visitors**, so the first visit wakes the app.
- Resources are modest (about 1 GB RAM, as reported on the forum).
- Hosting is US-only.
- **The daily cap resets when the app restarts.** Nebius Token Factory documents no per-key spending cap, so the real backstop is a small prepaid balance with auto top-up off.
- One analysis at a time: a 50-episode upload took about 2 minutes ($0.18) in testing.
- Uploaded scripts are sent to Nebius for inference; the app says so.

## Outputs (in `--output-dir`)

| File | Contents |
|---|---|
| `promos_report.md` | Readable report: ranked promos (segments, hook, cliffhanger, why it works, why this rank, scores, caveats), best promo per episode, limitations |
| `promos_ranked.json` / `.csv` | The ranked "post first" list |
| `promos_by_episode.json` | Best valid promo for each episode |
| `promos_all.json` | Every evaluated candidate with its validity checks |
| `narrative.json` | Series story model, verified annotations, consensus, every model call |
| `episodes.json` / `failures.json` | Parsed episodes; parse diagnostics and failed calls |
| `run_manifest.json` | Run ID, input hashes, models and parameters, prompt hashes, scoring config, counts |
| `hookrank.sqlite3` | Raw model-response cache (what makes runs resumable) |

## Cost and throughput (free tiers, measured 2026-11-19)

- 3 calls per episode: Gemini story, Gemini annotation, gpt-oss annotation. Each is about 2.5–4k prompt tokens plus 0.5–2k output tokens.
- **Gemini flash-lite** answered 6 calls in 31s. It returns no rate-limit headers, so its free-tier quotas are not yet measured.
- **gpt-oss-120b on Groq** is the bottleneck.
  - 8,000 tokens/min allows about 1 call per 40s.
  - **A daily token limit also applies:** after a day of development runs it returned a 26-minute retry wait, and an earlier full run stalled at 11 episodes.
  - gpt-oss now makes 1 call per episode instead of 2, so roughly twice as many episodes fit per day.
  - Long waits now log the provider's message, so the limit can be measured.
- A 50-episode show may still need more than one day on free tiers. Re-running the same command continues from the cache.
- On paid tiers or a local model the pipeline runs without these waits, with no code changes.

## Limitations

- Timestamps are **script time**; the edited video may differ.
- Text cannot judge performance, visuals or sound. Hooks described only by camera directions are unverified on screen.
- Quality weights are provisional: no editor or audience labels exist to fit them.
- Model agreement shows consistency, not correctness.
- The clip check discriminates but makes mistakes. In a reviewed sample it over-rejected some endings as "resolved", judged near-identical cuts differently, and accepted a few quiet or incoherent clips (decision log: S8). Unchecked candidates in `promos_all.json` carry provisional label-only scores.
- `physical_action` labels remain noisy: models tag routine visible movement. Weight and event-type rules limit the damage, and the clip check keeps such openings out of the shortlist.
- Only the studio's screenplay template is supported; other templates parse to zero events and fail loudly.
- **The standalone quality checker is not built yet.** Its task list (independent validity re-checks, evidence integrity, structural proxies, stability, failure analysis, an optional bounded cold-viewer experiment) is in [docs/quality_checker_todo.md](docs/quality_checker_todo.md).

## Privacy

`data/`, `outputs/`, `.env` and SQLite files are git-ignored. A run sends **each episode's full screenplay text** to every configured provider (by default Google and Groq), and free tiers may retain or use that content. If scripts must not leave the machine, point `OPENAI_BASE_URL` at a local OpenAI-compatible server (Ollama exposes one at `/v1`). This is **not yet tested** with HookRank, and small local models may struggle with strict JSON.
