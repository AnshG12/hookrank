# HookRank architecture

> **The one idea:** AI reads the story, code makes the promos.
> Models only *describe* what happens in each scene. Python decides every timestamp, applies every rule and ranks the promos, so every answer traces back to a line in the script.

**Contents**
1. [The big picture](#1-the-big-picture)
2. [The pipeline, stage by stage](#2-the-pipeline-stage-by-stage)
3. [One episode through the narrative engine](#3-one-episode-through-the-narrative-engine)
4. [How a promo is cut](#4-how-a-promo-is-cut)
5. [Three separate scores](#5-three-separate-scores)
6. [The clip check and selection rounds](#6-the-clip-check-and-selection-rounds)
7. [Swappable models](#7-swappable-models)
8. [The web app and its safety boundary](#8-the-web-app-and-its-safety-boundary)
9. [Scale and cost](#9-scale-and-cost)
10. [Code map](#10-code-map)

---

## 1. The big picture

```mermaid
flowchart LR
    S[/"🎬 Screenplays<br/>.txt or .zip"/]:::data
    P["Parse<br/>timed events"]:::code
    U["Understand<br/>story and scene labels"]:::ai
    V["Verify<br/>quotes and IDs"]:::code
    B["Build and score<br/>every legal cut"]:::code
    C["Clip check<br/>classify the clip"]:::ai
    R["Decide and rank"]:::code
    O[/"📋 Ranked shortlist<br/>with evidence"/]:::data
    H(["👤 Editor picks<br/>and cuts the video"]):::human

    S --> P --> U --> V --> B --> C --> R --> O --> H

    classDef ai fill:#ede9fe,stroke:#6d28d9,color:#2e1065,stroke-width:2px
    classDef code fill:#e0f2fe,stroke:#0369a1,color:#082f49,stroke-width:2px
    classDef data fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef human fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:2px
```

| Colour | Who | Does what | Why |
|---|---|---|---|
| 🟣 Purple | **AI** | Judgements text rules can't make: who is who, what is shocking, what is still unresolved | That's the part that needs language understanding |
| 🔵 Blue | **Code** | Timing, cut points, the 3 s and 20–60 s rules, validity, scoring, ranking | A model can't be trusted with exact timestamps, and the rules must hold every time |
| 🟢 Green | **Human** | Picks from the shortlist and cuts the video | No editor or audience data exists yet, so the tool never posts by itself |

**What the models are never asked:** "Is this a good promo?" When we tried that kind of question, models answered "yes" to 23 of 24 clips (experiment S7), so it carries no information.

---

## 2. The pipeline, stage by stage

```mermaid
flowchart TD
    IN[/"Screenplays"/]:::data

    subgraph ING["1 · Ingest (code)"]
        P1["Parse template<br/>scenes · speakers · Start/End times"]:::code
        P2["Legal cut points<br/>any moment not inside a spoken line"]:::code
    end

    subgraph NAR["2 · Narrative engine (3 AI calls per episode)"]
        N1["Story model<br/>characters · aliases · storylines"]:::ai
        N2["Annotator A · Qwen3-235B<br/>tags events from a fixed list"]:::ai
        N3["Annotator B · gpt-oss-120b<br/>same task, different model family"]:::ai
        N4["Verify every tag<br/>quote must match the event word for word"]:::code
        N5["Series story model<br/>merge aliases · link storylines"]:::code
    end

    subgraph PRO["3 · Promo engine (code)"]
        G1["Generate candidates<br/>single · two-segment · cross-episode"]:::code
        G2["Validity gate<br/>hard rules"]:::code
        G3["Score<br/>quality and confidence"]:::code
        G4["Rank with variety"]:::code
    end

    subgraph CHK["4 · Clip check (AI classifies, code decides)"]
        K1["Model sees only the clip"]:::ai
        K2["Python rules accept or reject"]:::code
    end

    OUT[/"Top promos · best per episode<br/>report · CSV · JSON · manifest"/]:::data

    IN --> P1 --> P2 --> N1
    N1 --> N2 & N3
    N2 & N3 --> N4 --> N5 --> G1 --> G2 --> G3 --> G4 --> K1 --> K2
    K2 -- "rejected: try the next cut" --> G4
    K2 -- "accepted" --> OUT
    style ING fill:#f8fafc,stroke:#cbd5e1,color:#334155
    style NAR fill:#f8fafc,stroke:#cbd5e1,color:#334155
    style PRO fill:#f8fafc,stroke:#cbd5e1,color:#334155
    style CHK fill:#f8fafc,stroke:#cbd5e1,color:#334155

    classDef ai fill:#ede9fe,stroke:#6d28d9,color:#2e1065,stroke-width:2px
    classDef code fill:#e0f2fe,stroke:#0369a1,color:#082f49,stroke-width:2px
    classDef data fill:#f1f5f9,stroke:#64748b,color:#0f172a
```

| Stage | Key rule | Why |
|---|---|---|
| Parse | Fails loudly on an unknown template instead of returning nothing | A silent empty result looks like "no good promos" |
| Story model | One call per episode; episodes run in parallel and are cached separately | An edited episode re-runs only itself |
| Annotators | Two different model families; **tags only, no scores** | Self-reported scores just restated rank order (S1); two families give an independent second opinion |
| Verify | A tag whose quote belongs to a neighbouring event is moved there; a quote found nowhere drops the tag | Models sometimes cite the wrong event ID |
| Event-type rules | "Physical action" and "striking visual" need a visible event; "shocking line" needs dialogue | Stops a sound effect being called a slap |
| Series model | Aliases merged ("Lady in Gold Gown" = Clarissa); storylines linked only when they share 2+ characters | A shared protagonist alone must not merge two plots |

**The 14 event tags:**
- Narrative: `betrayal` `new_conflict` `escalation` `reversal` `reveal` `open_question` `resolution`
- Attention: `physical_action` `shocking_line` `threat` `confrontation` `humiliation` `emotional_shift` `striking_visual`

---

## 3. One episode through the narrative engine

```mermaid
sequenceDiagram
    autonumber
    participant P as Parser (code)
    participant S as Story model (AI)
    participant A as Annotator A (AI)
    participant B as Annotator B (AI)
    participant V as Verifier (code)
    participant M as Series model (code)

    P->>S: Episode as numbered events
    S-->>P: Characters, aliases, storylines (JSON)
    Note over S: A name is accepted only if it appears in this episode's text
    par Two independent opinions
        P->>A: Events + story context
        A-->>V: Tags, each with a quote
    and
        P->>B: Events + story context
        B-->>V: Tags, each with a quote
    end
    V->>V: Check quotes, repair IDs, enforce event-type rules
    V->>V: Consensus counts how many models agree (confidence only)
    V->>M: Verified tags
    M->>M: Merge aliases across episodes, link storylines
```

- **Cut-off answers are never trusted.** If a model runs out of output tokens, the episode is re-annotated in smaller windows of events.
- **Every response is cached** under a key that covers the model, its parameters and the full prompt. Re-runs are free and an interrupted run resumes where it stopped.

---

## 4. How a promo is cut

```mermaid
flowchart LR
    A["⏱ 0 s<br/>Start on a legal cut<br/>at most 2 s before the hook"]:::code
    B["⚡ Hook event<br/>lands inside the first 3 s"]:::ai
    C["Build-up<br/>story moves forward"]:::data
    D["❓ Cliffhanger event<br/>open question · threat · reveal"]:::ai
    E["✂ Cut<br/>at most 1.5 s later<br/>never inside a spoken line"]:::code
    A --> B --> C --> D --> E

    classDef ai fill:#ede9fe,stroke:#6d28d9,color:#2e1065,stroke-width:2px
    classDef code fill:#e0f2fe,stroke:#0369a1,color:#082f49,stroke-width:2px
    classDef data fill:#f1f5f9,stroke:#64748b,color:#0f172a
```

**Hard rules (every promo, checked by code):**
- 20–60 seconds in total, with at most 2 segments.
- Never cut inside a spoken line.
- The hook starts within 2 s; the cut is within 1.5 s of the cliffhanger.
- Nothing is resolved in the final 6 s. An event tagged `resolution` can never be the cliffhanger.

**When two segments are joined:**

```mermaid
flowchart LR
    H["Episode 27 · segment 1 · 6 s<br/>⚡ hook: 'What is that maid doing<br/>in Her Grace's private chamber?'"]:::ai
    T["Episode 27 · segment 2 · 53 s<br/>❓ ends on Juliana's terrified eyes<br/>as the trap closes"]:::ai
    H -- "✂ 39 s skipped" --> T

    classDef ai fill:#ede9fe,stroke:#6d28d9,color:#2e1065,stroke-width:2px
```

A promo is joined from two segments only to:
- connect a strong hook to a cliffhanger more than 60 s away;
- skip a resolution that a continuous clip would include;
- bridge to one of the next two episodes on a shared storyline.

The two segments must share a storyline or a character. The example above is the #1 promo of the final 50-episode run.

---

## 5. Three separate scores

```mermaid
flowchart LR
    C["Candidate cut"]:::code --> V{"Validity<br/>every hard rule?"}:::gate
    V -- "no" --> X["Dropped<br/>with the reason"]:::bad
    V -- "yes" --> Q["Quality 0–100<br/>from tag evidence"]:::code
    V -- "yes" --> F["Confidence 0–1<br/>do the models agree?"]:::code
    Q --> R["Ranking<br/>by quality"]:::code
    F -. "tie-break only" .-> R

    classDef code fill:#e0f2fe,stroke:#0369a1,color:#082f49,stroke-width:2px
    classDef gate fill:#fef3c7,stroke:#b45309,color:#451a03,stroke-width:2px
    classDef bad fill:#fee2e2,stroke:#b91c1c,color:#450a0a
```

```mermaid
%%{init: {"theme": "base", "themeVariables": {"pie1": "#6d28d9", "pie2": "#0369a1", "pie3": "#15803d", "pie4": "#b45309", "pie5": "#64748b", "pieSectionTextColor": "#ffffff", "pieStrokeColor": "#ffffff", "pieOuterStrokeWidth": "0px", "pieTitleTextSize": "18px"}}}%%
pie showData
    title Quality weights (the brief's 3 criteria carry 80%)
    "Hook" : 30
    "Cliffhanger" : 30
    "Makes sense alone" : 20
    "Story progression" : 12
    "Intensity" : 8
```

- **Validity** is pass/fail. Only valid promos are ranked.
- **Quality** comes from tag evidence only. "Makes sense alone" is penalised by context needed from earlier episodes, a crowded cast, names mentioned but never shown, and extra segments.
- **Confidence** reflects agreement, repaired IDs and parse warnings. It is shown beside quality and **never replaces it**: models agreeing is not proof of quality.
- **Variety:** each promo already picked from the same episode lowers a candidate by 10%, and from the same storyline by 4%. Near-duplicates (same hook, same cliffhanger, or more than 50% overlap) are skipped.
- The weights are **stated assumptions, not fitted values**. No editor labels exist yet, and editor choices are the data that would fit them.

---

## 6. The clip check and selection rounds

```mermaid
flowchart TD
    R["Rank all valid candidates"]:::code --> S["Take the unchecked shortlist"]:::code
    S --> M["Model sees ONLY the clip text<br/>no story memory · no logline"]:::ai
    M --> E["It classifies and extracts:<br/>• type of the first-3 s event<br/>• type of the final beat<br/>• the central question<br/>• any line that settles it"]:::ai
    E --> D{"Python rules"}:::gate
    D -- "routine opening · settled ending ·<br/>music-only ending · broken jump" --> J["Reject, with the reason"]:::bad
    D -- "passes" --> A["Accept"]:::good
    J --> R
    A --> T{"Top N all checked<br/>or budget used?"}:::gate
    T -- "no" --> R
    T -- "yes" --> O[/"Final ranking"/]:::data

    classDef ai fill:#ede9fe,stroke:#6d28d9,color:#2e1065,stroke-width:2px
    classDef code fill:#e0f2fe,stroke:#0369a1,color:#082f49,stroke-width:2px
    classDef gate fill:#fef3c7,stroke:#b45309,color:#451a03,stroke-width:2px
    classDef bad fill:#fee2e2,stroke:#b91c1c,color:#450a0a
    classDef good fill:#dcfce7,stroke:#15803d,color:#052e16
    classDef data fill:#f1f5f9,stroke:#64748b,color:#0f172a
```

- **Why "classify, don't judge":** the verdict version passed nearly everything. The classify version passed all 6 strong test cuts and caught 3 of the 4 clearly weak ones (S7).
- **A failed check doesn't hide the clip:** a failed opening or ending scales that score by 0.4, and the reason appears in the report.
- **Budgets:** up to 40 checks in the web app and 80 on the command line, so the cost stays bounded.
- **Known weakness:** the checker sometimes treats a twist as a "resolution" and can judge near-identical cuts differently. See *Where it picks badly* in the solution doc.

---

## 7. Swappable models

```mermaid
flowchart LR
    subgraph CFG["Configuration only · .env or app secrets"]
        R1["HOOKRANK_STORY_MODEL<br/>nebius:openai/gpt-oss-120b"]:::cfg
        R2["HOOKRANK_ANNOTATOR_MODELS<br/>two Nebius models"]:::cfg
        R3["HOOKRANK_VERIFIER_MODEL<br/>nebius:openai/gpt-oss-120b"]:::cfg
        K["Keys and prices<br/>NEBIUS_API_KEY · ANTHROPIC_API_KEY · …"]:::cfg
    end

    REG["Provider registry<br/>15 built in + any added by name"]:::code
    CL["One model client<br/>retries · pacing · cut-off detection<br/>cache · spending guard"]:::code

    subgraph WIRE["Two wire formats"]
        W1["OpenAI request format<br/>Nebius · GPT · Gemini · OpenRouter ·<br/>your own API"]:::ai
        W2["Claude Messages API<br/>official Anthropic SDK"]:::ai
    end

    R1 & R2 & R3 & K --> REG --> CL --> W1 & W2
    style CFG fill:#f8fafc,stroke:#cbd5e1,color:#334155
    style WIRE fill:#f8fafc,stroke:#cbd5e1,color:#334155

    classDef ai fill:#ede9fe,stroke:#6d28d9,color:#2e1065,stroke-width:2px
    classDef code fill:#e0f2fe,stroke:#0369a1,color:#082f49,stroke-width:2px
    classDef cfg fill:#fef9c3,stroke:#a16207,color:#422006
```

- **Swapping a model is a configuration change.** Set the provider's key and point a role at `provider:model`. No code changes.
- **Nebius is the default.** A role whose provider has no key falls back to its Nebius model.
- **Any OpenAI-compatible API** can be added with `<NAME>_BASE_URL` and `<NAME>_API_KEY`.
- **Spending is guarded before any call is sent.** Every paid call reserves its worst-case cost first, and a paid model with no known price is refused.
  - Prices come from Nebius's and OpenRouter's catalogues, a built-in Claude list, or `HOOKRANK_PRICES`.
- **Parameters a model doesn't accept** (for example temperature on reasoning models) are dropped and the request retried, instead of hard-coding rules per model.

---

## 8. The web app and its safety boundary

```mermaid
sequenceDiagram
    autonumber
    actor U as Visitor
    participant W as Streamlit page
    participant A as Adapter (studio.py)
    participant P as Pipeline (same as CLI)
    participant L as Model providers

    U->>W: Upload .zip / .txt
    W->>A: Validate (size, ZIP safety, episode limit)
    A-->>W: "Detected 50 episodes · 3,727 timed events"
    U->>W: Click Analyze
    W->>A: Cost estimate vs per-run and daily caps
    A->>P: Run in a fresh temporary folder
    P->>L: Model calls (keys stay on the server)
    L-->>P: Labels and clip checks
    P-->>A: Ranked promos
    A->>A: Delete the temporary folder
    A-->>W: Top 10 table · details · CSV / JSON
    Note over U,A: Visitors never see a key or choose a model.<br/>Results live only in their browser session.
```

| Protection | How |
|---|---|
| Keys stay secret | Read on the server from secrets; never rendered, logged or exported (tested) |
| Spend is bounded | Hard cap per run ($0.50 default) · daily cap for the app · one analysis at a time · upload limits |
| Uploads are safe | ZIP traversal, ZIP bombs and encrypted entries are rejected; names are sanitised |
| Nothing is kept | Each run gets its own temporary folder, deleted at the end; no data is shared between sessions |

---

## 9. Scale and cost

**Measured on the full 50-episode pack:**

| | |
|---|---|
| Model calls | 3 per episode + up to 40–80 clip checks (240 in total) |
| Candidate cuts scored | 11,398, by code, at no model cost |
| Time | ≈ 2 minutes (Nebius, parallel requests) |
| Cost | ≈ $0.19–0.20 per 50-episode show |
| Re-run of unchanged episodes | free (cached) |

- **Cost grows with episodes, not with candidates.** About 100 new 50-episode shows a week would cost about $20.
- **What changes for a fleet of machines:**
  - a parser adapter for each new screenplay format;
  - a shared database instead of the local SQLite cache and spend ledger;
  - provider rate limits;
  - aligning script time to the edited video.

---

## 10. Code map

| Area | Files |
|---|---|
| Entry points | `main.py` (CLI) · `streamlit_app.py` (web page) · `hookrank/studio.py` (web adapter) · `hookrank/run.py` (one pipeline for both) |
| Parsing | `hookrank/parsing.py` · `hookrank/schemas.py` |
| Narrative engine | `hookrank/narrative/` (`extraction.py` story · `annotation.py` tags · `consensus.py` · `story.py` series model · `vocab.py` the 14 tags) |
| Promo engine | `hookrank/promos/` (`generate.py` cuts · `scoring.py` weights · `rank.py` · `verify.py` clip check) · report written in `hookrank/run.py` |
| Models | `hookrank/config.py` providers and roles · `hookrank/llm.py` client · `hookrank/calls.py` cached calls · `hookrank/budget.py` spending guard · `hookrank/cache.py` |
| Prompts | `prompts/narrative_story.md` · `prompts/narrative_annotate.md` · `prompts/clip_check.md` |
| Evidence | `docs/decisions.md`: every decision, experiment and trade-off |
