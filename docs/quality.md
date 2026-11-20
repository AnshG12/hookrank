# Quality checking: how we know the output is good

**Ground rule:** there are no editor or audience labels, and we did no human labelling. So we measured what can be measured on real runs. For every experiment, the pass criteria were written **before** running it, and every result, including the failures, is logged in [decisions.md](decisions.md).

## 1. Checked on every run, for every candidate (code, not AI)

- **Validity:** 20–60 s long, hook inside the first 3 s, never cut inside a spoken line, nothing resolved in the final seconds. In the final run, 174 of 11,398 candidate cuts were rejected for resolved endings.
- **Evidence:** every AI label must quote the script word for word.
  - A quote that sits on a neighbouring event moves the label there.
  - A quote found nowhere drops the label.
  - This mattered: in the first test, 8% of one model's labels pointed at the wrong event.
- **Coverage:** every one of the 50 episodes has a valid promo. All of the top 25 passed through the clip check, and the best promo per episode was accepted in 44 of 50. There were 0 failed model calls.
- **Reproducibility:** replaying the final run from its saved model answers gives the identical ranking. Each run records its models, prompt fingerprints and input fingerprints. 71 automated tests pass.

## 2. Experiments with pass criteria set in advance

| Experiment | Question | Result |
|---|---|---|
| S1: label test (5 episodes, 2 model families) | Can models label events reliably? | 0 invented event IDs. Episode endings were found far above chance, and the models agreed on key moments above chance. But the models' own "strength" scores just restated their ranking order, so we dropped them. |
| S4/S5: model choice (11 Nebius models) | Which model labels best? | The rule was fixed before seeing results. Reasoning models failed by running out of output tokens. Qwen3-235B won on finding endings (5/5) and rerun stability (0.96 hooks / 0.82 cliffhangers). |
| S6: prompt revision | Do tighter label definitions help? | The crowded top improved (17% of candidates scoring 95+ fell to 3.2%). But 2 of the 5 criteria **failed**, so we did not claim success. |
| S7: clip-check calibration (12 cuts) | Can an AI check a clip? | Asking "does this pass?" got a "yes" on 23 of 24 cuts, so it was useless. Asking it to *classify* the opening and ending, with code deciding, passed all 6 strong cuts and caught 3 of the 4 clearly weak ones. |
| S8: label fix | Does a stricter definition of "physical action" help? | **No.** Its tag count rose from 191 to 255. We replaced it with rules enforced in code and a lower weight. |

## 3. Before and after: top 10 promos across three versions

The clips were read in full by an AI reviewer (Claude), not by a human editor, so treat this as directional.

| Measure | Version 1 (D12) | Version 2 (D13) | Final (D14) |
|---|---|---|---|
| Candidates scoring 95+ (a crowded top means the ranking can't separate promos) | 17% | 3.2% | 1.3% |
| Top 10 opening on a routine movement | – | 2 | 0 |
| Top 10 ending on a resolved or music-only moment | – | 3 | 1 |
| Top 10 ready for an editor | – | 6/10 | 7/10 |

## 4. Honest failures (from the final run)

- **Resolved ending got through:** promo #2 (Episode 13) ends after the ring is found, so the question is already answered, and the clip check still passed it.
- **The clip checker is inconsistent:**
  - The same "Her?!" ending (Episode 44) was rejected in one cut and accepted in another.
  - It accepted an incoherent jump between scenes (Episode 20) and a low-stakes arrival (Episode 6).
  - It rejected a twist as if it were a resolution (Episode 11).
  - Its verdict can flip between runs (Episode 39).
- **"Physical action" labels are noisy:** only about 3 in 10 of Qwen's were genuine in a sample.
- **The exact order isn't stable:** two earlier model setups shared only 1 identical promo in their top 25, though 18 of the 25 came from the same episodes. The shortlist is a reliable *set* of strong moments, not a reliable *order*.
- **A strong cut was lost:** Episode 18's cut never passed the checker in 3 tries.
- **No cross-episode promo reached the top 25,** even though 1,377 were valid. Either the penalty is too harsh or they really are weaker; only editor data can tell.
- **Text only:** acting, visuals and sound can't be judged, and timestamps are script time, not video time.

## 5. What we could not measure

Without labels we could not measure precision of the top 10, the editor acceptance rate, agreement with human rankings, or how posted promos perform with real viewers. These need an editor's accept/reject decisions, which the app could start recording.

## 6. Not done because of time: quality ablation

We could have run a **quality ablation**: switch off one component at a time and measure how the top 10 changes. That would show what each part actually contributes. The planned runs, from the backlog in [quality_checker_todo.md](quality_checker_todo.md):

- the clip check turned off;
- one annotator only (A, then B) versus both together;
- the code-enforced label rules turned off;
- the quality weights moved by ±20%, to see whether the ranking depends on our unvalidated weights;
- the same run repeated, to measure rerun stability;
- every cut shifted to the nearest alternative cut point, to see how many promos stay valid.

**We did not run it because of time.** It needs no labels and costs about $0.20 per 50-episode run, so it is the clearest next step.
