"""S1 validation spike: are per-episode LLM event labels worth building on?

See docs/decisions.md (P1, S1). For a sample of episodes, one call per episode asks a
model to rank hook and cliffhanger *events* by ID. We measure, against chance:
  1. validity   - JSON parses, IDs exist, quotes are verbatim
  2. ending     - does a top cliffhanger pick land in the episode's final 10s?
  3. agreement  - overlap between two unrelated model families
  4. stability  - overlap between two runs of the same model
  5. saturation - do strength scores use the range, or bunch at the top?

Leakage control: 22/50 scripts contain the literal word "cliffhanger" in stage
directions; it is removed from model input so the ending test is not rigged.

Usage: python experiments/s1_label_spike.py [--episodes 1,13,25,37,50]
Outputs (screenplay-derived, git-ignored): outputs/spikes/s1/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from hookrank.config import Settings  # noqa: E402
from hookrank.parsing import chronology_key, fmt_ms, load_episodes  # noqa: E402
from hookrank.providers import USER_AGENT, ProviderError, extract_json  # noqa: E402


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 90) -> dict:
    """The HTTP helper this spike originally ran with (kept local so the experiment stays
    reproducible after the legacy provider module was removed; see docs/decisions.md D8)."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise ProviderError(f"Provider returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise ProviderError(f"Could not reach provider: {exc}") from exc
    raise ProviderError("Provider request failed after retries.")

PRIMARY = "openai/gpt-oss-120b"
JUDGE = "qwen/qwen3.8-27b"
# gpt-oss is a reasoning model: with Groq defaults its hidden reasoning exhausts the
# output budget and returns empty/truncated JSON (observed 4/10 failures). Effort and
# budget are therefore always explicit. Runs: name -> (model, extra request params).
# Groq free tier (headers, 2026-11-19): 8,000 tokens/min and 1,000 requests/day per
# model. A request whose prompt + max_completion_tokens budget exceeds 8k is rejected
# outright (HTTP 413), so medium effort (~4-6k reasoning tokens) cannot fit; output
# budgets are kept small. Observed low-effort completions: ~700-1,100 tokens.
RUNS = {
    "primary_low_a": (PRIMARY, {"reasoning_effort": "low", "max_completion_tokens": 3000}),
    "primary_low_b": (PRIMARY, {"reasoning_effort": "low", "max_completion_tokens": 3000}),
    "judge": (JUDGE, {"max_completion_tokens": 3000}),
}
COMPARISONS = {  # name -> (run a, run b)
    "stability_low": ("primary_low_a", "primary_low_b"),
    "agreement_low_vs_judge": ("primary_low_a", "judge"),
}
PICKS = 6
ENDING_WINDOW_MS = 10_000
MATCH_TOLERANCE_MS = 3_000
LEAK_RE = re.compile(r"\s*\bcliff-?hangers?\b", re.I)

PROMPT = """You are annotating one episode of a vertical microdrama screenplay to help find promo clips for TikTok/Instagram/YouTube.
Each line below is one timed event, with an ID.

Pick, by event ID only:
1. "hooks": the {n} events that would most make a scrolling stranger STOP if a clip OPENED on that event (e.g. a slap, a reveal, a shocking line, a striking image). Rank strongest first.
2. "cliffhangers": the {n} events where CUTTING a clip right after that event would leave the strongest unresolved question, so the viewer needs to know what happens next. Rank strongest first. These may be anywhere in the episode.

For every pick return an object with:
- "id": the event ID exactly as listed (e.g. "E012")
- "strength": integer 1-5 (1 very weak, 3 adequate, 5 exceptional). Use the full range honestly; not every pick is a 5.
- "type": hooks: slap|reveal|shocking_line|threat|confrontation|visual|other ; cliffhangers: unanswered_question|threat|pending_reveal|pending_decision|danger|other
- "quote": up to 15 words copied EXACTLY from that event's text
- "context_needed": "none"|"some"|"heavy" - how much a first-time viewer, who has never seen the show, needs earlier episodes to understand this moment
- "reason": at most 20 words

Rules: use only IDs listed below; never invent events or quotes. Return only one JSON object: {{"hooks":[...],"cliffhangers":[...]}}

EPISODE (times are MM:SS.cc from episode start):
{events}
"""


def render_episode(episode) -> tuple[str, dict[str, object], int]:
    """Return prompt text, short-ID -> event map, and number of leak redactions."""
    lines, id_map, redactions = [], {}, 0
    for i, event in enumerate(episode.events, 1):
        short = f"E{i:03d}"
        id_map[short] = event
        content, n = LEAK_RE.subn("", " ".join(event.content.split()))
        redactions += n
        label = event.event_type + (f" {event.speaker}" if event.speaker else "")
        if event.delivery:
            label += f" ({event.delivery})"
        lines.append(f"{short} [{fmt_ms(event.start_ms)}-{fmt_ms(event.end_ms)}] {label}: {content}")
    return "\n".join(lines), id_map, redactions


def call(settings: Settings, model: str, prompt: str, extra: dict) -> tuple[str, dict, float, str]:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.1, "response_format": {"type": "json_object"}, **extra}
    headers = {"Authorization": f"Bearer {settings.endpoint('openai').api_key}"}
    for attempt in range(6):
        started = time.time()
        try:
            data = _post_json(f"{settings.endpoint('openai').base_url}/chat/completions", payload, headers, timeout=180)
            choice = data["choices"][0]
            return choice["message"]["content"], data.get("usage", {}), time.time() - started, choice.get("finish_reason", "")
        except ProviderError as exc:
            wait = re.search(r"try again in ([\d.]+)(ms|s)", str(exc))
            if "HTTP 429" not in str(exc) or attempt == 5:
                raise
            delay = (float(wait.group(1)) / (1000 if wait.group(2) == "ms" else 1)) if wait else 10 * (attempt + 1)
            print(f"    rate-limited; waiting {delay + 1:.1f}s", flush=True)
            time.sleep(delay + 1)
    raise ProviderError("unreachable")


def norm(text: str) -> str:
    return " ".join(LEAK_RE.sub("", text).split()).strip("\"'“”‘’ .,!?…").casefold()


def check(result: dict, id_map: dict) -> dict:
    out = {"json_ok": True}
    for key in ("hooks", "cliffhangers"):
        picks = result.get(key) if isinstance(result.get(key), list) else []
        ids = [p.get("id") for p in picks if isinstance(p, dict)]
        valid = [i for i in ids if i in id_map]
        quotes_ok = sum(1 for p in picks if isinstance(p, dict) and p.get("id") in id_map
                        and norm(str(p.get("quote", ""))) and norm(str(p.get("quote", ""))) in norm(id_map[p["id"]].content))
        out[key] = {"returned": len(ids), "valid_ids": len(valid), "invalid_ids": [i for i in ids if i not in id_map],
                    "duplicate_ids": len(ids) - len(set(ids)), "verbatim_quotes": quotes_ok,
                    "strengths": [p.get("strength") for p in picks if isinstance(p, dict)],
                    "context_needed": [p.get("context_needed") for p in picks if isinstance(p, dict)],
                    "types": [p.get("type") for p in picks if isinstance(p, dict)],
                    "ids": list(dict.fromkeys(valid))}
    return out


def overlap(a: list[str], b: list[str], id_map: dict) -> dict:
    """Exact-ID Jaccard plus a tolerant match: same moment if starts are within 3s."""
    sa, sb = set(a), set(b)
    jaccard = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    tolerant = sum(1 for x in a if any(abs(id_map[x].start_ms - id_map[y].start_ms) <= MATCH_TOLERANCE_MS for y in b))
    return {"exact_shared": len(sa & sb), "jaccard": round(jaccard, 3),
            "tolerant_matched": tolerant, "of": len(a), "top1_same": bool(a and b and a[0] == b[0])}


def chance_shared(n_events: int, k: int) -> float:
    """Expected exact overlap of two independent random k-subsets of n events."""
    return k * k / n_events if n_events else 0.0


def chance_ending_hit(n_events: int, n_ending: int, k: int) -> float:
    """P(at least one of k random events falls in the ending window)."""
    if n_events < k:
        return 1.0
    return 1 - comb(n_events - n_ending, k) / comb(n_events, k)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="1,13,25,37,50")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "spikes" / "s1"))
    args = parser.parse_args()
    wanted = {int(x) for x in args.episodes.split(",")}
    settings = Settings.from_env()
    if not settings.endpoint('openai').api_key:
        print("OPENAI_API_KEY is not set (Groq key expected).")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes, failures = load_episodes(ROOT / "data", "spike")
    episodes = [e for e in sorted(episodes, key=chronology_key) if chronology_key(e)[0] in wanted]
    report, raw_log = {"primary": PRIMARY, "judge": JUDGE, "picks_per_list": PICKS, "episodes": []}, []

    for episode in episodes:
        text, id_map, redactions = render_episode(episode)
        prompt = PROMPT.format(n=PICKS, events=text)
        n = len(id_map)
        ending_ids = [i for i, e in id_map.items() if e.end_ms >= episode.duration_ms - ENDING_WINDOW_MS]
        row = {"episode": episode.episode_number, "events": n, "duration_s": episode.duration_ms / 1000,
               "leak_redactions": redactions, "prompt_chars": len(prompt), "runs": {}}
        print(f"{episode.episode_number}: {n} events, {redactions} leak word(s) removed", flush=True)
        for run_name, (model, extra) in RUNS.items():
            try:
                raw, usage, secs, finish = call(settings, model, prompt, extra)
                raw_log.append({"episode": episode.episode_number, "run": run_name, "model": model, "raw": raw})
                try:
                    result = check(extract_json(raw), id_map)
                except ProviderError as exc:
                    result = {"json_ok": False, "error": str(exc)}
                result.update({"model": model, "params": extra, "seconds": round(secs, 1), "usage": usage, "finish_reason": finish})
            except ProviderError as exc:
                result = {"json_ok": False, "model": model, "params": extra, "error": str(exc)[:400]}
            row["runs"][run_name] = result
            status = "ok" if result.get("json_ok") else f"FAILED: {result.get('error', '')[:120]}"
            print(f"  {run_name:<9} {model:<22} {status} {result.get('seconds', '')}s", flush=True)
            time.sleep(2)  # gentle pacing for free-tier limits

        runs = row["runs"]
        row["ending_events"] = len(ending_ids)
        row["ending"] = {}
        for run_name, run in runs.items():
            if run.get("json_ok"):
                cliffs = run["cliffhangers"]["ids"]
                row["ending"][run_name] = {"hit": any(i in ending_ids for i in cliffs),
                                           "top1_in_ending": bool(cliffs and cliffs[0] in ending_ids),
                                           "picks_in_ending": sum(i in ending_ids for i in cliffs), "of": len(cliffs),
                                           "chance_hit": round(chance_ending_hit(n, len(ending_ids), len(cliffs)), 3)}
        row["chance_exact_shared"] = round(chance_shared(n, PICKS), 2)
        for name, (ra, rb) in COMPARISONS.items():
            if runs[ra].get("json_ok") and runs[rb].get("json_ok"):
                row[name] = {k: overlap(runs[ra][k]["ids"], runs[rb][k]["ids"], id_map) for k in ("hooks", "cliffhangers")}
        a = runs["primary_low_a"]
        row["primary_picks"] = {k: [{"id": i, "at": fmt_ms(id_map[i].start_ms), "type": id_map[i].event_type,
                                     "text": id_map[i].content[:110]} for i in a[k]["ids"]]
                                for k in ("hooks", "cliffhangers")} if a.get("json_ok") else {}
        report["episodes"].append(row)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "raw_outputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_log), encoding="utf-8")
    print(f"\nWrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
