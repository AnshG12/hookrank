"""HookRank command line: configure, doctor, run (full pipeline) and narrative (engine only)."""
from __future__ import annotations

import argparse
from dataclasses import replace
import getpass
import json
import logging
import sys
from pathlib import Path

from hookrank.budget import build_guard
from hookrank.cache import CallCache
from hookrank.config import PROVIDERS, Settings
from hookrank.llm import ChatClient, spec_for
from hookrank.narrative import NarrativeEngine
from hookrank.parsing import chronology_key, load_episodes
from hookrank.providers import ProviderError, extract_json
from hookrank.run import run as run_hookrank
from hookrank.tui import Prompt, banner, console, error, info, success


def write_env(values: dict[str, str]) -> None:
    """Set selected .env keys while preserving comments and unrelated settings."""
    path = Path(".env")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(values)
    updated: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
        if key in remaining:
            updated.append(f"{key}={remaining.pop(key)}")
        else:
            updated.append(line)
    if updated and updated[-1].strip():
        updated.append("")
    updated.extend(f"{key}={value}" for key, value in remaining.items())
    path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def configure() -> None:
    """Set provider keys and model roles. Models are provider:model references (see hookrank/config.py)."""
    settings = Settings.from_env()
    banner()
    console.print("\n[bold]Configure the narrative engine[/]")
    console.print("[dim]Models are written as provider:model, e.g. anthropic:claude-opus-5-5, openai:gpt-5.5, "
                  "gemini:gemini-3.5-flash-lite, nebius:openai/gpt-oss-120b. Leave a key blank to keep it.[/]")
    values = {}
    for name, provider in PROVIDERS.items():
        endpoint = settings.endpoint(name)
        if name == "custom":
            url = Prompt.ask("OpenAI-compatible gateway URL (optional)", default=endpoint.base_url or "")
            if url:
                values["CUSTOM_BASE_URL"] = url
        key = getpass.getpass(f"{provider.label} API key, {endpoint.key_variable} (input hidden; blank keeps current): ")
        if key:
            values[endpoint.key_variable] = key
    values["HOOKRANK_STORY_MODEL"] = Prompt.ask("Story model", default=settings.story_model)
    values["HOOKRANK_ANNOTATOR_MODELS"] = Prompt.ask("Annotator models (comma-separated, ideally two families)",
                                                     default=",".join(settings.annotator_models))
    values["HOOKRANK_VERIFIER_MODEL"] = Prompt.ask("Clip-check model", default=settings.verifier_model)
    write_env(values)
    success("Saved configuration in .env (file permissions restricted where supported). Run `python main.py doctor`.")


def doctor() -> int:
    """Health-check every model the pipeline will call (story model + annotators)."""
    settings = Settings.from_env()
    banner()
    console.print("\n[bold]Model health check[/]")
    if settings.missing_keys():
        error(f"Missing {', '.join(settings.missing_keys())}. Add it with `python main.py configure`.")
        return 1
    status = 0
    guard = build_guard(settings)   # paid health checks are counted against the cap like any other call
    roles = ([("story", settings.story_model)] + [("annotator", m) for m in settings.annotator_models]
             + [("verifier", settings.verifier_model)])
    for role, model in dict.fromkeys(roles):
        try:
            spec = replace(spec_for(model, settings), max_output_tokens=64)
            result = ChatClient(spec, budget=guard).complete('Return exactly this JSON object: {"status":"ok"}',
                                                             min_output_tokens=16)
            if extract_json(result.text).get("status") != "ok":
                raise ProviderError("responded, but the JSON health check did not match")
            success(f"{role:<9} {model}  ok ({result.latency_s:.1f}s)  [{spec_for(model, settings).base_url}]")
        except ProviderError as exc:
            error(f"{role:<9} {model}  {str(exc)[:200]}")
            status = 1
    console.print(f"[dim]Spend so far: ${guard.spent():.4f} of ${guard.cap_usd:.2f} cap[/]")
    guard.close()
    return status


def interactive() -> int:
    while True:
        settings = Settings.from_env()
        banner()
        console.print(f"\n[bold]Workspace[/]  [dim]{Path.cwd()}[/]")
        missing = settings.missing_keys()
        console.print(f"[bold]Keys[/]       " + ("[bright_green]all set[/]" if not missing else f"[red]missing {', '.join(missing)}[/]"))
        console.print(f"[bold]Story[/]      {settings.story_model}")
        console.print(f"[bold]Annotators[/] {', '.join(settings.annotator_models)}")
        console.print("\n[bold]1[/] Configure    [bold]2[/] Test models    [bold]3[/] Quit")
        action = Prompt.ask("", choices=["1", "2", "3"], default="2")
        if action == "1":
            configure()
        elif action == "2":
            doctor()
        else:
            return 0
        Prompt.ask("Press Enter to return", default="")


def _setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def run_narrative(args: argparse.Namespace) -> int:
    """Run the narrative engine alone: story extraction, two-model annotation, consensus."""
    output_dir = Path(args.output_dir)
    _setup_logging(output_dir / "narrative.log")
    settings = Settings.from_env()
    if settings.missing_keys() and not args.cache_only:
        error(f"Missing {', '.join(settings.missing_keys())}; run `python main.py configure` or use --cache-only.")
        return 1
    episodes, failures = load_episodes(Path(args.input), series_id=args.series_id)
    episodes.sort(key=chronology_key)
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",") if x.strip()}
        episodes = [e for e in episodes if chronology_key(e)[0] in wanted]
    if not episodes:
        error("No episodes to process.")
        return 1
    cache = CallCache(Path(args.memory_db) if args.memory_db else output_dir / "hookrank.sqlite3")
    engine = NarrativeEngine(spec_for(settings.story_model, settings),
                             [spec_for(m, settings) for m in settings.annotator_models],
                             cache, args.series_id, cache_only=args.cache_only)
    result = engine.run(episodes)
    cache.close()
    (output_dir / "narrative.json").write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    calls = [c for n in result.episodes.values() for c in n.calls]
    failed = [c for c in calls if c.failed]
    banner()
    console.print("\n[bold]Narrative engine complete[/]")
    console.print(f"Episodes: [bright_cyan]{len(result.episodes)}[/]  ·  story model [yellow]{result.story_model}[/]"
                  f"  ·  annotators [yellow]{', '.join(result.annotator_models)}[/]")
    console.print(f"Model calls: [bright_cyan]{len(calls)}[/] ({sum(c.cached for c in calls)} cached, "
                  f"{sum(c.ok and not c.cached for c in calls)} live, [red]{len(failed)} failed[/])")
    console.print(f"Characters: [bright_cyan]{len(result.series.characters)}[/] "
                  f"({sum(c.principal for c in result.series.characters)} principal)  ·  "
                  f"series threads: [bright_cyan]{len(result.series.threads)}[/]")
    console.print(f"Annotated events (consensus): [bright_cyan]{sum(len(n.consensus) for n in result.episodes.values())}[/]")
    console.print(f"Output: [bright_cyan]{(output_dir / 'narrative.json').resolve()}[/]")
    if failed:
        info(f"{len(failed)} calls failed; re-run the same command to resume (successful calls are cached).")
    if failures:
        info(f"{len(failures)} input files could not be read; see failures in the run output.")
    return 0 if not failed else 3


def main() -> int:
    parser = argparse.ArgumentParser(description="HookRank: find and rank promos in microdrama screenplays")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("tui", help="Open the interactive setup screen")
    sub.add_parser("doctor", help="Health-check every configured model")
    sub.add_parser("configure", help="Set endpoint, API key and model roles")
    sub.add_parser("spend", help="Show spend at paid endpoints against the budget cap")
    run_parser = sub.add_parser("run", help="Find and rank promos: parse, narrative engine, promo construction")
    run_parser.add_argument("--input", required=True, help="A screenplay .txt file, a folder, or a ZIP archive")
    run_parser.add_argument("--series-id", default="series", help="Namespace for this series")
    run_parser.add_argument("--output-dir", default="outputs/run", help="Where outputs and the model-call cache are written")
    run_parser.add_argument("--episodes", help="Optional comma-separated episode numbers, e.g. 1,13,25")
    run_parser.add_argument("--top", type=int, default=25, help="How many promos to rank for posting")
    run_parser.add_argument("--cache-only", action="store_true", help="Use cached model responses only; make no API calls")
    run_parser.add_argument("--memory-db", help="SQLite cache path; defaults to <output-dir>/hookrank.sqlite3")
    run_parser.add_argument("--verify-top", type=int, default=80,
                            help="Clip-check budget: verify up to this many shortlist candidates and alternatives "
                                 "(clip text only); 0 disables")
    run_parser.add_argument("--episode-tries", type=int, default=3,
                            help="Clip-check up to this many candidates per episode for the best-per-episode list; 0 disables")
    narrative_parser = sub.add_parser("narrative", help="Run the narrative engine (story extraction + two-model annotation)")
    narrative_parser.add_argument("--input", required=True, help="A screenplay .txt file, a folder, or a ZIP archive")
    narrative_parser.add_argument("--series-id", default="series", help="Namespace for this series")
    narrative_parser.add_argument("--output-dir", default="outputs/run", help="Where narrative.json and the cache are written")
    narrative_parser.add_argument("--episodes", help="Optional comma-separated episode numbers, e.g. 1,13,25")
    narrative_parser.add_argument("--memory-db", help="SQLite cache path; defaults to <output-dir>/hookrank.sqlite3")
    narrative_parser.add_argument("--cache-only", action="store_true", help="Use cached model responses only; make no API calls")
    args = parser.parse_args()
    if args.command in (None, "tui"):
        return interactive()
    if args.command == "configure":
        configure()
        return 0
    if args.command == "spend":
        guard = build_guard(Settings.from_env())
        summary = guard.summary()
        guard.close()
        banner()
        console.print(f"\n[bold]Spend[/]  ${summary['spent_usd']:.4f} of ${summary['cap_usd']:.2f} cap  ·  "
                      f"[bright_green]${summary['remaining_usd']:.4f} remaining[/]")
        for row in summary["by_model"]:
            console.print(f"  {row['model']:<44} {row['calls']:>5} calls  {row['prompt_tokens']:>9} in  "
                          f"{row['completion_tokens']:>8} out  ${row['cost_usd']:.4f}")
        return 0
    if args.command == "doctor":
        return doctor()
    if args.command == "run":
        output_dir = Path(args.output_dir)
        _setup_logging(output_dir / "run.log")
        settings = Settings.from_env()
        if settings.missing_keys() and not args.cache_only:
            error(f"Missing {', '.join(settings.missing_keys())}; run `python main.py configure` or use --cache-only.")
            return 1
        wanted = {int(x) for x in args.episodes.split(",") if x.strip()} if args.episodes else None
        try:
            outcome = run_hookrank(Path(args.input), output_dir, args.series_id, settings, episode_numbers=wanted,
                                   top_n=max(1, args.top), cache_only=args.cache_only,
                                   memory_db=Path(args.memory_db) if args.memory_db else None,
                                   verify_top=max(0, args.verify_top), episode_tries=max(0, args.episode_tries))
        except (ValueError, OSError, ProviderError) as exc:
            error(str(exc))
            return 1
        m = outcome["manifest"]
        banner()
        console.print("\n[bold]Run complete[/]  [dim]" + m["run_id"] + "[/]")
        console.print(f"Episodes: [bright_cyan]{m['episodes']}[/]  ·  timed events: [bright_cyan]{m['events']}[/]")
        console.print(f"Model calls: [bright_cyan]{m['model_calls']['total']}[/] ({m['model_calls']['cached']} cached, "
                      f"{m['model_calls']['live']} live, [red]{m['model_calls']['failed']} failed[/])")
        console.print(f"Characters: [bright_cyan]{m['narrative']['characters']}[/]  ·  storylines: "
                      f"[bright_cyan]{m['narrative']['series_threads']}[/]  ·  annotated events: "
                      f"[bright_cyan]{m['narrative']['annotated_events']}[/]")
        console.print(f"Promo candidates: [bright_cyan]{m['promos']['candidates_evaluated']}[/] evaluated, "
                      f"[bright_green]{m['promos']['valid_candidates']}[/] valid, [bright_green]{m['promos']['ranked']}[/] ranked; "
                      f"{m['promos']['episodes_with_valid_promo']}/{m['promos']['episodes_total']} episodes covered")
        cc = m["clip_check"]
        if cc["checked"]:
            r = cc["recommendations"]
            console.print(f"Clip check: [bright_cyan]{cc['checked']}[/] checked - accept {r['accept']}, revise {r['revise']}, "
                          f"reject {r['reject']}, unverified {r['unverified']}  ·  failed: "
                          + ", ".join(f"{k} {n}" for k, n in cc["failed"].items())
                          + f"  ·  shortlist verified {cc['shortlist_verified']}/{m['promos']['ranked']}"
                          + f"  ·  episode picks accepted {cc['episode_picks_accepted']}/{cc['episodes']}")
        console.print(f"Spend: [yellow]${m['spend']['this_run_usd']:.4f} this run[/], ${m['spend']['spent_usd']:.4f} of "
                      f"${m['spend']['cap_usd']:.2f} cap in total")
        console.print(f"Report: [bright_cyan]{(output_dir / 'promos_report.md').resolve()}[/]")
        if m["model_calls"]["failed"]:
            info(f"{m['model_calls']['failed']} model calls failed; re-run the same command to resume (successes are cached).")
        return 0
    if args.command == "narrative":
        return run_narrative(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
