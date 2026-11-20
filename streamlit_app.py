"""HookRank - single-page Streamlit app: upload -> analyze -> review -> download.

All AI work is done by the existing pipeline through hookrank/studio.py. The Nebius key lives in
Streamlit secrets (or environment variables locally) and is used only on the server. The app is open
to anyone with the link (no sign-in, D16): spend is bounded by the per-run cap, the app-wide daily
cap, one analysis at a time and the upload limits.
"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import threading
import time

import streamlit as st

from hookrank import studio
from hookrank.config import load_dotenv
from hookrank.providers import ProviderError

st.set_page_config(page_title="HookRank", page_icon="🎬", layout="centered")
load_dotenv()            # local development only; on Streamlit Community Cloud, secrets are used
log = logging.getLogger("hookrank.app")

CONFIG_KEYS = ("NEBIUS_API_KEY", "NEBIUS_CONCURRENCY", "HOOKRANK_STORY_MODEL", "HOOKRANK_ANNOTATOR_MODELS",
               "HOOKRANK_VERIFIER_MODEL")


def secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:            # no secrets file (local runs fall back to environment variables)
        pass
    return os.environ.get(key, default)


RUN_BUDGET = float(secret("HOOKRANK_RUN_BUDGET_USD", "0.50"))
LIMITS = studio.Limits(max_episodes=int(secret("HOOKRANK_MAX_EPISODES", "60")))


@st.cache_resource
def shared() -> dict:
    """Process-wide objects shared by all sessions (no user data): one analysis at a time, daily spend."""
    return {"slot": threading.BoundedSemaphore(1),
            "daily": studio.DailyBudget(float(secret("HOOKRANK_APP_DAILY_BUDGET_USD", "3.00")))}


@st.cache_resource(ttl=3600)
def cached_prices() -> dict:
    """Public per-token prices from the Nebius catalogue (not user data)."""
    try:
        return studio.nebius_prices(studio.studio_settings(model_config(), studio.Path("/dev/null"), RUN_BUDGET))
    except ProviderError:
        return {}


def model_config() -> dict:
    return {k: v for k in CONFIG_KEYS if (v := secret(k))}


CSS = """
<style>
.excerpt {font-size: 0.86rem; line-height: 1.45; border: 1px solid rgba(128,128,128,.25); border-radius: 8px;
          padding: .5rem .7rem; max-height: 460px; overflow-y: auto;}
.excerpt .row {padding: .12rem .3rem; border-radius: 4px;}
.excerpt .t {display: inline-block; width: 3.4rem; opacity: .6; font-variant-numeric: tabular-nums;}
.excerpt .st {display: inline-block; width: 8.6rem; opacity: .45; font-variant-numeric: tabular-nums; font-size: .78rem;}
.excerpt .hook {background: rgba(255, 196, 0, .22);}
.excerpt .cliffhanger {background: rgba(160, 90, 255, .22);}
.excerpt .opening, .excerpt .ending {background: rgba(128, 128, 128, .08);}
.excerpt .tag {font-size: .7rem; font-weight: 700; letter-spacing: .04em; padding: 0 .35rem; margin-left: .3rem;
               border-radius: 3px; border: 1px solid currentColor; opacity: .8;}
.excerpt .cut {text-align: center; opacity: .6; font-size: .78rem; margin: .3rem 0; border-top: 1px dashed rgba(128,128,128,.5);}
</style>
"""


# ---- upload ------------------------------------------------------------------------------------

def handle_upload(files) -> None:
    key = hashlib.sha256("|".join(f"{f.name}:{f.size}:{getattr(f, 'file_id', '')}" for f in files).encode()).hexdigest()
    if st.session_state.get("upload_key") == key:
        return
    st.session_state.upload_key = key
    for k in ("sources", "preview", "upload_error"):
        st.session_state.pop(k, None)
    try:
        sources = studio.validate_uploads([(f.name, f.getvalue()) for f in files], LIMITS)
        pv = studio.preview(sources, LIMITS)
        st.session_state.sources = sources                   # session memory only, never shared or cached
        st.session_state.preview = {"episodes": len(pv.usable), "events": pv.events,
                                    "skipped": len(pv.episodes) - len(pv.usable), "unreadable": len(pv.unreadable),
                                    "numbers": [e.episode_number for e in pv.usable]}
    except studio.UploadError as exc:
        st.session_state.upload_error = str(exc)


def run_analysis() -> None:
    sources, pv = st.session_state.sources, st.session_state.preview
    slot, daily = shared()["slot"], shared()["daily"]
    if not slot.acquire(blocking=False):
        st.warning("Another analysis is running right now. Please try again in a few minutes.")
        return
    try:
        cfg = model_config()
        settings = studio.studio_settings(cfg, studio.Path("/dev/null"), RUN_BUDGET)
        est = studio.estimate([e for e in studio.preview(sources, LIMITS).usable], settings, cached_prices())
        if est.usd > RUN_BUDGET:
            st.error(f"This upload would need about ${est.usd:.2f}, above the ${RUN_BUDGET:.2f} limit per analysis. "
                     "Upload fewer episodes.")
            return
        if est.usd > daily.remaining():
            st.error("Today's analysis budget for this demo is used up. Please try again tomorrow.")
            return
        progress = studio.Progress(narrative_calls=pv["episodes"] * (1 + len(settings.annotator_models)))
        box: dict = {}

        def worker() -> None:
            try:
                box["result"] = studio.analyze(sources, cfg, RUN_BUDGET, progress)
            except (studio.UploadError, ProviderError) as exc:
                box["error"] = str(exc)
            except Exception as exc:                          # details stay in server logs (type only)
                log.error("analysis failed: %s", type(exc).__name__)
                box["error"] = "The analysis failed unexpectedly. Please try again."

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        bar = st.progress(0.0, text="Starting analysis")
        while thread.is_alive():
            fraction, text = progress.snapshot()
            bar.progress(fraction, text=text)
            time.sleep(0.5)
        thread.join()
        bar.empty()
        if "error" in box:
            st.error(box["error"])
            return
        result = box["result"]
        daily.record(result.spend_usd)
        st.session_state.analysis = result
    finally:
        slot.release()


# ---- results -----------------------------------------------------------------------------------

def show_details(p, episodes) -> None:
    cc = p.clip_check or {}
    segments = " + ".join(f"{s.episode} {s.start}–{s.end}" for s in p.segments)
    st.markdown(f"#### #{p.rank} · {segments} · {p.duration_seconds:.0f}s")
    who = lambda m: f"{m.speaker}: " if m.speaker else f"{m.event_type.title()}: "
    st.markdown(f"**Hook** · {p.hook.episode} {p.hook.time}  \n{who(p.hook)}{p.hook.text}")
    st.markdown(f"**Cliffhanger** · {p.cliffhanger.episode} {p.cliffhanger.time}  \n{who(p.cliffhanger)}{p.cliffhanger.text}")
    question = (cc.get("ending") or {}).get("unresolved_question") or p.cliffhanger.question
    if question:
        st.markdown(f"**Unanswered question** · {question}")
    st.markdown(f"**Why it works** · {p.why}")

    rows = studio.excerpt_rows(p, episodes)
    lines = []
    for r in rows:
        if r["cut"]:
            lines.append(f"<div class='cut'>✂ {html.escape(r['text'])}</div>")
            continue
        tag = {"hook": "HOOK", "cliffhanger": "CLIFFHANGER"}.get(r["role"], "")
        lines.append(f"<div class='row {r['role']}'><span class='t'>{r['promo_time']}</span>"
                     f"<span class='st'>{html.escape(r['script_time'])}</span><b>{html.escape(r['label'])}</b>: "
                     f"{html.escape(r['text'])}{f'<span class=tag>{tag}</span>' if tag else ''}</div>")
    st.markdown("**Excerpt** · promo time, script time, line")
    st.markdown(CSS + "<div class='excerpt'>" + "".join(lines) + "</div>", unsafe_allow_html=True)

    comps = p.scores.components
    st.caption(f"Quality {p.scores.quality:.0f}/100 · confidence {p.scores.confidence_label} · "
               f"hook {comps['hook']:.2f} · cliffhanger {comps['cliffhanger']:.2f} · standalone {comps['standalone']:.2f} · "
               f"progression {comps['progression']:.2f} · engagement {comps['engagement']:.2f}")
    if cc.get("status") == "verified":
        st.caption(f"Clip check: {cc['recommendation']} (opening: {cc['opening'].get('type', '').replace('_', ' ')}; "
                   f"ending: {cc['ending'].get('type', '').replace('_', ' ')})")
    else:
        st.warning("This promo was not clip-checked; treat its ranking as provisional.")
    if p.scores.confidence_notes:
        st.caption("Confidence notes: " + "; ".join(p.scores.confidence_notes))


def show_results(a) -> None:
    st.subheader("Top promos")
    st.caption(f"{a.episode_count} episodes · {a.candidates:,} candidate cuts · {a.seconds:.0f}s · "
               f"model cost ${a.spend_usd:.3f} · times are script time (MM:SS.cc)")
    for warning in a.warnings:
        st.warning(warning)
    if not a.promos:
        st.info("No valid promo was found in these episodes.")
        return
    event = st.dataframe(studio.summary_rows(a.promos), hide_index=True, on_select="rerun",
                         selection_mode="single-row", key="promo_table", width="stretch")
    selected = event.selection.rows[0] if event.selection.rows else 0   # selecting only re-renders stored results
    show_details(a.promos[selected], a.episodes)
    left, right = st.columns(2)
    left.download_button("Download CSV", a.csv_bytes, "hookrank_promos.csv", "text/csv", width="stretch")
    right.download_button("Download JSON", a.json_bytes, "hookrank_promos.json", "application/json", width="stretch")


# ---- page --------------------------------------------------------------------------------------

st.title("HookRank")
st.markdown("*AI-powered promo discovery for microdramas.*")
st.caption("Uploaded screenplays are sent to Nebius Token Factory for analysis, processed in a temporary "
           "workspace that is deleted when the analysis ends, and kept only in your browser session.")

files = st.file_uploader("Upload a ZIP of episodes or .txt screenplay files", type=["zip", "txt"],
                         accept_multiple_files=True)
if files:
    handle_upload(files)
else:
    for k in ("upload_key", "sources", "preview", "upload_error"):
        st.session_state.pop(k, None)
if st.session_state.get("upload_error"):
    st.error(st.session_state.upload_error)
pv = st.session_state.get("preview")
if pv:
    detail = f"Detected **{pv['episodes']} episode{'s' if pv['episodes'] != 1 else ''}** · {pv['events']:,} timed events"
    if pv["skipped"] or pv["unreadable"]:
        detail += f" · {pv['skipped'] + pv['unreadable']} file(s) without recognisable timestamps skipped"
    st.markdown(detail)
if st.button("Analyze", type="primary", disabled=not pv):
    run_analysis()
if st.session_state.get("analysis"):
    show_results(st.session_state.analysis)
