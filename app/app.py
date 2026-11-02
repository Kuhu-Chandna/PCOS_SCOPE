import json
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent          # PCOS_SCOPE/app
BASE_DIR = APP_DIR.parent                           # PCOS_SCOPE
MODEL_FILE = "pcos_scope_random_forest_pipeline.joblib"
META_FILE = "pcos_scope_model_metadata.json"


def _find_file(name, pattern):
    """Exact name in the usual folders first, then any matching file inside the project."""
    for d in (APP_DIR / "models", BASE_DIR / "models", APP_DIR, BASE_DIR):
        if (d / name).exists():
            return d / name
    for root in (BASE_DIR, APP_DIR):
        try:
            hits = [h for h in sorted(root.rglob(pattern)) if ".ipynb_checkpoints" not in str(h)]
        except OSError:
            hits = []
        if hits:
            return hits[0]
    return BASE_DIR / "models" / name


MODEL_PATH = _find_file(MODEL_FILE, "*pcos*.joblib")
META_PATH = _find_file(META_FILE, "*pcos*metadata*.json")

st.set_page_config(
    page_title="PCOS SCOPE | Your Companion",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling (PCOS awareness colours: teal & orchid purple)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>

:root {
  --pcos-teal: #C62828;
  --pcos-teal-dark: #8E0000;
  --pcos-teal-soft: #FDECEC;
  --pcos-purple: #E53935;
  --pcos-purple-dark: #5C0A0A;
  --pcos-purple-soft: #FFF5F5;
  --pcos-lavender: #F4C2C2;
  --ink: #3A1414;
  --muted: #7A4A4A;
  --line: #F3D0D0;
}

html, body, .stApp, .stApp * {
  font-family: 'Times New Roman', Times, serif !important;
}
/* keep Streamlit's icon glyphs working */
.stApp [data-testid="stIconMaterial"],
.stApp [class*="material-symbols"],
.stApp [class*="material-icons"] {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

.stApp {
  background:
    radial-gradient(ellipse at 10% 0%, rgba(198,40,40,0.12) 0%, transparent 40%),
    radial-gradient(ellipse at 90% 10%, rgba(229,57,53,0.10) 0%, transparent 40%),
    linear-gradient(180deg, #FFFFFF 0%, #FFF1F1 100%);
  color: var(--ink);
}

.block-container { max-width: 1440px; padding-top: 5rem; padding-bottom: 3rem; }

/* Welcome box (home page) */
.welcome-box {
  max-width: 820px; margin: 3rem auto 2.5rem auto;
  padding: 0.8rem 1.5rem; text-align: center;
  background: linear-gradient(180deg, #FFFFFF 0%, var(--pcos-teal-soft) 100%);
  border: 2px solid var(--pcos-teal);
  border-top: 8px solid var(--pcos-teal-dark);
  border-radius: 4px;
  box-shadow: 0 10px 26px rgba(198,40,40,0.18), inset 0 0 0 4px #FFFFFF;
margin-bottom: 5rem;
}
.welcome-box h1 {
  font-weight: 900 !important; font-size: 4.6rem !important; line-height: 1.15 !important;
  color: var(--pcos-teal) !important; margin: 0 !important; letter-spacing: .04em;margin-bottom: 5rem;
}
.welcome-box .centered-divider {
  width: 120px; height: 3px; margin: 0.4rem auto;
  background: linear-gradient(90deg, var(--pcos-teal-dark), var(--pcos-purple), var(--pcos-teal-dark));
  border-radius: 2px;
}
.welcome-box .centered-welcome {
  font-size: 2.35rem; font-weight: 700; color: var(--pcos-purple-dark); line-height: 1.2;margin-bottom: 2rem;
}
.welcome-box .centered-welcome span { color: var(--pcos-teal); }
.compact-title {
  text-align: center; font-size: 1.6rem; font-weight: 800;
  color: var(--pcos-teal); letter-spacing: .05em; margin: 0.3rem 0;
}

h2, h3, h4, [data-testid="stMetricValue"] {
    color: var(--pcos-teal-dark);
  letter-spacing: -.02em;
}

/* Full-width sticky navigation bar (no radio dots) */
.stApp { overflow-x: hidden; }
div[data-testid="stRadio"]:has(input[type="radio"]) {
  position: fixed !important; top: 0; left: 0; right: 0;
  width: 100% !important; box-sizing: border-box;
  z-index: 1000000;
  margin: 0 !important;
  padding: 0.6rem 2rem;
  background: linear-gradient(135deg, var(--pcos-teal-dark), var(--pcos-teal) 50%, var(--pcos-purple-dark));
  border-radius: 0;
  box-shadow: 0 8px 22px rgba(142,0,0,0.25);
}
/* hide Streamlit's own top toolbar so the red bar is the top of the page */
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stRadio"] > div { width: 100%; }
div[data-testid="stRadio"] div[role="radiogroup"] {
  width: 100%; gap: 0.5rem; justify-content: center; flex-wrap: nowrap;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
  flex: 1 1 0; justify-content: center; text-align: center; cursor: pointer;
  background: transparent; border: 1px solid transparent;
  border-radius: 6px; padding: 0.55rem 1rem; margin: 0;
  color: #FFFFFF; font-weight: 700; transition: all 0.2s ease;
}
/* hide the radio circle so each item looks like a tab */
div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])) {
  display: none !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label p {
  color: inherit; font-weight: inherit; font-size: 1.15rem; text-align: center; width: 100%;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  background: #FFFFFF; border-color: #FFFFFF; color: var(--pcos-teal-dark);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,0.18); }
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked):hover { background: #FFFFFF; }
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color: var(--pcos-teal-dark); }
[data-testid="stSidebar"] { display: none !important; }

/* Hover cards: facts / myths / taboos */
.fact-card {
  background: #FFFFFF;
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.5rem;
  min-height: 200px;
  box-shadow: 0 6px 16px rgba(198,40,40,0.06);
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  cursor: help;
}
.fact-card .card-badge {
  display: inline-block; padding: 0.2rem 0.6rem; border-radius: 8px;
  font-size: 1.85rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: 0.05em; margin-bottom: 0.8rem;
}
.fact-card.type-fact { border-left: 5px solid var(--pcos-teal); }
.fact-card.type-fact .card-badge { background: var(--pcos-teal-soft); color: var(--pcos-teal-dark); }
.fact-card.type-myth { border-left: 5px solid var(--pcos-purple); }
.fact-card.type-myth .card-badge { background: var(--pcos-purple-soft); color: var(--pcos-purple-dark); }
.fact-card h4 { margin: 0 0 0.5rem 0; font-size: 1.15rem; color: var(--ink); }
.fact-card .reveal-text {
  opacity: 0; transform: translateY(10px);
  transition: all 0.35s ease;
  color: #555; font-size: 0.95rem; margin-top: 0.5rem; line-height: 1.5;
}
.fact-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 12px 24px rgba(198,40,40,0.14);
  background: var(--pcos-teal-soft);
}
.fact-card.type-myth:hover { background: var(--pcos-purple-soft); }
.fact-card:hover .reveal-text { opacity: 1; transform: translateY(0); }

/* Dashboard cards */
.soft-card {
  border: 1px solid var(--line); border-radius: 22px; padding: 1.5rem; background: #FFFFFF;
  min-height: 160px; box-shadow: 0 6px 18px rgba(0,0,0,0.03); transition: all 0.2s ease;
}
.soft-card:hover { transform: translateY(-3px); box-shadow: 0 12px 24px rgba(198,40,40,0.15); border-color: var(--pcos-purple); }
.soft-card h4 { margin: 0.2rem 0 0.4rem 0; }
.soft-card p { color: var(--muted); margin: 0; }
.pill {
  display: inline-block; padding: .3rem .75rem; border-radius: 99px;
  background: var(--pcos-purple-soft); color: var(--pcos-purple-dark);
  font-weight: 800; font-size: .75rem; margin-bottom: 0.5rem;
}
.hero {
  padding: 2.5rem; border-radius: 24px; margin-bottom: 1.5rem;
  background: linear-gradient(120deg, var(--pcos-teal-soft) 0%, #FFFFFF 50%, var(--pcos-purple-soft) 100%);
  border: 1px solid var(--line); box-shadow: 0 10px 30px rgba(0,0,0,0.04);
}
.hero h2 { margin: 0.3rem 0 0.6rem 0; font-size: 2.2rem; }
.hero p { color: var(--muted); font-size: 1.05rem; margin: 0; max-width: 900px; }
.note {
  background: var(--pcos-teal-soft); border: 1px solid var(--line);
  padding: 1rem; border-radius: 16px; color: var(--pcos-teal-dark); margin-bottom: 1rem;
}
.footer { text-align: center; color: var(--muted); font-size: .9rem; padding: 1rem 0; }

.stButton button, [data-testid="stFormSubmitButton"] button {
  border-radius: 14px; background: #FFFFFF; border: 1px solid var(--pcos-teal);
  color: var(--pcos-teal); font-weight: 700;
}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
  background: var(--pcos-teal); color: #fff; border-color: var(--pcos-teal);
}
.stButton button[kind="primary"]:hover { background: var(--pcos-teal-dark); border-color: var(--pcos-teal-dark); }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--pcos-teal-dark); border-bottom-color: var(--pcos-teal); }

/* Learn & Explore: clickable feature buttons (popovers) in the red palette */
[data-testid="stPopover"] button {
  border-radius: 12px; background: #FFFFFF; border: 1px solid var(--line);
  border-left: 5px solid var(--pcos-teal); color: var(--pcos-purple-dark); font-weight: 700;
  box-shadow: 0 4px 12px rgba(198,40,40,0.07); transition: all 0.2s ease;
}
[data-testid="stPopover"] button:hover {
  background: var(--pcos-teal-soft); border-color: var(--pcos-teal); color: var(--pcos-teal-dark);
  box-shadow: 0 8px 18px rgba(198,40,40,0.16);
}
.feat-info { font-size: 1.05rem; line-height: 1.5; color: var(--ink); }
.feat-info b { color: var(--pcos-teal-dark); }
.feat-info small { color: var(--muted); }


/* ---- Always-readable expanders and popovers (fixes black box / invisible text on click) ---- */
:root { color-scheme: light; }
[data-testid="stExpander"] {
  background: #FFFFFF !important; border: 1px solid var(--line) !important;
  border-left: 5px solid var(--pcos-teal) !important; border-radius: 12px !important;
  box-shadow: 0 4px 12px rgba(198,40,40,0.07); overflow: hidden;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] details[open],
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:active,
[data-testid="stExpander"] details[open] > summary {
  background: #FFFFFF !important; color: var(--ink) !important; outline: none !important;
}
[data-testid="stExpander"] summary:hover { background: var(--pcos-teal-soft) !important; }
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {
  color: var(--pcos-purple-dark) !important; font-weight: 700; font-size: 1.1rem;
}
[data-testid="stExpander"] summary svg { fill: var(--pcos-teal) !important; color: var(--pcos-teal) !important; }
[data-testid="stExpanderDetails"] { background: #FFFFFF !important; color: var(--ink) !important; }
[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] li,
[data-testid="stExpanderDetails"] label,
[data-testid="stExpanderDetails"] span,
[data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"] { color: var(--ink) !important; }
[data-testid="stExpanderDetails"] .chip { color: var(--pcos-purple-dark) !important; }

/* Popover (click-to-read feature info) */
[data-testid="stPopoverBody"],
div[data-baseweb="popover"] > div {
  background: #FFFFFF !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 12px !important;
}
[data-testid="stPopoverBody"] *, .feat-info, .feat-info * { color: var(--ink) !important; }
[data-testid="stPopoverBody"] .feat-info b { color: var(--pcos-teal-dark) !important; }
[data-testid="stPopoverBody"] .feat-info small { color: var(--muted) !important; }
[data-testid="stPopover"] button, [data-testid="stPopover"] button p,
[data-testid="stPopover"] button:focus, [data-testid="stPopover"] button:active {
  background: #FFFFFF !important; color: var(--pcos-purple-dark) !important;
}
[data-testid="stPopover"] button:hover, [data-testid="stPopover"] button:hover p {
  background: var(--pcos-teal-soft) !important; color: var(--pcos-teal-dark) !important;
}
[data-testid="stPopover"] button svg { fill: var(--pcos-teal) !important; }


/* Intro box under the welcome box */
.intro-box {
  position: relative;
  width: 100vw;
  margin: 5rem 0 4rem calc(50% - 50vw);   /* stretch edge to edge, like the nav bar */
  padding: 1.3rem 2rem 1.5rem 2rem; text-align: center; box-sizing: border-box;
  background: linear-gradient(180deg, #FFFFFF 0%, var(--pcos-purple-soft) 100%);
  border: none;
  box-shadow: 0 8px 22px rgba(198,40,40,0.10);
}
/* stop the 100vw band from creating a sideways scrollbar */
[data-testid="stMain"], [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }
.intro-box .intro-title {
  font-size: 2rem; font-weight: 800; letter-spacing: .03em;
  color: var(--pcos-teal-dark); line-height: 1.2; margin: 0;
}
.intro-box .intro-tagline {
  font-size: 1.6rem; font-style: italic; font-weight: 700;
  color: var(--pcos-teal); margin: 0.4rem 0 0.7rem 0; line-height: 1.25;
}
.intro-box .intro-divider {
  width: 90px; height: 2px; margin: 0 auto 0.8rem auto;
  background: linear-gradient(90deg, transparent, var(--pcos-teal), transparent);
}
.intro-box .intro-text {
  font-size: 1.2rem; line-height: 1.6; color: var(--ink);
  max-width: 880px; margin: 0 auto;
}


/* Spacing between home-page elements */
.block-container h3 { padding: 0.4rem 0 1rem 0 !important; }
[data-testid="stCaptionContainer"] { margin-bottom: 1.2rem; }
.gap { width: 100%; }

/* Module cards: balanced white + soft pink, red kept as accent */
.module-card {
  position: relative; overflow: hidden; min-height: 210px; box-sizing: border-box;
  margin-bottom: 0.8rem;
  background: #FFFFFF; border: 1px solid var(--line);
  border-top: 4px solid var(--pcos-teal); border-radius: 6px;
  box-shadow: 0 6px 16px rgba(198,40,40,0.08);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.module-card:hover { transform: translateY(-3px); box-shadow: 0 12px 26px rgba(198,40,40,0.16); }
.module-card .module-head {
  display: flex; justify-content: space-between; align-items: center; gap: 0.8rem;
  padding: 0.55rem 1.2rem;
  background: var(--pcos-teal-soft); border-bottom: 1px solid var(--line);
}
.module-card .module-head .num { font-size: 1.8rem; font-weight: 800; color: var(--pcos-teal); line-height: 1; }
.module-card .module-head .label {
  font-size: 0.9rem; font-weight: 800; letter-spacing: .07em; text-transform: uppercase;
  color: var(--pcos-purple-dark); text-align: right; line-height: 1.2;
}
.module-card .module-body { padding: 1rem 1.2rem 1.2rem 1.2rem; }
.module-card .module-body h4 { margin: 0 0 0.5rem 0; font-size: 1.35rem; line-height: 1.25; color: var(--pcos-purple-dark); }
.module-card .module-body p { margin: 0; font-size: 1.02rem; line-height: 1.5; color: var(--ink); }

/* Animated PCOS-themed background */
.block-container { position: relative; z-index: 1; }
div[data-testid="stElementContainer"]:has(.bg-anim) { height: 0 !important; margin: 0 !important; min-height: 0 !important; }
.bg-anim { position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: -1; }
@keyframes floatUp {
  0%   { transform: translate3d(0, 0, 0) rotate(0deg); opacity: 0; }
  10%  { opacity: .55; }
  50%  { transform: translate3d(35px, -60vh, 0) rotate(180deg); }
  90%  { opacity: .55; }
  100% { transform: translate3d(-20px, -120vh, 0) rotate(360deg); opacity: 0; }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 0%, 100% { opacity: .35; } 50% { opacity: 1; } }
@keyframes wave { to { background-position-x: -300px; } }
.bg-anim .bubble {
  position: absolute; bottom: -90px; border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.95), rgba(229,57,53,.22) 60%, rgba(198,40,40,.14));
  border: 1px solid rgba(198,40,40,.22);
  animation: floatUp linear infinite;
}
.bg-anim .ribbon {
  position: absolute; bottom: -70px; opacity: 0;
  filter: hue-rotate(-50deg) saturate(1.6);
  animation: floatUp linear infinite;
}
.bg-anim .ovary { position: absolute; width: 220px; height: 220px; opacity: .55; animation: spin 70s linear infinite; }
.bg-anim .ovary.o1 { top: 14%; left: -70px; }
.bg-anim .ovary.o2 { bottom: 10%; right: -80px; width: 280px; height: 280px; animation-direction: reverse; animation-duration: 90s; }
.bg-anim .ovary .core {
  position: absolute; inset: 28%; border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #FFFFFF, rgba(229,57,53,.22));
  border: 2px solid rgba(198,40,40,.25);
}
.bg-anim .ovary i {
  position: absolute; left: 50%; top: 50%; width: 18px; height: 18px; margin: -9px;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #FFFFFF, rgba(198,40,40,.35));
  border: 1px solid rgba(198,40,40,.3);
  transform: rotate(var(--a)) translateX(var(--r));
  animation: pulse 4s ease-in-out infinite; animation-delay: var(--d);
}
.bg-anim .wave {
  position: absolute; left: 0; right: 0; bottom: 0; height: 80px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='80'%3E%3Cpath d='M0 40 Q75 0 150 40 T300 40' fill='none' stroke='rgba(198,40,40,0.28)' stroke-width='2'/%3E%3C/svg%3E");
  background-repeat: repeat-x; background-position: 0 100%;
  animation: wave 14s linear infinite;
}
@media (prefers-reduced-motion: reduce) { .bg-anim * { animation: none !important; } }

/* Horizontal module cards (staircase layout) */
.module-card.row {
  display: flex; align-items: stretch; min-height: 0; margin-bottom: 0;
  border-top: 1px solid var(--line); border-left: 6px solid var(--pcos-teal);
}
.module-card.row .module-head {
  flex: 0 0 210px; flex-direction: column; align-items: flex-start; justify-content: center;
  gap: 0.4rem; padding: 1rem 1.4rem;
  border-bottom: none; border-right: 1px solid var(--line);
}
.module-card.row .module-head .num { font-size: 2.4rem; }
.module-card.row .module-head .label { text-align: left; }
.module-card.row .module-body {
  flex: 1; padding: 1.1rem 1.6rem; display: flex; flex-direction: column; justify-content: center;
}


/* =====================================================================
   MY HEALTH PROFILE PAGE
   ===================================================================== */

/* Form widgets restyled to the red palette */
[data-testid="stWidgetLabel"] p { color: var(--pcos-purple-dark); font-weight: 700; font-size: 1.05rem; }
[data-testid="stTextInput"], [data-testid="stSelectbox"] { margin-bottom: 0.5rem; }
[data-testid="stWidgetLabel"] { margin-bottom: 0.35rem; }

/* Text inputs: soft white fields with a red accent edge */
div[data-baseweb="input"], div[data-baseweb="base-input"] { background: #FFFFFF !important; }
div[data-baseweb="input"] {
  border: 1.5px solid var(--line) !important; border-left: 5px solid var(--pcos-teal) !important;
  border-radius: 12px !important; box-shadow: 0 4px 12px rgba(198,40,40,0.07);
  transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
}
div[data-baseweb="input"]:hover {
  box-shadow: 0 8px 18px rgba(198,40,40,0.16); transform: translateY(-1px);
}
div[data-baseweb="input"]:focus-within {
  border-color: var(--pcos-teal) !important; box-shadow: 0 0 0 4px rgba(198,40,40,0.16) !important;
}
div[data-baseweb="input"] input {
  background: transparent !important; color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important; caret-color: var(--pcos-teal);
  font-size: 1.12rem; font-weight: 700; padding: 0.8rem 1rem !important;
}
div[data-baseweb="input"] input::placeholder {
  color: #C99A9A !important; -webkit-text-fill-color: #C99A9A !important;
  font-style: italic; font-weight: 400; opacity: 1;
}

/* Dropdowns: same look as the text inputs */
div[data-baseweb="select"] > div {
  background: #FFFFFF !important; border: 1.5px solid var(--line) !important;
  border-left: 5px solid var(--pcos-teal) !important; border-radius: 12px !important;
  min-height: 3.1rem; box-shadow: 0 4px 12px rgba(198,40,40,0.07);
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}
div[data-baseweb="select"] > div:hover { box-shadow: 0 8px 18px rgba(198,40,40,0.16); transform: translateY(-1px); }
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--pcos-teal) !important; box-shadow: 0 0 0 4px rgba(198,40,40,0.16) !important;
}
div[data-baseweb="select"] div, div[data-baseweb="select"] span {
  color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; font-weight: 700;
}
div[data-baseweb="select"] svg { fill: var(--pcos-teal) !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {
  background: #FFFFFF !important; color: var(--ink) !important;
}
div[data-baseweb="popover"] li:hover { background: var(--pcos-teal-soft) !important; }
.field-warn {
  margin: -0.4rem 0 0.8rem 0; padding: 0.35rem 0.7rem; border-radius: 8px;
  background: #FFF4E5; border: 1px solid #F5C987; color: #8A5300; font-size: 0.92rem;
}

/* Hero with progress ring */
.ph-hero {
  display: grid; grid-template-columns: 1.5fr 1fr; gap: 2rem; align-items: center;
  padding: 2.4rem 2.6rem; border-radius: 24px; margin-bottom: 2.4rem;
  background: linear-gradient(120deg, var(--pcos-teal-soft) 0%, #FFFFFF 55%, var(--pcos-purple-soft) 100%);
  border: 1px solid var(--line); border-left: 8px solid var(--pcos-teal);
  box-shadow: 0 12px 32px rgba(198,40,40,0.12);
}
.ph-hero h2 { margin: 0.4rem 0 0.7rem 0; font-size: 2.9rem; line-height: 1.1; }
.ph-hero p { color: var(--muted); font-size: 1.18rem; line-height: 1.55; margin: 0; max-width: 640px; }
.ph-right { display: flex; align-items: center; justify-content: center; gap: 1.6rem; flex-wrap: wrap; }
.ph-ring {
  --p: 0; position: relative; width: 132px; height: 132px; border-radius: 50%;
  display: grid; place-items: center;
  background: conic-gradient(var(--pcos-teal) calc(var(--p) * 1%), var(--line) 0);
  box-shadow: 0 8px 20px rgba(198,40,40,0.18);
}
.ph-ring::before { content: ""; position: absolute; inset: 12px; border-radius: 50%; background: #FFFFFF; }
.ph-ring span { position: relative; font-size: 1.9rem; font-weight: 800; color: var(--pcos-teal-dark); }
.ph-stats { display: flex; flex-direction: column; gap: 0.6rem; }
.ph-stat {
  background: #FFFFFF; border: 1px solid var(--line); border-left: 4px solid var(--pcos-teal);
  border-radius: 10px; padding: 0.4rem 1rem; min-width: 190px;
}
.ph-stat b { display: block; font-size: 1.35rem; color: var(--pcos-teal-dark); line-height: 1.2; }
.ph-stat small {
  color: var(--muted); letter-spacing: .07em; text-transform: uppercase; font-size: 0.72rem; font-weight: 800;
}

/* Section selector cards */
.sec-card {
  display: flex; flex-direction: column; min-height: 196px; box-sizing: border-box; overflow: hidden;
  margin-bottom: 0.6rem; background: #FFFFFF; border: 1px solid var(--line);
  border-top: 4px solid var(--pcos-teal); border-radius: 6px;
  box-shadow: 0 6px 16px rgba(198,40,40,0.08);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.sec-card:hover { transform: translateY(-3px); box-shadow: 0 12px 26px rgba(198,40,40,0.16); }
.sec-card .sec-head {
  display: flex; justify-content: space-between; align-items: center; gap: 0.6rem;
  padding: 0.5rem 1.1rem; background: var(--pcos-teal-soft); border-bottom: 1px solid var(--line);
}
.sec-card .sec-num { font-size: 1.8rem; font-weight: 800; color: var(--pcos-teal); line-height: 1; }
.sec-card .sec-state {
  font-size: 0.8rem; font-weight: 800; letter-spacing: .07em; text-transform: uppercase;
  color: var(--pcos-purple-dark);
}
.sec-card .sec-body { padding: 0.9rem 1.1rem 1.1rem 1.1rem; flex: 1; display: flex; flex-direction: column; }
.sec-card h4 { margin: 0 0 0.35rem 0; font-size: 1.3rem; line-height: 1.25; color: var(--pcos-purple-dark); }
.sec-card p { margin: 0 0 0.9rem 0; font-size: 1.02rem; line-height: 1.4; color: var(--ink); flex: 1; }
.sec-bar, .rev-bar { height: 8px; background: var(--line); border-radius: 99px; overflow: hidden; }
.sec-bar i, .rev-bar i {
  display: block; height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, var(--pcos-teal-dark), var(--pcos-purple));
}
.sec-card.active {
  border-top-width: 6px; background: linear-gradient(180deg, #FFFFFF 0%, var(--pcos-teal-soft) 100%);
  box-shadow: 0 14px 30px rgba(198,40,40,0.24); border-color: var(--pcos-teal);
}

/* Workspace header */
.ws-head {
  display: flex; align-items: center; gap: 1.4rem; padding: 1.1rem 1.6rem; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, var(--pcos-teal-dark), var(--pcos-teal) 55%, var(--pcos-purple-dark));
  border-radius: 8px; box-shadow: 0 10px 24px rgba(142,0,0,0.25); color: #FFFFFF;
}
.ws-head .ws-num { font-size: 3rem; font-weight: 900; line-height: 1; color: rgba(255,255,255,0.9); }
.ws-head .ws-text { flex: 1; }
.ws-head h3 { margin: 0 !important; padding: 0 !important; color: #FFFFFF !important; font-size: 1.75rem; }
.ws-head p { margin: 0.2rem 0 0 0; color: rgba(255,255,255,0.88); font-size: 1.05rem; }
.ws-head .ws-count {
  text-align: center; background: #FFFFFF; color: var(--pcos-teal-dark); border-radius: 10px;
  padding: 0.4rem 1.1rem; font-size: 1.6rem; font-weight: 800; line-height: 1.1;
}
.ws-head .ws-count small {
  display: block; font-size: 0.68rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
}

/* Review panel */
.rev-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin-bottom: 1.2rem; }
.rev-tile {
  background: #FFFFFF; border: 1px solid var(--line); border-top: 4px solid var(--pcos-teal);
  border-radius: 8px; padding: 0.9rem 1rem; text-align: center; box-shadow: 0 6px 16px rgba(198,40,40,0.08);
}
.rev-tile b { display: block; font-size: 2.2rem; color: var(--pcos-teal-dark); line-height: 1.1; }
.rev-tile small { color: var(--muted); letter-spacing: .07em; text-transform: uppercase; font-size: 0.74rem; font-weight: 800; }
.rev-row {
  display: grid; grid-template-columns: 1.7fr 1.2fr 60px; gap: 0.9rem; align-items: center;
  padding: 0.55rem 0.2rem; border-bottom: 1px dashed var(--line);
}
.rev-row .rev-name { color: var(--pcos-purple-dark); font-weight: 700; font-size: 1.05rem; }
.rev-row .rev-count { text-align: right; color: var(--pcos-teal-dark); font-weight: 800; }
.chip {
  display: inline-block; margin: 0.15rem 0.25rem 0.15rem 0; padding: 0.15rem 0.65rem; border-radius: 99px;
  background: var(--pcos-purple-soft); border: 1px solid var(--line); color: var(--pcos-purple-dark); font-size: 0.92rem;
}
.tip-box {
  margin: 1rem 0; padding: 1rem 1.2rem; border-radius: 10px;
  background: linear-gradient(120deg, var(--pcos-teal-soft), #FFFFFF);
  border: 1px solid var(--line); border-left: 6px solid var(--pcos-teal);
  color: var(--ink); font-size: 1.05rem; line-height: 1.5;
}

/* Results page: verdict banner */
.verdict {
  margin: 1rem 0 1.4rem 0; padding: 1.4rem 1.8rem; border-radius: 12px; background: #FFFFFF;
  border: 1px solid var(--line); border-left: 10px solid var(--pcos-teal);
  box-shadow: 0 10px 24px rgba(198,40,40,0.12);
}
.verdict small {
  color: var(--muted); letter-spacing: .08em; text-transform: uppercase; font-size: 0.78rem; font-weight: 800;
}
.verdict h2 { margin: 0.2rem 0 0.4rem 0; font-size: 2.3rem; }
.verdict.inconclusive { border-left-color: #E0A100; background: #FFFBEA; }
.verdict.inconclusive h2 { color: #8A5300; }
.verdict.positive { border-left-color: #C62828; background: #FFF1F1; }
.verdict.negative { border-left-color: #2E7D32; background: #F1FAF2; }
.verdict.negative h2 { color: #1B5E20; }
.verdict ul { margin: 0.4rem 0 0 1.1rem; padding: 0; color: var(--ink); font-size: 1.05rem; }

@media (max-width: 900px) {
  .ph-hero { grid-template-columns: 1fr; }
  .rev-tiles { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Animated background (pure CSS, decorative only)
# ---------------------------------------------------------------------------
def build_background():
    import random
    rnd = random.Random(7)
    parts = ['<div class="bg-anim">']
    for _ in range(16):  # soft follicle-like bubbles
        size = rnd.randint(18, 70)
        parts.append(
            f'<span class="bubble" style="left:{rnd.randint(0, 96)}%;width:{size}px;height:{size}px;'
            f'animation-duration:{rnd.randint(18, 40)}s;animation-delay:-{rnd.randint(0, 40)}s"></span>'
        )
    for _ in range(9):  # awareness ribbons
        parts.append(
            f'<span class="ribbon" style="left:{rnd.randint(2, 95)}%;font-size:{rnd.randint(22, 44)}px;'
            f'animation-duration:{rnd.randint(26, 50)}s;animation-delay:-{rnd.randint(0, 50)}s">\U0001F397\ufe0f</span>'
        )
    for name, radius in (("o1", 80), ("o2", 105)):  # "string of pearls" ovary motif
        dots = "".join(
            f'<i style="--a:{k * 36}deg;--r:{radius}px;--d:{k * 0.4:.1f}s"></i>' for k in range(10)
        )
        parts.append(f'<div class="ovary {name}"><b class="core"></b>{dots}</div>')
    parts.append('<div class="wave"></div></div>')
    return "".join(parts)


st.markdown(build_background(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------
# Five research groups (used by the Data Lab experiments and Learn & Explore)
GROUPS = {
    "Physical and metabolic": [
        "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI", "Pulse rate(bpm)",
        "RR (breaths/min)", "Hb(g/dl)", "Hip(inch)", "Waist(inch)",
        "Waist:Hip Ratio", "RBS(mg/dl)", "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)",
    ],
    "Hormonal and biochemical": [
        "I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)", "FSH(mIU/mL)",
        "LH(mIU/mL)", "FSH/LH", "TSH (mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)",
        "Vit D3 (ng/mL)", "PRG(ng/mL)",
    ],
    "Menstrual and reproductive history": [
        "Cycle(R/I)", "Cycle length(days)", "Marraige Status (Yrs)",
        "Pregnant(Y/N)", "No. of aborptions",
    ],
    "Symptoms and lifestyle": [
        "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
        "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)",
    ],
    "Ovarian measurements": [
        "Follicle No. (L)", "Follicle No. (R)", "Avg. F size (L) (mm)",
        "Avg. F size (R) (mm)", "Endometrium (mm)",
    ],
}

# Six user-facing sections for the My Health Profile page.
# Same 40 model features as GROUPS, just organised more clearly for data entry.
PROFILE_SECTIONS = [
    {
        "num": "01", "title": "Basic Health & Measurements",
        "sub": "Age, weight, height, BMI and vital signs",
        "features": [
            "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI", "Pulse rate(bpm)",
            "RR (breaths/min)", "Hb(g/dl)", "Hip(inch)", "Waist(inch)",
            "Waist:Hip Ratio", "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)",
        ],
    },
    {
        "num": "02", "title": "Menstrual & Reproductive History",
        "sub": "pregnancy and reproductive history",
        "features": [
            "Cycle(R/I)","Cycle length(days)", "Marraige Status (Yrs)",
            "Pregnant(Y/N)", "No. of aborptions",
            "I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",
        ],
    },
    {
        "num": "03", "title": "Hormonal & Biochemical Markers",
        "sub": "Hormone levels and laboratory measurements",
        "features": [
            "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "TSH (mIU/L)", "AMH(ng/mL)",
            "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)", "RBS(mg/dl)",
        ],
    },
    {
        "num": "04", "title": "Symptoms",
        "sub": "Hair growth, hair loss, skin changes and pimples",
        "features": [
            "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
            "Hair loss(Y/N)", "Pimples(Y/N)",
        ],
    },
    {
        "num": "05", "title": "Lifestyle",
        "sub": "Fast-food consumption and regular exercise",
        "features": ["Fast food (Y/N)", "Reg.Exercise(Y/N)"],
    },
    {
        "num": "06", "title": "Ovarian & Ultrasound Measurements",
        "sub": "Follicle counts, follicle sizes and endometrium",
        "features": [
            "Follicle No. (L)", "Follicle No. (R)", "Avg. F size (L) (mm)",
            "Avg. F size (R) (mm)", "Endometrium (mm)",
        ],
    },
]
ALL_FEATURES = [f for s in PROFILE_SECTIONS for f in s["features"]]

# feature -> (display label, unit, soft minimum, soft maximum)
FIELDS = {
    "Age (yrs)": ("Age", "years", 10, 80),
    "Weight (Kg)": ("Weight", "kg", 25, 200),
    "Height(Cm)": ("Height", "cm", 120, 210),
    "BMI": ("BMI", "kg/m²", 10, 60),
    "Pulse rate(bpm)": ("Pulse rate", "bpm", 40, 200),
    "RR (breaths/min)": ("Respiratory rate", "breaths/min", 8, 40),
    "Hb(g/dl)": ("Haemoglobin", "g/dL", 5, 20),
    "Hip(inch)": ("Hip circumference", "inch", 20, 70),
    "Waist(inch)": ("Waist circumference", "inch", 20, 70),
    "Waist:Hip Ratio": ("Waist : hip ratio", "ratio", 0.5, 1.5),
    "BP _Systolic (mmHg)": ("Systolic blood pressure", "mmHg", 70, 220),
    "BP _Diastolic (mmHg)": ("Diastolic blood pressure", "mmHg", 40, 140),
    "Cycle(R/I)": ("Cycle pattern", "", None, None),
    "Cycle length(days)": ("Cycle length", "days", 1, 90),
    "Marraige Status (Yrs)": ("Years of marriage", "years", 0, 50),
    "Pregnant(Y/N)": ("Currently pregnant", "", None, None),
    "No. of aborptions": ("Number of abortions", "count", 0, 10),
    "I   beta-HCG(mIU/mL)": ("β-HCG (I)", "mIU/mL", 0, None),
    "II    beta-HCG(mIU/mL)": ("β-HCG (II)", "mIU/mL", 0, None),
    "FSH(mIU/mL)": ("FSH", "mIU/mL", 0, 200),
    "LH(mIU/mL)": ("LH", "mIU/mL", 0, 200),
    "FSH/LH": ("FSH : LH ratio", "ratio", 0, 50),
    "TSH (mIU/L)": ("TSH", "mIU/L", 0, 100),
    "AMH(ng/mL)": ("AMH", "ng/mL", 0, 100),
    "PRL(ng/mL)": ("Prolactin", "ng/mL", 0, 500),
    "Vit D3 (ng/mL)": ("Vitamin D3", "ng/mL", 0, 200),
    "PRG(ng/mL)": ("Progesterone", "ng/mL", 0, 100),
    "RBS(mg/dl)": ("Random blood sugar", "mg/dL", 40, 600),
    "Weight gain(Y/N)": ("Weight gain", "", None, None),
    "hair growth(Y/N)": ("Excess hair growth", "", None, None),
    "Skin darkening (Y/N)": ("Skin darkening", "", None, None),
    "Hair loss(Y/N)": ("Hair loss", "", None, None),
    "Pimples(Y/N)": ("Pimples / acne", "", None, None),
    "Fast food (Y/N)": ("Regular fast-food consumption", "", None, None),
    "Reg.Exercise(Y/N)": ("Regular exercise", "", None, None),
    "Follicle No. (L)": ("Follicle count, left ovary", "count", 0, 60),
    "Follicle No. (R)": ("Follicle count, right ovary", "count", 0, 60),
    "Avg. F size (L) (mm)": ("Average follicle size, left", "mm", 0, 40),
    "Avg. F size (R) (mm)": ("Average follicle size, right", "mm", 0, 40),
    "Endometrium (mm)": ("Endometrium thickness", "mm", 0, 30),
}

BINARY_FEATURES = {
    "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)",
    "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)",
    "Fast food (Y/N)", "Reg.Exercise(Y/N)",
}

# One-line plain-language explanation of every feature (used in Learn & Explore)
FEATURE_INFO = {
    # Physical and metabolic
    "Age (yrs)": "Your age in years; PCOS is most often identified during the reproductive years.",
    "Weight (Kg)": "Your body weight in kilograms; it helps assess body composition and metabolic health.",
    "Height(Cm)": "Your standing height in centimetres; it is used together with weight to calculate BMI.",
    "BMI": "Body Mass Index (weight divided by height squared), a quick indicator of whether weight is in a healthy range.",
    "Pulse rate(bpm)": "How many times your heart beats per minute at rest, showing general cardiovascular health.",
    "RR (breaths/min)": "How many breaths you take per minute at rest, a basic vital sign of respiratory health.",
    "Hb(g/dl)": "Haemoglobin level in the blood; low values signal anaemia, which can be linked to heavy or long periods.",
    "Hip(inch)": "Hip circumference in inches; used with waist size to see how body fat is distributed.",
    "Waist(inch)": "Waist circumference in inches; a larger waist can point to central fat and insulin resistance.",
    "Waist:Hip Ratio": "Waist measurement divided by hip measurement; higher values mean more fat is stored around the abdomen.",
    "RBS(mg/dl)": "Random blood sugar, the glucose level in your blood at any time of day, which helps screen for insulin problems.",
    "BP _Systolic (mmHg)": "The upper blood pressure number, measuring the pressure in your arteries when the heart beats.",
    "BP _Diastolic (mmHg)": "The lower blood pressure number, measuring the pressure in your arteries when the heart rests between beats.",
    # Hormonal and biochemical
    "I   beta-HCG(mIU/mL)": "The first pregnancy-hormone (β-HCG) reading; it helps confirm or rule out pregnancy.",
    "II    beta-HCG(mIU/mL)": "A second β-HCG reading taken later, used to check how the pregnancy hormone level is changing.",
    "FSH(mIU/mL)": "Follicle-stimulating hormone, which makes the ovaries grow and mature eggs in follicles.",
    "LH(mIU/mL)": "Luteinising hormone, which triggers ovulation; it is often raised in PCOS.",
    "FSH/LH": "The ratio of FSH to LH; a lower ratio is a commonly noted hormonal pattern in PCOS.",
    "TSH (mIU/L)": "Thyroid-stimulating hormone, which shows how well the thyroid works; thyroid problems can mimic PCOS symptoms.",
    "AMH(ng/mL)": "Anti-Müllerian hormone, made by small ovarian follicles and an indicator of ovarian reserve; it is often high in PCOS.",
    "PRL(ng/mL)": "Prolactin, the milk-producing hormone; high levels can disturb periods and ovulation.",
    "Vit D3 (ng/mL)": "Vitamin D3 level in the blood; low levels are commonly seen in PCOS and relate to metabolism and hormones.",
    "PRG(ng/mL)": "Progesterone, the hormone released after ovulation; low levels suggest ovulation may not be happening.",
    # Menstrual and reproductive history
    "Cycle(R/I)": "Whether your periods come at a predictable interval (regular) or not (irregular); irregular cycles are a key sign of PCOS.",
    "Cycle length(days)": "The number of days of your period (bleeding), which helps describe your menstrual pattern.",
    "Marraige Status (Yrs)": "How many years you have been married, which gives context to reproductive history and fertility.",
    "Pregnant(Y/N)": "Whether you are currently pregnant, which changes many hormone readings.",
    "No. of aborptions": "The number of pregnancy losses or abortions you have had, part of your reproductive history.",
    # Symptoms and lifestyle
    "Weight gain(Y/N)": "Whether you have recently gained weight, often linked to insulin resistance in PCOS.",
    "hair growth(Y/N)": "Whether you have excess coarse hair on the face or body (hirsutism), caused by raised male hormones.",
    "Skin darkening (Y/N)": "Whether dark, velvety patches appear on the neck, armpits or groin, a sign of insulin resistance.",
    "Hair loss(Y/N)": "Whether you have thinning or hair fall from the scalp, which can be driven by excess male hormones.",
    "Pimples(Y/N)": "Whether you have frequent acne, which can be caused by excess male hormones.",
    "Fast food (Y/N)": "Whether you regularly eat fast food; diet quality affects weight, insulin and hormones.",
    "Reg.Exercise(Y/N)": "Whether you exercise regularly; physical activity improves insulin sensitivity and hormone balance.",
    # Ovarian measurements
    "Follicle No. (L)": "The number of small follicles (fluid-filled sacs holding eggs) counted in the left ovary on ultrasound.",
    "Follicle No. (R)": "The number of small follicles counted in the right ovary on ultrasound; high counts are a hallmark of PCOS.",
    "Avg. F size (L) (mm)": "The average diameter of follicles in the left ovary, measured in millimetres on ultrasound.",
    "Avg. F size (R) (mm)": "The average diameter of follicles in the right ovary, measured in millimetres on ultrasound.",
    "Endometrium (mm)": "The thickness of the uterine lining on ultrasound, which reflects hormonal activity and ovulation.",
}

TOP_IMPORTANCE = [
    ("Follicle No. (R)", 0.0656, "Ovarian measurements"),
    ("hair growth(Y/N)", 0.0139, "Symptoms and lifestyle"),
    ("Weight gain(Y/N)", 0.0095, "Symptoms and lifestyle"),
    ("Follicle No. (L)", 0.0094, "Ovarian measurements"),
    ("Cycle(R/I)", 0.0069, "Menstrual and reproductive history"),
    ("AMH(ng/mL)", 0.0051, "Hormonal and biochemical"),
    ("Skin darkening (Y/N)", 0.0050, "Symptoms and lifestyle"),
    ("FSH/LH", 0.0035, "Hormonal and biochemical"),
    ("Pimples(Y/N)", 0.0027, "Symptoms and lifestyle"),
    ("LH(mIU/mL)", 0.0022, "Hormonal and biochemical"),
]

MISSING_RATE_RESULTS = pd.DataFrame([
    {"Missing values simulated": "0%",  "Accuracy": .9083, "Sensitivity": .8333, "F1-score": .8571, "ROC-AUC": .9425},
    {"Missing values simulated": "10%", "Accuracy": .9000, "Sensitivity": .7944, "F1-score": .8399, "ROC-AUC": .9369},
    {"Missing values simulated": "20%", "Accuracy": .8945, "Sensitivity": .7444, "F1-score": .8225, "ROC-AUC": .9321},
    {"Missing values simulated": "30%", "Accuracy": .8569, "Sensitivity": .6306, "F1-score": .7430, "ROC-AUC": .9078},
])

MISSING_GROUP_RESULTS = pd.DataFrame([
    {"Scenario": "All five groups", "Groups available": 5, "Features": 40, "Accuracy": .9083, "Sensitivity": .8333, "ROC-AUC": .9425},
    {"Scenario": "Without physical and metabolic", "Groups available": 4, "Features": 27, "Accuracy": .8991, "Sensitivity": .8333, "ROC-AUC": .9543},
    {"Scenario": "Without menstrual and reproductive history", "Groups available": 4, "Features": 35, "Accuracy": .9174, "Sensitivity": .8611, "ROC-AUC": .9382},
    {"Scenario": "Without hormonal and biochemical", "Groups available": 4, "Features": 30, "Accuracy": .9083, "Sensitivity": .8611, "ROC-AUC": .9315},
    {"Scenario": "Without symptoms and lifestyle", "Groups available": 4, "Features": 33, "Accuracy": .8532, "Sensitivity": .7778, "ROC-AUC": .9146},
    {"Scenario": "Without ovarian measurements", "Groups available": 4, "Features": 35, "Accuracy": .8440, "Sensitivity": .7500, "ROC-AUC": .8838},
])

INDIVIDUAL_GROUP_RESULTS = pd.DataFrame([
    {"Feature group": "Ovarian measurements", "Features": 5, "Accuracy": .8440, "Sensitivity": .7778, "ROC-AUC": .8752},
    {"Feature group": "Symptoms and lifestyle", "Features": 7, "Accuracy": .7431, "Sensitivity": .7778, "ROC-AUC": .8518},
    {"Feature group": "Menstrual and reproductive history", "Features": 5, "Accuracy": .6422, "Sensitivity": .4722, "ROC-AUC": .6657},
    {"Feature group": "Hormonal and biochemical", "Features": 10, "Accuracy": .6514, "Sensitivity": .3611, "ROC-AUC": .6581},
    {"Feature group": "Physical and metabolic", "Features": 13, "Accuracy": .5872, "Sensitivity": .3333, "ROC-AUC": .5660},
])


# ---------------------------------------------------------------------------
# Model + metadata loading
# ---------------------------------------------------------------------------
# The path and file-modified time are passed in so Streamlit's cache is refreshed
# whenever the file moves or changes (otherwise an old "not found" result is reused).
@st.cache_resource
def load_model(path_str, modified):
    path = Path(path_str)
    if not path.exists():
        return None, f"Model file not found at: {path}"
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"Could not load the saved model: {exc}"


@st.cache_data
def load_metadata(path_str, modified):
    path = Path(path_str)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


model, model_error = load_model(str(MODEL_PATH), _mtime(MODEL_PATH))
metadata = load_metadata(str(META_PATH), _mtime(META_PATH))

# Result tables written by 02_results_validation.ipynb (the app still runs if they are missing)
REPORTS_DIR = BASE_DIR / "reports"


def _read_report(name):
    p = REPORTS_DIR / name
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


DEPLOYED_RESULTS = _read_report("deployed_model_missing_results.csv")
STRATEGY_RESULTS = _read_report("B_strategy_summary.csv")
ABSTAIN_RESULTS = _read_report("D_abstention_policy_evaluation.csv")
GROUP_TO_SCENARIO = {
    "Physical and metabolic": "No physical",
    "Hormonal and biochemical": "No hormonal",
    "Menstrual and reproductive history": "No menstrual",
    "Symptoms and lifestyle": "No symptoms",
    "Ovarian measurements": "No ovarian",
}
expected_features = (
    metadata.get("numeric_features", []) + metadata.get("categorical_features", [])
) or [f for feats in GROUPS.values() for f in feats]

# ---------------------------------------------------------------------------
# Model scoring helpers (verified against the saved pipeline:
#   steps = preprocessor -> classifier, classifier = RandomForest, 300 trees)
# ---------------------------------------------------------------------------
# Decision rules for the three-way verdict. These are app design choices,
# NOT results from the notebook. Tune them with cross-validation and say so in the report.
BAND_LOW, BAND_HIGH = 0.35, 0.65     # scores inside this band are "close to the boundary"
MIN_OVARIAN_INPUTS = 3                # of the 5 ovarian/ultrasound inputs
MIN_SYMPTOM_INPUTS = 3                # of the 5 symptom inputs (section 04)


def pos_index(m):
    """Column of predict_proba that corresponds to PCOS = 1."""
    classes = list(getattr(m, "classes_", m[-1].classes_))
    return next((i for i, c in enumerate(classes) if str(c) in ("1", "1.0", "True")), len(classes) - 1)


def fill_derived(X):
    """Knowledge-based imputation: BMI, waist:hip ratio and FSH:LH are exact formulas of other inputs
    (verified in the dataset audit). When the ingredients are present and the value itself is blank,
    calculate it instead of letting a median stand in. Only blank cells are touched."""
    X = X.copy()

    def has(*cols):
        return all(c in X.columns for c in cols)

    if has("BMI", "Weight (Kg)", "Height(Cm)"):
        m = X["BMI"].isna() & X["Weight (Kg)"].notna() & (X["Height(Cm)"] > 0)
        X.loc[m, "BMI"] = X.loc[m, "Weight (Kg)"] / (X.loc[m, "Height(Cm)"] / 100) ** 2
    if has("Waist:Hip Ratio", "Waist(inch)", "Hip(inch)"):
        m = X["Waist:Hip Ratio"].isna() & X["Waist(inch)"].notna() & (X["Hip(inch)"] > 0)
        X.loc[m, "Waist:Hip Ratio"] = X.loc[m, "Waist(inch)"] / X.loc[m, "Hip(inch)"]
    if has("FSH/LH", "FSH(mIU/mL)", "LH(mIU/mL)"):
        m = X["FSH/LH"].isna() & X["FSH(mIU/mL)"].notna() & (X["LH(mIU/mL)"] > 0)
        X.loc[m, "FSH/LH"] = X.loc[m, "FSH(mIU/mL)"] / X.loc[m, "LH(mIU/mL)"]
    return X


def build_X(values, features, fill=True):
    """One-row DataFrame in the training feature order; blanks become NaN for the imputers.
    Derived ratios (BMI, waist:hip, FSH:LH) are calculated first when their ingredients were entered."""
    row = {f: (np.nan if values.get(f) is None else values[f]) for f in features}
    X = pd.DataFrame([row], columns=features).apply(pd.to_numeric, errors="coerce")
    return fill_derived(X) if fill else X


def derived_used(values, features):
    """Which blank inputs were calculated automatically for this profile: [(label, value), ...]."""
    raw = build_X(values, features, fill=False)
    filled = fill_derived(raw)
    out = []
    for f in ("BMI", "Waist:Hip Ratio", "FSH/LH"):
        if f in raw.columns and pd.isna(raw[f].iloc[0]) and pd.notna(filled[f].iloc[0]):
            out.append((FIELDS.get(f, (f,))[0], float(filled[f].iloc[0])))
    return out


def score_only(m, X):
    return float(m.predict_proba(X)[0][pos_index(m)])


def tree_votes(m, X):
    """How many of the forest's trees vote for PCOS (probability > 0.5 for class 1)."""
    pi = pos_index(m)
    rf = m[-1]
    Xt = m[:-1].transform(X)
    p = np.array([t.predict_proba(Xt)[0][pi] for t in rf.estimators_])
    return int((p > 0.5).sum()), len(rf.estimators_)


def explain_row(m, values, features):
    """Per-person effect of each entered value: score(with it) - score(with it blanked)."""
    base = score_only(m, build_X(values, features))
    rows = []
    for f in features:
        if values.get(f) is None:
            continue
        s = score_only(m, build_X({**values, f: None}, features))
        rows.append((FIELDS.get(f, (f,))[0], base - s))
    df = pd.DataFrame(rows, columns=["Feature", "Effect on score"])
    if df.empty:
        return df
    return df.reindex(df["Effect on score"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def make_verdict(score, values):
    ov = PROFILE_SECTIONS[5]["features"]
    sy = PROFILE_SECTIONS[3]["features"]
    ov_got = sum(values.get(f) is not None for f in ov)
    sy_got = sum(values.get(f) is not None for f in sy)
    reasons = []
    if ov_got < MIN_OVARIAN_INPUTS:
        reasons.append(
            f"Only {ov_got}/{len(ov)} ovarian/ultrasound inputs were provided. This is the group whose "
            "removal hurt accuracy most in the Data Lab."
        )
    if sy_got < MIN_SYMPTOM_INPUTS:
        reasons.append(f"Only {sy_got}/{len(sy)} symptom inputs were provided.")
    if BAND_LOW <= score <= BAND_HIGH:
        reasons.append(
            f"The model score ({score:.1%}) is close to the decision boundary "
            f"({BAND_LOW:.0%} to {BAND_HIGH:.0%})."
        )
    if reasons:
        return "Inconclusive", "inconclusive", reasons
    if score >= 0.5:
        return "PCOS-like pattern", "positive", []
    return "Not PCOS-like pattern", "negative", []


def group_coverage(values):
    """Provided count per research group + Data Lab accuracy when that group is missing."""
    acc = MISSING_GROUP_RESULTS.set_index("Scenario")["Accuracy"]
    dep = (DEPLOYED_RESULTS.set_index("Scenario")["Accuracy"] if DEPLOYED_RESULTS is not None else None)
    col = ("Held-out accuracy of this model if the group is missing" if dep is not None
           else "Data Lab accuracy if this group is missing (model retrained without it)")
    rows = []
    for g, feats in GROUPS.items():
        got = sum(values.get(f) is not None for f in feats)
        if dep is not None:
            sc = GROUP_TO_SCENARIO.get(g)
            val = f"{dep[sc]:.1%}" if sc in dep.index else "n/a"
        else:
            key = "Without " + g.lower()
            val = f"{acc[key]:.1%}" if key in acc else "n/a"
        rows.append({"Feature group": g, "Provided": f"{got}/{len(feats)}", col: val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Session state + navigation
# ---------------------------------------------------------------------------
if "profile_values" not in st.session_state:
    st.session_state.profile_values = {}
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "active_section" not in st.session_state:
    st.session_state.active_section = 0

PAGES = ["Discover", "My Health Profile", "Understand My Results", "Data Lab", "Learn & Explore"]

# Apply a requested page change BEFORE the radio widget is created
if st.session_state.get("next_page") in PAGES:
    st.session_state["page_nav"] = st.session_state.pop("next_page")

# Header: welcome box on the home page, compact title on other pages
_current_page = st.session_state.get("page_nav", PAGES[0])
if _current_page == "Discover":
    st.markdown(
        '<div class="welcome-box">'
        "<h1>PCOS SCOPE</h1>"
        '<div class="centered-divider"></div>'
        '<div class="centered-welcome">Welcome Back : <span>Kuhu Chandna</span></div>'
        "</div>"
        '<div class="intro-box">'
        '<div class="intro-tagline">&ldquo;Look beyond the prediction&rdquo;</div>'
        '<div class="intro-divider"></div>'
        '<div class="intro-text">An explainable machine learning application exploring PCOS '
        "classification, clinical feature contributions, and the impact of incomplete data on "
        "model performance.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="compact-title">PCOS SCOPE</div>', unsafe_allow_html=True)

page = st.radio(
    "Main navigation", PAGES, horizontal=True, key="page_nav", label_visibility="collapsed"
)


def gap(rem=2.5):
    st.markdown(f'<div class="gap" style="height:{rem}rem"></div>', unsafe_allow_html=True)


def hero(eyebrow, title, copy):
    st.markdown(
        f'<div class="hero"><span class="pill">{eyebrow}</span>'
        f"<h2>{title}</h2><p>{copy}</p></div>",
        unsafe_allow_html=True,
    )


def fact_card(kind, badge, title, body):
    # Kept on one line: indented/blank lines would make Markdown render it as code
    return (
        f'<div class="fact-card type-{kind}"><span class="card-badge">{badge}</span>'
        f'<h4>{title}</h4><div class="reveal-text">{body}</div></div>'
    )


def module_card(num, label, tagline, body):
    from html import escape
    return (
        f'<div class="module-card row"><div class="module-head"><span class="num">{num}</span>'
        f'<span class="label">{escape(label)}</span></div>'
        f'<div class="module-body"><h4>{escape(tagline)}</h4><p>{escape(body)}</p></div></div>'
    )


MODULES = [
    ("01", "My Health Profile", "Your data, your starting point.",
     "Enter clinical inputs by feature group and spot what is missing.",
     "Build My Profile", "My Health Profile", "mod_profile"),
    ("02", "Prediction & Insights", "Understand what the model predicts.",
     "See the classification, key features and limitations.",
     "Explore My Results", "Understand My Results", "mod_results"),
    ("03", "Data Lab", "What happens when data is missing?",
     "Test how missing inputs change model performance.",
     "Enter the Data Lab", "Data Lab", "mod_lab"),
    ("04", "Learn & Explore", "Make sense of the science.",
     "Clinical features, ML concepts and research methods.",
     "Start Exploring", "Learn & Explore", "mod_learn"),
]


# ---------------------------------------------------------------------------
# Profile page helpers
# ---------------------------------------------------------------------------
def go_section(index):
    st.session_state.active_section = max(0, min(index, len(PROFILE_SECTIONS) - 1))


def go_results():
    st.session_state["next_page"] = "Understand My Results"


def reset_profile():
    st.session_state.profile_values = {}
    st.session_state.prediction_result = None
    for k in [k for k in st.session_state.keys() if str(k).startswith("input_")]:
        del st.session_state[k]


def section_count(sec):
    values = st.session_state.profile_values
    return sum(values.get(f) is not None for f in sec["features"])


def total_provided():
    values = st.session_state.profile_values
    return sum(values.get(f) is not None for f in ALL_FEATURES)


# Cycle pattern: what the user picks -> the code the model was trained on
# (audit: code 2 = 390 records, code 4 = 150 records, code 5 = a single record, so it is not offered)
CYCLE_OPTIONS = {
    "Regular (periods come at a predictable interval each month)": 2,
    "Irregular (periods are unpredictable, skipped or far apart)": 4,
}


def render_field(feature, col, old):
    """Draw one input in `col`; return its value (None when blank/invalid)."""
    label, unit, lo, hi = FIELDS.get(feature, (feature, "", None, None))

    if feature in BINARY_FEATURES:
        options = ["Not provided", "No", "Yes"]
        idx = 0 if old is None else (2 if int(old) == 1 else 1)
        chosen = col.selectbox(label, options, index=idx, key=f"input_{feature}")
        return None if chosen == "Not provided" else (1 if chosen == "Yes" else 0)

    if feature == "Cycle(R/I)":
        options = ["Not provided"] + list(CYCLE_OPTIONS.keys())
        reverse = {v: k for k, v in CYCLE_OPTIONS.items()}
        idx = 0 if old is None else options.index(reverse.get(int(old), "Not provided"))
        chosen = col.selectbox(
            label, options, index=idx, key=f"input_{feature}",
            help="Choose Regular if your periods usually arrive about the same number of days apart. "
                 "Choose Irregular if the gap between periods varies a lot, or periods are often "
                 "missed or delayed.",
        )
        return CYCLE_OPTIONS.get(chosen)

    shown = f"{label} ({unit})" if unit else label
    if old is None:
        old_text = ""
    else:
        old_text = str(int(old)) if float(old).is_integer() else str(old)
    raw = col.text_input(
        shown, value=old_text,
        placeholder=(f"Enter value in {unit}" if unit else "Enter value"),
        key=f"input_{feature}",
    ).strip()
    if raw == "":
        return None
    try:
        val = float(raw)
    except ValueError:
        col.error("Enter a number or leave blank.")
        return None
    if not np.isfinite(val):
        col.error("Enter a valid number or leave blank.")
        return None
    if val < 0:
        col.error("Value cannot be negative.")
        return None
    if (lo is not None and val < lo) or (hi is not None and val > hi):
        if lo is not None and hi is not None:
            span = f"{lo:g} to {hi:g}"
        elif lo is not None:
            span = f"{lo:g} or above"
        else:
            span = f"up to {hi:g}"
        col.markdown(
            f'<div class="field-warn">Outside the typical entry range ({span} {unit}). '
            "Please double-check.</div>",
            unsafe_allow_html=True,
        )
    return val


def derived_hint(feature, vals):
    """Suggest a calculated value for ratio fields. Never filled in automatically."""
    def g(k):
        return vals.get(k)
    if feature == "BMI" and g("Weight (Kg)") and g("Height(Cm)"):
        return g("Weight (Kg)") / (g("Height(Cm)") / 100) ** 2
    if feature == "Waist:Hip Ratio" and g("Waist(inch)") and g("Hip(inch)"):
        return g("Waist(inch)") / g("Hip(inch)")
    if feature == "FSH/LH" and g("FSH(mIU/mL)") is not None and g("LH(mIU/mL)"):
        return g("FSH(mIU/mL)") / g("LH(mIU/mL)")
    return None


def section_card(sec, done, total, active):
    pct = int(round(100 * done / total)) if total else 0
    state = "Open" if active else ("Complete" if done == total else ("In progress" if done else "Not started"))
    return (
        f'<div class="sec-card{" active" if active else ""}">'
        f'<div class="sec-head"><span class="sec-num">{sec["num"]}</span>'
        f'<span class="sec-state">{state}</span></div>'
        f'<div class="sec-body"><h4>{sec["title"]}</h4><p>{sec["sub"]}</p>'
        f'<div class="sec-bar"><i style="width:{pct}%"></i></div></div></div>'
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "Discover":
    st.markdown('### <span style="color:#C62828;">&#10148;</span>&nbsp;&nbsp;Hover to Reveal: Facts, Myths & Taboos', unsafe_allow_html=True)
    st.caption("Place your mouse cursor over a card below to uncover important PCOS insights.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown(
            fact_card(
                "fact", "Uncommon Fact", "Not Just About Ovaries",
                "Despite the name, you do not need to have ovarian cysts to be diagnosed with PCOS. "
                "It is fundamentally a metabolic and endocrine condition rather than an anatomical disease.",
            ),
            unsafe_allow_html=True,
        )
    with col_f2:
        st.markdown(
            fact_card(
                "myth", "Busting Myths", "&ldquo;Weight Loss Cures It&rdquo;",
                "PCOS is a chronic, lifelong condition with no magic cure. Managing weight can reduce "
                "insulin-resistance symptoms, but lean individuals develop PCOS too.",
            ),
            unsafe_allow_html=True,
        )
    with col_f3:
        st.markdown(
            fact_card(
                "myth", "Breaking Taboos", "The Silence on Hair Growth",
                "Hirsutism (excess facial/body hair due to androgens) causes heavy social anxiety and "
                "stigma, yet it is reported in a large majority of people with PCOS worldwide.",
            ),
            unsafe_allow_html=True,
        )

    gap(6)   # big space between the hover cards and "Modules Overview"
    st.markdown('### <span style="color:#C62828;">&#10148;</span>&nbsp;&nbsp;Modules Overview', unsafe_allow_html=True)
    gap(1.5)

    STEP = 1.4          # how far each card shifts right (diagonal)
    CARD_W, BTN_W = 7, 2.4
    for i, (num, label, tagline, body, btn, target, key) in enumerate(MODULES):
        left_space = i * STEP
        right_space = (len(MODULES) - 1 - i) * STEP
        cols = st.columns(
            [max(left_space, 0.01), CARD_W, BTN_W, max(right_space, 0.01)],
            vertical_alignment="center",
            gap="medium",
        )
        with cols[1]:
            st.markdown(module_card(num, label, tagline, body), unsafe_allow_html=True)
        with cols[2]:
            if st.button(btn, key=key, type="primary", use_container_width=True):
                st.session_state["next_page"] = target
                st.rerun()
        gap(1.8)        # space between one module box and the next


elif page == "My Health Profile":
    # Layout containers are created in display order, then filled in the order the
    # data flows (workspace first), so every counter on the page is always up to date.
    hero_box = st.container()
    gap(1.5)
    overview_box = st.container()
    gap(3.5)
    workspace_box = st.container()
    gap(4)
    review_box = st.container()

    active = st.session_state.active_section
    sec = PROFILE_SECTIONS[active]

    # ---------------- Workspace: inputs for the open section ----------------
    with workspace_box:
        head_box = st.container()
        st.caption(f"{len(sec['features'])} parameters · Leave anything unknown blank. Entries save automatically.")
        new_vals = {}
        feats = sec["features"]
        for start in range(0, len(feats), 3):
            row_cols = st.columns(3, gap="large")
            for c, feature in zip(row_cols, feats[start:start + 3]):
                new_vals[feature] = render_field(
                    feature, c, st.session_state.profile_values.get(feature)
                )
            gap(0.8)

        current = st.session_state.profile_values
        if any(current.get(f) != v for f, v in new_vals.items()):
            st.session_state.profile_values = {**current, **new_vals}
            st.session_state.prediction_result = None  # profile changed; old result is stale

        vals = st.session_state.profile_values
        hints = []
        for feature in sec["features"]:
            if vals.get(feature) is None:
                h = derived_hint(feature, vals)
                if h is not None:
                    hints.append(f"{FIELDS[feature][0]} &asymp; <b>{h:.2f}</b>")
        if hints:
            st.markdown(
                '<div class="note"><b>Calculated from your entries:</b> ' + " &nbsp;·&nbsp; ".join(hints)
                + ". If you leave the field blank, the model calculates this value for you automatically "
                "(it is exact in the dataset audit); you can still type your own value to override it.</div>",
                unsafe_allow_html=True,
            )

        gap(1.2)
        n1, n2, _ = st.columns([1, 1, 3])
        n1.button("← Previous section", key="prev_sec", on_click=go_section, args=(active - 1,),
                  disabled=active == 0, use_container_width=True)
        n2.button("Next section →", key="next_sec", on_click=go_section, args=(active + 1,),
                  disabled=active == len(PROFILE_SECTIONS) - 1, type="primary", use_container_width=True)

        done = section_count(sec)
        with head_box:
            st.markdown(
                f'<div class="ws-head"><div class="ws-num">{sec["num"]}</div>'
                f'<div class="ws-text"><h3>{sec["title"]}</h3><p>{sec["sub"]}</p></div>'
                f'<div class="ws-count">{done}/{len(sec["features"])}<small>provided</small></div></div>',
                unsafe_allow_html=True,
            )

    # ---------------- Hero (live progress ring) ----------------
    total = len(ALL_FEATURES)
    provided = total_provided()
    pct = int(round(100 * provided / total))
    status = "Not started" if provided == 0 else ("Complete" if provided == total else "In progress")
    with hero_box:
        st.markdown(
            '<div class="ph-hero"><div class="ph-left"><span class="pill">Your clinical data</span>'
            "<h2>BUILD your HEALTH PROFILE.</h2>"
            "<p>Enter the information you have available. Your profile is organised into clinical "
            "feature groups so you can review each category separately. Nothing here is a prediction "
            "yet &mdash; this page only collects and checks your data.</p></div>"
            f'<div class="ph-right"><div class="ph-ring" style="--p:{pct}"><span>{pct}%</span></div>'
            '<div class="ph-stats">'
            f'<div class="ph-stat"><b>{len(PROFILE_SECTIONS)}</b><small>Feature groups</small></div>'
            f'<div class="ph-stat"><b>{provided} / {total}</b><small>Inputs provided</small></div>'
            f'<div class="ph-stat"><b>{status}</b><small>Input status</small></div>'
            "</div></div></div>",
            unsafe_allow_html=True,
        )

    # ---------------- Overview: pick a section ----------------
    with overview_box:
        st.markdown('### <span style="color:#C62828;">&#10148;</span>&nbsp;&nbsp;Your Information', unsafe_allow_html=True)
        st.caption("Select a group to enter its parameters. Leave unavailable information blank.")
        gap(0.8)
        for start in range(0, len(PROFILE_SECTIONS), 3):
            row_cols = st.columns(3, gap="large")
            for c, i in zip(row_cols, range(start, min(start + 3, len(PROFILE_SECTIONS)))):
                s = PROFILE_SECTIONS[i]
                is_active = i == active
                with c:
                    st.markdown(
                        section_card(s, section_count(s), len(s["features"]), is_active),
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "● Currently open" if is_active else "Open section",
                        key=f"sec_btn_{i}",
                        type="primary" if is_active else "secondary",
                        on_click=go_section, args=(i,),
                        use_container_width=True,
                    )
            gap(1.6)

    # ---------------- Review ----------------
    with review_box:
        st.markdown('### <span style="color:#C62828;">&#10148;</span>&nbsp;&nbsp;Profile Review', unsafe_allow_html=True)
        st.caption(
            "Review what you have entered, spot missing parameters, and continue when ready. "
            "Missing information is handled by the model pipeline's training-time imputers."
        )
        gap(1)
        left, right = st.columns([1, 1.15], gap="large")

        with left:
            acc = MISSING_GROUP_RESULTS.set_index("Scenario")["Accuracy"]
            ovarian_sec, symptom_sec = PROFILE_SECTIONS[5], PROFILE_SECTIONS[3]
            if provided == 0:
                st.markdown(
                    '<div class="tip-box"><b>Getting started.</b> Open a section above and enter whatever '
                    "values you have. Your progress will appear here.</div>",
                    unsafe_allow_html=True,
                )
            elif (
                section_count(ovarian_sec) < len(ovarian_sec["features"])
                or section_count(symptom_sec) < len(symptom_sec["features"])
            ):
                st.markdown(
                    '<div class="tip-box"><b>Where to focus first.</b> In your Data Lab experiments, '
                    "removing ovarian measurements lowered accuracy the most "
                    f"(to {acc['Without ovarian measurements']:.1%}), followed by symptoms and lifestyle "
                    f"(to {acc['Without symptoms and lifestyle']:.1%}). Completing those groups gives "
                    "the model the most to work with.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="tip-box"><b>Key groups covered.</b> Your ovarian measurements and symptoms '
                    "are complete &mdash; the two groups that mattered most in your Data Lab experiments. "
                    "You can continue to your results.</div>",
                    unsafe_allow_html=True,
                )

        with right:
            rows = ""
            for s in PROFILE_SECTIONS:
                d, t = section_count(s), len(s["features"])
                rows += (
                    f'<div class="rev-row"><span class="rev-name">{s["num"]} &middot; {s["title"]}</span>'
                    f'<div class="rev-bar"><i style="width:{int(round(100 * d / t))}%"></i></div>'
                    f'<span class="rev-count">{d}/{t}</span></div>'
                )
            st.markdown(rows, unsafe_allow_html=True)

        gap(1.4)
        with st.expander("Show missing parameters"):
            any_missing = False
            for s in PROFILE_SECTIONS:
                missing = [f for f in s["features"] if st.session_state.profile_values.get(f) is None]
                if missing:
                    any_missing = True
                    chips = "".join(f'<span class="chip">{FIELDS[f][0]}</span>' for f in missing)
                    st.markdown(f"**{s['num']} · {s['title']}**<br>{chips}", unsafe_allow_html=True)
            if not any_missing:
                st.success("Every parameter has a value.")

        gap(1.6)
        b1, b2, _ = st.columns([1, 1.3, 2])
        b1.button("Reset profile", key="reset_profile", on_click=reset_profile, use_container_width=True)
        b2.button(
            "Review & Continue →", key="go_results", type="primary", on_click=go_results,
            disabled=provided == 0, use_container_width=True,
        )
        if provided == 0:
            st.caption("Enter at least one value to continue to your results.")


elif page == "Understand My Results":
    hero(
        "Step 2 · Model output",
        "See what the model returns.",
        "This page runs the saved Random Forest pipeline using your profile. The result is a "
        "research estimate, not a calibrated personal probability or a medical diagnosis.",
    )

    with st.expander("Model diagnostics (open this if results do not appear)"):
        st.write("This app file lives in:", str(APP_DIR))
        st.write("Project folder (one level up):", str(BASE_DIR))
        st.write("Files in project folder:", sorted(x.name for x in BASE_DIR.iterdir())[:40] if BASE_DIR.exists() else [])
        _models_dir = BASE_DIR / "models"
        st.write("Files in models folder:", sorted(x.name for x in _models_dir.iterdir()) if _models_dir.exists() else "models folder not found here")
        st.write("Model path checked:", str(MODEL_PATH))
        st.write("Model file found:", MODEL_PATH.exists())
        st.write("Metadata file found:", META_PATH.exists())
        st.write("Model loaded:", model is not None)
        st.write("Features the app will send:", len(expected_features))
        if model_error:
            st.code(model_error)

    if model_error:
        st.error(model_error)
    elif model is None:
        st.warning("The saved model file is not available in the models folder.")
    elif not any(v is not None for v in st.session_state.profile_values.values()):
        st.info("Start by entering any available information in My Health Profile.")
    else:
        values = st.session_state.profile_values
        provided = sum(values.get(f) is not None for f in expected_features)
        st.metric("Inputs provided", f"{provided} / {len(expected_features)}")

        if st.button("Run research model", type="primary"):
            try:
                with st.spinner("Scoring your profile and computing explanations..."):
                    X = build_X(values, expected_features)
                    score = score_only(model, X)
                    votes, n_trees = tree_votes(model, X)
                    label, css, reasons = make_verdict(score, values)
                    explanation = explain_row(model, values, expected_features)
                st.session_state.prediction_result = {
                    "score": score,
                    "votes": votes,
                    "n_trees": n_trees,
                    "label": label,
                    "css": css,
                    "reasons": reasons,
                    "provided": provided,
                    "total": len(expected_features),
                    "explain": explanation,
                }
            except Exception as exc:
                st.error(f"The model could not process this profile: {exc}")
                with st.expander("Technical details (share this if you need help)"):
                    st.code(traceback.format_exc())

        r = st.session_state.prediction_result
        if r:
            st.divider()

            reasons_html = ""
            if r["reasons"]:
                reasons_html = "<ul>" + "".join(f"<li>{x}</li>" for x in r["reasons"]) + "</ul>"
            st.markdown(
                f'<div class="verdict {r["css"]}"><small>Research estimate</small>'
                f'<h2>{r["label"]}</h2>{reasons_html}</div>',
                unsafe_allow_html=True,
            )
            if r["label"] == "Inconclusive":
                st.caption(
                    "Inconclusive means the app does not have enough reliable information to lean either "
                    "way. Add the missing groups in My Health Profile and run the model again."
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Model score (class 1)", f"{r['score']:.1%}")
            c2.metric("Trees voting PCOS", f"{r['votes']} / {r['n_trees']}")
            c3.metric("Inputs provided", f"{r['provided']} / {r['total']}")
            st.progress(min(max(r["score"], 0.0), 1.0))
            auto = derived_used(values, expected_features)
            if auto:
                st.caption(
                    "Calculated automatically from your entries: "
                    + ", ".join(f"{lbl} = {val:.2f}" for lbl, val in auto) + "."
                )
            st.caption(
                "BMI, waist:hip ratio and FSH:LH are calculated when their ingredients are entered. Any other "
                "blank input is filled by the pipeline's imputers (median for numbers, most frequent for "
                "categories), so the model sees an 'average patient' plus your entries. "
                "The score is the share of tree probability for class 1, not a calibrated probability."
            )

            gap(1)
            st.markdown("### What drove *this* result")
            ex = r["explain"]
            if ex is None or ex.empty:
                st.info("No entered values to explain.")
            else:
                top = ex.head(10)
                st.bar_chart(top.set_index("Feature")["Effect on score"], color="#C62828")
                st.caption(
                    "Positive = pushed the score towards PCOS; negative = pushed it away. Computed by blanking "
                    "each entered value in turn and re-scoring, so it is specific to this profile."
                )
                with st.expander("Show all effects"):
                    st.dataframe(ex, hide_index=True, use_container_width=True)

            gap(1)
            st.markdown("### Data coverage vs. reliability")
            st.dataframe(group_coverage(values), hide_index=True, use_container_width=True)
            st.caption(
                "Held-out accuracy of the deployed model when that whole group is unavailable "
                "(from the Data Lab validation run)."
                if DEPLOYED_RESULTS is not None else
                "Accuracy figures come from Data Lab experiments in which the model was retrained without "
                "that group (all five groups: 90.8%). Run 02_results_validation.ipynb to replace them with "
                "figures for this exact deployed model."
            )

            with st.expander("Global feature importance (same for every person)"):
                imp_df = pd.DataFrame(TOP_IMPORTANCE, columns=["Feature", "Importance", "Feature group"])
                st.bar_chart(imp_df.set_index("Feature")["Importance"], color="#C62828")

            st.markdown(
                '<div class="note"><b>Limitations.</b> The model was trained on 541 records. The verdict '
                "thresholds (35% to 65% boundary band, minimum ovarian and symptom inputs) are app design "
                "choices, not notebook results. This is a research prototype, not a diagnostic tool.</div>",
                unsafe_allow_html=True,
            )

elif page == "Data Lab":
    hero(
        "Step 3 · Research experiments",
        "What changes when data goes missing?",
        "Explore previously calculated experiments from your notebook dataset.",
    )

    view = st.selectbox(
        "Choose an experiment scenario",
        ["Random missing values", "Missing a feature group (model retrained without it)",
         "One group at a time", "Deployed model: group missing at prediction time",
         "Missing-data strategies compared", "Is 'Inconclusive' justified?"],
    )

    if view == "Random missing values":
        chart_df = MISSING_RATE_RESULTS.set_index("Missing values simulated")[
            ["Accuracy", "Sensitivity", "F1-score", "ROC-AUC"]
        ]
        st.line_chart(chart_df, color=["#8E0000", "#C62828", "#E57373", "#F4A3A3"])
        st.dataframe(MISSING_RATE_RESULTS, use_container_width=True, hide_index=True)

    elif view == "Missing a feature group (model retrained without it)":
        chosen = st.selectbox("Select a group scenario", MISSING_GROUP_RESULTS["Scenario"].tolist())
        row = MISSING_GROUP_RESULTS[MISSING_GROUP_RESULTS["Scenario"] == chosen].iloc[0]
        st.metric("Accuracy achieved", f"{100 * row['Accuracy']:.1f}%")
        st.dataframe(MISSING_GROUP_RESULTS, use_container_width=True, hide_index=True)

    elif view == "One group at a time":
        st.bar_chart(INDIVIDUAL_GROUP_RESULTS.set_index("Feature group")[["Accuracy", "ROC-AUC"]], color=["#8E0000", "#E57373"])
        st.dataframe(INDIVIDUAL_GROUP_RESULTS, use_container_width=True, hide_index=True)

    elif view == "Deployed model: group missing at prediction time":
        st.markdown(
            '<div class="note"><b>This is what the app actually does.</b> The deployed model is trained on '
            "complete records, then a record with a whole group missing is passed to it and the gaps are "
            "imputed. The earlier 'retrained without it' tables describe a different, smaller model.</div>",
            unsafe_allow_html=True,
        )
        if DEPLOYED_RESULTS is None:
            st.info("Run 02_results_validation.ipynb to create reports/deployed_model_missing_results.csv.")
        else:
            st.bar_chart(DEPLOYED_RESULTS.set_index("Scenario")[["Accuracy", "Sensitivity", "ROC_AUC"]],
                         color=["#8E0000", "#C62828", "#E57373"])
            st.dataframe(DEPLOYED_RESULTS, use_container_width=True, hide_index=True)

    elif view == "Missing-data strategies compared":
        if STRATEGY_RESULTS is None:
            st.info("Run 02_results_validation.ipynb to create reports/B_strategy_summary.csv.")
        else:
            st.caption("Mean ROC-AUC from repeated 5-fold cross-validation on all 541 records. Every strategy "
                       "was tested on identical masked data.")
            auc = STRATEGY_RESULTS.pivot(index="Scenario", columns="Strategy", values="ROC_AUC")
            auc = auc.loc[[s for s in STRATEGY_RESULTS["Scenario"].drop_duplicates() if s in auc.index]]
            st.dataframe(auc.round(3), use_container_width=True)
            names = STRATEGY_RESULTS[["Strategy", "Strategy_name"]].drop_duplicates()
            for _, r_ in names.iterrows():
                st.write(f"**{r_['Strategy']}** = {r_['Strategy_name']}")
            cells = [s for s in auc.index if s == "Complete" or s.startswith("Cells")]
            st.line_chart(auc.loc[cells])

    elif view == "Is 'Inconclusive' justified?":
        if ABSTAIN_RESULTS is None:
            st.info("Run 02_results_validation.ipynb to create reports/D_abstention_policy_evaluation.csv.")
        else:
            st.caption("Simulated partial profiles: each clinical group is absent with the chosen probability. "
                       "A good rule gives higher accuracy on the cases it decides than on the ones it refuses.")
            q_ = st.selectbox("Chance that each group is missing", sorted(ABSTAIN_RESULTS["Group_missing_prob"].unique()))
            m_ = st.selectbox("Model", sorted(ABSTAIN_RESULTS["Model"].unique()))
            sub = ABSTAIN_RESULTS[(ABSTAIN_RESULTS["Group_missing_prob"] == q_) & (ABSTAIN_RESULTS["Model"] == m_)]
            st.dataframe(sub.drop(columns=["Model", "Group_missing_prob"]), use_container_width=True, hide_index=True)

elif page == "Learn & Explore":
    hero(
        "Step 4 · Educational Corner",
        "Understand Clinical Features",
        "Explore details about variables monitored within research ecosystems. "
        "Open a group, then click any feature to see what it is and what it does.",
    )
    for group, features in GROUPS.items():
        with st.expander(f"{group} Group"):
            st.caption("Click a feature to read a one-line explanation.")
            for start in range(0, len(features), 3):
                row_cols = st.columns(3, gap="medium")
                for c, feature in zip(row_cols, features[start:start + 3]):
                    label, unit, _, _ = FIELDS.get(feature, (feature, "", None, None))
                    with c:
                        with st.popover(label, use_container_width=True):
                            unit_txt = f"<br><small>Unit: {unit}</small>" if unit else ""
                            st.markdown(
                                f'<div class="feat-info"><b>{label}</b><br>'
                                f'{FEATURE_INFO.get(feature, "No description available.")}{unit_txt}</div>',
                                unsafe_allow_html=True,
                            )

st.divider()
st.markdown(
    '<div class="footer">PCOS SCOPE  🎗️ Made By - Kuhu Chandna  </div>',
    unsafe_allow_html=True,
)