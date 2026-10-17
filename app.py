import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"
FEATURE_COLUMNS = ["studytime", "absences", "G1"]  # must match training order
PASS_MARK = 10  # G3 >= 10 counts as passing

PASS_COLOR = "#2f8f5b"  # circle colour when the student is on track
FAIL_COLOR = "#d64545"  # circle colour when the student is at risk

# Where the title + subtitle go:
#   True  -> inside the left column, above the input card. This matches the
#            mockup: the output card starts at the top of the page next to the
#            title, and both cards end on the same bottom line.
#   False -> across the whole top of the page, so both cards start on the same
#            line AND end on the same line (two equal boxes side by side).
HEADER_IN_LEFT_COLUMN = True

st.set_page_config(
    page_title="Grade Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# HELPER: render raw HTML without it turning into a code block
#
# Streamlit runs st.markdown() through a Markdown parser. Any line indented by
# 4+ spaces (or any blank line inside a <div>) makes it show your HTML as plain
# text. Stripping each line's indentation and joining into one line avoids it.
# ---------------------------------------------------------------------------
def html(block: str) -> None:
    flat = " ".join(line.strip() for line in block.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CSS  (colours sampled from the mockup)
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Finger+Paint&family=Press+Start+2P&display=swap');

/* ===== GLOBAL PAGE ===== */
.stApp {
    background: linear-gradient(180deg,
        #cbfbb9 0%, #91cca0 21%, #539682 45%, #366a5b 68%, #214130 91%, #1b3424 100%);
    color: #000;
    font-family: "Comic Sans MS", "Chalkboard SE", sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 100%;
    padding: 1.25rem 2.5rem 2rem 2.5rem;
}

/* ===== TITLE + SUBTITLE ===== */
.header-title {
    font-family: "Press Start 2P", "Brick Sans", Impact, sans-serif;
    font-size: clamp(16px, 2.2vw, 48px);
    line-height: 1.3;
    letter-spacing: 0;
    color: #000;
    margin: 0;
}
.header-subtitle {
    background: #326f51;
    color: #d9d9d9;
    border-radius: 22px;
    padding: 14px 20px;
    max-width: 760px;
    text-align: center;
    font-family: "Comic Sans MS", "Chalkboard SE", sans-serif;
    font-size: clamp(14px, 1.5vw, 24px);
    line-height: 1.35;
    font-weight: 600;
    margin: 0;
}

/* ===== LAYOUT: equal-height columns, each card fills its column =====
   The two cards end on the same line because each one stretches to the
   bottom of its column, and both columns are the same height. */
div[data-testid="stHorizontalBlock"] { align-items: stretch !important; }

div[data-testid="stColumn"]:has(.st-key-input-card),
div[data-testid="stColumn"]:has(.st-key-output-card),
div[data-testid="column"]:has(.st-key-input-card),
div[data-testid="column"]:has(.st-key-output-card) {
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stColumn"]:has(.st-key-input-card) > div,
div[data-testid="stColumn"]:has(.st-key-output-card) > div,
div[data-testid="column"]:has(.st-key-input-card) > div,
div[data-testid="column"]:has(.st-key-output-card) > div {
    flex: 1 1 auto !important;
}
/* Newer Streamlit wraps containers in an extra div; take it out of the layout */
div[data-testid="stColumn"]:has(.st-key-input-card) [data-testid="stLayoutWrapper"],
div[data-testid="stColumn"]:has(.st-key-output-card) [data-testid="stLayoutWrapper"] {
    display: contents !important;
}

/* ===== INPUT CARD ===== */
.st-key-input-card {
    flex: 1 1 auto !important;
    background: linear-gradient(180deg,
        #b8e4f9 0%, #a4c8e9 13%, #859cd3 27%, #5f64b8 45%,
        #40438e 63%, #232a5f 81%, #071130 100%);
    border: 4px solid #fff;
    border-radius: 40px;
    padding: 0 26px 24px 26px;
    box-sizing: border-box;
}
.section-heading {
    width: 84%;
    margin: 0 auto 8px auto;
    box-sizing: border-box;
    background: #fff;
    color: #545454;
    border-radius: 0 0 36px 36px;
    padding: 14px 18px 16px 18px;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-family: "Finger Paint", cursive;
    font-size: clamp(18px, 1.9vw, 32px);
}

/* Labels */
.st-key-input-card [data-testid="stWidgetLabel"],
.st-key-input-card [data-testid="stWidgetLabel"] p,
.st-key-input-card [data-testid="stWidgetLabel"] label {
    color: #fff !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}

/* Number inputs: white pill, dark text and +/- */
.st-key-input-card [data-testid="stNumberInput"] { background: transparent !important; }
.st-key-input-card [data-testid="stNumberInput"] [data-baseweb="input"] {
    background: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    overflow: hidden !important;
    min-height: 46px;
}
.st-key-input-card [data-testid="stNumberInput"] [data-baseweb="input"] > div,
.st-key-input-card [data-testid="stNumberInput"] [data-baseweb="base-input"] {
    background: #fff !important;
    border: none !important;
    box-shadow: none !important;
}
.st-key-input-card [data-testid="stNumberInput"] input {
    background: #fff !important;
    color: #111 !important;
    -webkit-text-fill-color: #111 !important;
    caret-color: #111 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding-left: 20px !important;
}
.st-key-input-card [data-testid="stNumberInput"] button {
    background: #fff !important;
    color: #111 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
.st-key-input-card [data-testid="stNumberInput"] button svg {
    color: #111 !important; fill: #111 !important; stroke: #111 !important;
}
.st-key-input-card [data-testid="stNumberInput"] button:hover { background: #eee !important; }

/* Dropdown: white pill, dark text and arrow */
.st-key-input-card [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    min-height: 46px;
}
.st-key-input-card [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
    background: transparent !important;
}
.st-key-input-card [data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #111 !important; -webkit-text-fill-color: #111 !important;
}
.st-key-input-card [data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #111 !important; fill: #111 !important; stroke: #111 !important;
}
div[data-baseweb="popover"] ul[role="listbox"],
div[data-baseweb="popover"] li[role="option"] {
    background: #fff !important; color: #111 !important;
}
div[data-baseweb="popover"] li[role="option"] * {
    color: #111 !important; -webkit-text-fill-color: #111 !important;
}
div[data-baseweb="popover"] li[role="option"]:hover { background: #eee !important; }

/* ===== OUTPUT CARD ===== */
.st-key-output-card {
    flex: 1 1 auto !important;
    background: #2f6569;
    border: 4px solid #fff;
    border-radius: 40px;
    padding: 22px;
    box-sizing: border-box;
}
/* Inside the output card, drop Streamlit's wrapper divs so the pill, circle and
    outline box become direct flex children and can stretch to fill the card */
.st-key-output-card [data-testid="stElementContainer"],
.st-key-output-card [data-testid="element-container"],
.st-key-output-card [data-testid="stMarkdown"],
.st-key-output-card [data-testid="stMarkdownContainer"] {
    display: contents !important;
}
.st-key-output-card [data-testid="stHorizontalBlock"] { flex: 1 1 auto !important; }
.st-key-output-card [data-testid="stColumn"],
.st-key-output-card [data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
}
.st-key-output-card [data-testid="stColumn"] > div,
.st-key-output-card [data-testid="column"] > div { flex: 1 1 auto !important; }

.predicted-
