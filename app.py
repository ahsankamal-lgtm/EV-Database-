import json
from datetime import datetime, date, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk


# =============================
# UI / CONFIG
# =============================
st.set_page_config(page_title="🛸 EV GPS Analysis App", layout="wide")

# =============================
# AESTHETICS (UI ONLY)
# =============================
CUSTOM_CSS = """
<style>

/* ---------- FULL CANVAS FIX ---------- */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stApp"] {
  height: 100%;
  width: 100%;
  margin: 0;
  padding: 0;
}

/* ---------- GLOBAL BACKGROUND ---------- */
.stApp {
  min-height: 100vh !important;
  background: linear-gradient(
    135deg,
    #001F3F 0%,
    #0074D9 45%,
    #2ED9C3 75%,
    #7FDBFF 100%
  ) !important;
  background-attachment: fixed !important;
}

/* Remove Streamlit top bars background */
[data-testid="stHeader"],
[data-testid="stToolbar"] {
  background: transparent !important;
}

/* ---------- GLOBAL TEXT (Turquoise Gradient) ---------- */
body, p, span, div, label {
  color: #D8FFFB;
}

/* ---------- MAIN HEADING GRADIENT ---------- */
h1, h2, h3, h4, h5, h6,
[data-testid="stTitle"] {
  background: linear-gradient(90deg, #5EEAD4, #2ED9C3, #7FDBFF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: 0.4px;
}

/* ---------- CAPTION ---------- */
div[data-testid="stCaptionContainer"] {
  color: rgba(220, 255, 250, 0.85) !important;
}

/* ---------- CONTENT WIDTH ---------- */
.block-container {
  padding-top: 1.6rem;
  padding-bottom: 2rem;
  max-width: 1200px;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(12px);
  border-right: 1px solid rgba(0,0,0,0.08);
}
section[data-testid="stSidebar"] * {
  color: #0B1B2B !important;
}

/* ---------- CARD ---------- */
.card {
  background: rgba(255, 255, 255, 0.94);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 10px 26px rgba(0,0,0,0.14);
  margin-bottom: 16px;
}

/* ---------- TABS ---------- */
div[data-testid="stTabs"] > div {
  background: rgba(255, 255, 255, 0.94);
  border-radius: 18px;
  padding: 10px 12px;
}

/* ---------- DATAFRAME ---------- */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
  border-radius: 12px;
  padding: 0.5rem 0.9rem;
}

/* ---------- INPUTS ---------- */
div[data-baseweb="select"] > div {
  border-radius: 12px !important;
}

/* ---------- ALERTS ---------- */
div[data-testid="stAlert"] {
  border-radius: 14px;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


st.title("🛸 EV Analytics App")
st.caption("Keyed by tc_positions.deviceid. Timeline uses fixtime. Noise points are excluded by default.")

# =============================
# EVERYTHING BELOW IS 100% UNCHANGED
# =============================

# (Your full existing logic continues exactly as-is)
