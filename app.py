import json
from datetime import datetime, date, time, timedelta
import hmac

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk


# =============================
# UI / CONFIG
# =============================
st.set_page_config(page_title="🛸 EV GPS Analysis App", layout="wide")


# -----------------------------
# LOGIN HELPERS
# -----------------------------
def _check_password(username: str, password: str) -> bool:
   
    try:
        cfg = st.secrets["auth"]
        valid_user = str(cfg.get("username", ""))
        valid_pass = str(cfg.get("password", ""))
    except Exception:
        return False

    return hmac.compare_digest(username, valid_user) and hmac.compare_digest(password, valid_pass)


def login_gate() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return

    # ✅ ONLY CHANGE (Landing page): bigger, more beautiful, single enlarged card, inputs inside,
    # white/bigger labels, and borderless inputs (no boxes).
    st.markdown(
        """
        <style>
          /* Scope login-only styling (there are no other forms in the app) */
          div[data-testid="stForm"] label {
            color: #FFFFFF !important;
            font-size: 18px !important;
            font-weight: 800 !important;
          }
          /* Borderless "underline" inputs for login */
          div[data-testid="stForm"] input {
            background: rgba(255,255,255,0.10) !important;
            color: #FFFFFF !important;
            border: 0 !important;
            outline: none !important;
            box-shadow: none !important;
          }
          /* Remove the default input box chrome */
          div[data-testid="stForm"] div[data-baseweb="input"] > div {
            background: rgba(255,255,255,0.10) !important;
            border: 0 !important;
            box-shadow: none !important;
          }
          /* Add a subtle underline */
          div[data-testid="stForm"] div[data-baseweb="input"] {
            border-bottom: 2px solid rgba(255,255,255,0.35) !important;
            border-radius: 14px !important;
            padding: 6px 6px !important;
          }

          /* Make the sign-in button look nicer in the login card */
          div[data-testid="stForm"] button {
            border-radius: 14px !important;
            padding: 0.6rem 1.1rem !important;
            font-weight: 800 !important;
          }

          /* Reduce extra blank space under widgets inside the login form */
          div[data-testid="stForm"] .stTextInput,
          div[data-testid="stForm"] .stButton {
            margin-bottom: 10px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            max-width: 740px;
            margin: 10vh auto 0 auto;
            padding: 0;
            border-radius: 30px;
            overflow: hidden;
            box-shadow: 0 26px 70px rgba(0,0,0,0.24);
            border: 0;
        ">
          <div style="
              padding: 26px 28px;
              background: linear-gradient(135deg, rgba(0,116,217,0.92), rgba(46,217,195,0.88));
          ">
            <div style="display:flex; align-items:center; gap:16px;">
              <div style="
                width:64px; height:64px; border-radius:22px;
                display:flex; align-items:center; justify-content:center;
                background: rgba(255,255,255,0.18);
                color:white; font-size:28px; font-weight:900;
                box-shadow: 0 14px 34px rgba(0,0,0,0.18);
              ">EV</div>
              <div>
                <div style="font-size:34px; font-weight:950; color:#FFFFFF; line-height:1.05;">
                  EV Analytics App
                </div>
                <div style="font-size:18px; color:rgba(255,255,255,0.92); margin-top:8px;">
                  Please sign in to continue
                </div>
              </div>
            </div>
          </div>

          <div style="
              padding: 24px 28px 26px 28px;
              background: rgba(255,255,255,0.08);
              backdrop-filter: blur(10px);
          ">
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        # White/bigger labels are provided via custom HTML; we collapse Streamlit labels.
        st.markdown('<div style="font-size:18px; font-weight:900; color:#FFFFFF; margin:6px 0 6px 2px;">Username</div>', unsafe_allow_html=True)
        username = st.text_input("", label_visibility="collapsed", placeholder="Enter username")

        st.markdown('<div style="font-size:18px; font-weight:900; color:#FFFFFF; margin:14px 0 6px 2px;">Password</div>', unsafe_allow_html=True)
        password = st.text_input("", type="password", label_visibility="collapsed", placeholder="Enter password")

        submitted = st.form_submit_button("Sign in")

    if submitted:
        if _check_password(username.strip(), password):
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid User ID or Password.")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


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

/* ---------- GLOBAL TEXT (DARK GREY FOR CLARITY) ---------- */
/* ✅ ONLY CHANGE (3): any text on white should be dark grey */
body, p, span, div, label, li {
  color: #374151 !important;
}

/* ---------- HEADINGS (DARK GREY, NOT GRADIENT) ---------- */
/* IMPORTANT: exclude the custom title so it stays gradient */
/* ✅ ONLY CHANGE (3): white headings -> dark grey for clarity */
h1:not(.gradient-title), h2, h3, h4, h5, h6 {
  color: #374151 !important;
  background: none !important;
  -webkit-text-fill-color: #374151 !important;
}

/* ---------- TITLE GRADIENT ONLY (Turquoise/Blue -> White) ---------- */
h1.gradient-title {
  display: inline-block;
  margin: 0;
  font-weight: 800;
  letter-spacing: 0.5px;

  background: linear-gradient(90deg, #FFFFFF 100%) !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;

  -webkit-text-stroke: 0 !important;
  text-stroke: 0 !important;
}

/* ---------- CAPTION (DARK GREY) ---------- */
/* ✅ ONLY CHANGE (3) */
div[data-testid="stCaptionContainer"] {
  color: #374151 !important;
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
  color: #374151 !important; /* ✅ ONLY CHANGE (3) */
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


# ✅ CALL LOGIN GATE HERE (so nothing loads unless logged in)
login_gate()


def card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# Title: gradient fill (turquoise/blue/white) with NO border
st.markdown('<h1 class="gradient-title">🚴‍♀️ EV Analytics App</h1>', unsafe_allow_html=True)
st.caption("Keyed by tc_positions.deviceid. Timeline uses fixtime. Noise points are excluded by default.")


# =============================
# CONSTANTS
# =============================
DB_SCHEMA = "traccar_new"
POSITIONS_TABLE = f"{DB_SCHEMA}.tc_positions"
DEVICES_TABLE = f"{DB_SCHEMA}.tc_devices"  # <-- NEW (for name mapping)
MAX_BIKES = 5
NOISE_CUTOFF = datetime(2000, 1, 1)

# sanity clamp for per-reading increment derived from distance (meters)
MAX_PLAUSIBLE_SPEED_KMH = 120.0
SANITY_BUFFER = 1.5

# ✅ ONLY CHANGE: speed display cutoff
SPEED_CUTOFF_KMH = 85.0


# =============================
# HELPERS
# =============================
def dt_range_inclusive(start_d: date, end_d: date):
    start_dt = datetime.combine(start_d, time.min)
    end_dt = datetime.combine(end_d, time.max)
    return start_dt, end_dt


def knots_to_kmh(knots):
    try:
        if knots is None:
            return np.nan
        return float(knots) * 1.852
    except Exception:
        return np.nan


def safe_json_load(x):
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            return {}
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return {}
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def to_bool(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def format_hms(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)) or seconds <= 0:
        return "0:00:00"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def split_session_by_day(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
        return []

    out = []
    cur = start_ts

    while cur.date() < end_ts.date():
        day_end = pd.Timestamp(datetime.combine(cur.date(), time.max))
        seg_end = min(day_end, end_ts)
        out.append((cur.date(), (seg_end - cur).total_seconds()))
        cur = pd.Timestamp(datetime.combine(cur.date() + timedelta(days=1), time.min))

    out.append((cur.date(), (end_ts - cur).total_seconds()))
    return out


def alt_line(df, x, y, color, title, y_title=None):
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(x, title="Time"),
            y=alt.Y(y, title=y_title or y),
            color=alt.Color(color, legend=alt.Legend(title="Device ID")),
            tooltip=[color, x, y],
        )
        .properties(height=360, title=title)
        .interactive()
    )


# =============================
# DB DRIVER AUTO-DETECT
# =============================
DB_DRIVER = None
try:
    import pymysql  # type: ignore
    DB_DRIVER = "pymysql"
except Exception:
    try:
        import mysql.connector  # type: ignore
        DB_DRIVER = "mysql-connector"
    except Exception:
        DB_DRIVER = None

if DB_DRIVER is None:
    st.error(
        "Missing MySQL driver in the Streamlit environment.\n\n"
        "Fix:\n"
        "1) Ensure requirements.txt is in the repo ROOT (same folder as app.py)\n"
        "2) Put this inside requirements.txt:\n\n"
        "streamlit\npandas\nnumpy\naltair\npydeck\npymysql\nmysql-connector-python\n\n"
        "Then redeploy / reboot the app from Streamlit Cloud."
    )
    st.stop()


@st.cache_resource
def get_conn_params():
    cfg = st.secrets["mysql"]
    return {
        "host": cfg["host"],
        "port": int(cfg.get("port", 3306)),
        "user": cfg["username"],
        "password": cfg["password"],
        "database": cfg["database"],
    }


def fetch_df(query: str, params: tuple):
    cfg = get_conn_params()

    if DB_DRIVER == "pymysql":
        import pymysql  # type: ignore
        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            # ✅ ONLY CHANGE: remove default database selection at connect-time
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            return pd.DataFrame(rows)
        finally:
            conn.close()

    # mysql-connector fallback
    import mysql.connector  # type: ignore
    conn = mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        # ✅ ONLY CHANGE: remove default database selection at connect-time
    )
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        return pd.DataFrame(rows)
    finally:
        conn.close()


# =============================
# DEVICE LIST (NOW USING tc_devices.name IN UI)
# =============================
@st.cache_data(ttl=60)
def fetch_device_map():
    """
    Returns a DataFrame with columns:
      - deviceid (tc_devices.id)
      - device_name (tc_devices.name)
    Only includes devices that actually have positions after NOISE_CUTOFF.
    """
    q = f"""
        SELECT DISTINCT
            d.id   AS deviceid,
            d.name AS device_name
        FROM {DEVICES_TABLE} d
        INNER JOIN {POSITIONS_TABLE} p
            ON p.deviceid = d.id
        WHERE p.fixtime >= %s
        ORDER BY d.name;
    """
    df_map = fetch_df(q, (NOISE_CUTOFF,))
    if df_map.empty:
        return df_map

    df_map["deviceid"] = pd.to_numeric(df_map["deviceid"], errors="coerce")
    df_map["device_name"] = df_map["device_name"].astype(str)
    df_map = df_map.dropna(subset=["deviceid", "device_name"]).copy()
    df_map["deviceid"] = df_map["deviceid"].astype(int)

    # De-dupe names (in case of unexpected duplicates)
    df_map = df_map.drop_duplicates(subset=["deviceid"], keep="first")
    return df_map


device_map_df = fetch_device_map()
device_name_options = device_map_df["device_name"].tolist() if not device_map_df.empty else []


# =============================
# SIDEBAR FILTERS
# =============================
st.sidebar.header("Filters")

today = date.today()
default_start = today - timedelta(days=7)
default_end = today

start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)

if start_date > end_date:
    st.sidebar.error("Start date must be <= End date.")
    st.stop()

# --- CHANGED: Select names instead of IDs ---
selected_device_names = st.sidebar.multiselect(
    "Select up to 5 bike names",
    options=device_name_options,
    default=device_name_options[:1] if device_name_options else [],
)

if len(selected_device_names) == 0:
    st.info("Select at least one bike name to begin.")
    st.stop()

if len(selected_device_names) > MAX_BIKES:
    st.sidebar.error(f"Please select at most {MAX_BIKES} bikes.")
    st.stop()

# Map selected names -> device IDs (used everywhere else)
name_to_id = dict(zip(device_map_df["device_name"], device_map_df["deviceid"])) if not device_map_df.empty else {}
selected_devices = [int(name_to_id[n]) for n in selected_device_names if n in name_to_id]

if len(selected_devices) == 0:
    st.warning("No matching device IDs found for the selected names.")
    st.stop()

start_dt, end_dt = dt_range_inclusive(start_date, end_date)
num_days = (end_date - start_date).days + 1


# =============================
# LOAD POSITIONS
# =============================
@st.cache_data(ttl=60)
def fetch_positions(device_list, start_dt, end_dt):
    device_list = [int(x) for x in device_list]
    placeholders = ",".join(["%s"] * len(device_list))

    q = f"""
        SELECT
            id,
            protocol,
            deviceid,
            servertime,
            devicetime,
            fixtime,
            valid,
            latitude,
            longitude,
            altitude,
            speed,
            course,
            address,
            attributes,

            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.event')) AS event,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.ignition')) AS ignition_raw,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.door')) AS door_raw,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.motion')) AS motion_raw,

            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.fuel1')) AS DECIMAL(10,4)) AS fuel1,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.fuel2')) AS DECIMAL(18,4)) AS fuel2,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.temp1')) AS DECIMAL(18,0)) AS temp1,

            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.distance')) AS DECIMAL(18,8)) AS distance,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.totalDistance')) AS DECIMAL(18,8)) AS totalDistance

        FROM {POSITIONS_TABLE}
        WHERE deviceid IN ({placeholders})
          AND fixtime BETWEEN %s AND %s
          AND fixtime >= %s
          AND valid = 1
          AND latitude <> 0 AND longitude <> 0
        ORDER BY deviceid, fixtime;
    """

    params = tuple(device_list) + (start_dt, end_dt, NOISE_CUTOFF)
    df = fetch_df(q, params)

    if df.empty:
        return df

    for c in ["servertime", "devicetime", "fixtime"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["speed_kmh"] = df["speed"].apply(knots_to_kmh)

    # ✅ ONLY CHANGE: clamp/blank speeds above 85 km/h
    df.loc[df["speed_kmh"] > SPEED_CUTOFF_KMH, "speed_kmh"] = 0.0

    df["ignition"] = df["ignition_raw"].apply(to_bool)
    df["door"] = df["door_raw"].apply(to_bool)      # charging ON/OFF
    df["motion"] = df["motion_raw"].apply(to_bool)

    # Fallback parsing in case JSON_EXTRACT returns null
    if df["fuel1"].isna().all() or df["door"].isna().all() or df["distance"].isna().all():
        attrs = df["attributes"].apply(safe_json_load)
        if df["fuel1"].isna().all():
            df["fuel1"] = attrs.apply(lambda a: a.get("fuel1", np.nan))
        if df["door"].isna().all():
            df["door"] = attrs.apply(lambda a: a.get("door", None))
        if df["temp1"].isna().all():
            df["temp1"] = attrs.apply(lambda a: a.get("temp1", np.nan))
        if df["distance"].isna().all():
            df["distance"] = attrs.apply(lambda a: a.get("distance", np.nan))
        if df["totalDistance"].isna().all():
            df["totalDistance"] = attrs.apply(lambda a: a.get("totalDistance", np.nan))

    return df


with st.spinner("Loading positions..."):
    df = fetch_positions(selected_devices, start_dt, end_dt)

if df.empty:
    st.warning("No data returned for the selected filters (after noise/valid filtering). Try a wider date range.")
    st.stop()

# --- OPTIONAL (but kept minimal): add device_name column for reference/use later if needed ---
if not device_map_df.empty and "device_name" not in df.columns:
    df = df.merge(device_map_df, on="deviceid", how="left")


# =============================
# NEW: DISTANCE TRAVELLED OVER TIME (USING attributes.distance IN METERS)
# =============================
def build_distance_over_time_using_distance(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds an *increasing* distance travelled curve by using attributes.distance as an increment.

    Assumption (based on your dataset):
    - distance is the distance travelled since the previous point (increment)
    - it is in METERS
    Therefore:
    - per_row_km = distance / 1000
    - cumulative_km = cumsum(per_row_km)
    Also applies sanity clamping based on time delta between points.
    """
    d = raw_df.dropna(subset=["fixtime", "deviceid"]).copy()
    if d.empty:
        return d

    d = d.sort_values(["deviceid", "fixtime"]).reset_index(drop=True)

    d["distance"] = pd.to_numeric(d.get("distance"), errors="coerce")
    d["distance_step_km"] = np.nan
    d["distance_travelled_km"] = np.nan

    for deviceid, g in d.groupby("deviceid", sort=False):
        g = g.sort_values("fixtime").copy()

        # meters -> km (treating distance as increment)
        step_km = (g["distance"].astype(float) / 1000.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        step_km = step_km.clip(lower=0.0)

        # time-based sanity clamp: step cannot exceed plausible speed * time gap
        dt_sec = g["fixtime"].diff().dt.total_seconds().fillna(0.0)
        max_km = (MAX_PLAUSIBLE_SPEED_KMH * (dt_sec / 3600.0)) * SANITY_BUFFER
        max_km = max_km.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

        step_km = np.minimum(step_km.values, max_km.values)

        # Build increasing curve
        cumulative_km = np.cumsum(step_km)

        d.loc[g.index, "distance_step_km"] = step_km
        d.loc[g.index, "distance_travelled_km"] = cumulative_km

    return d


dist_time_df = build_distance_over_time_using_distance(df)


# =============================
# TABS
# =============================
tab_overview, tab_graphs, tab_popular, tab_route = st.tabs(
    ["Overview", "Graphs", "Popular locations", "Route map"]
)


# =============================
# OVERVIEW TAB
# =============================
with tab_overview:
    card_open()
    st.subheader("Overview (per selected bike, per day)")

    # ✅ ONLY CHANGE: show chassis number (tc_devices.name) instead of deviceid in the table
    deviceid_to_chassis = dict(zip(device_map_df["deviceid"], device_map_df["device_name"])) if not device_map_df.empty else {}

    df_day = df.dropna(subset=["fixtime"]).copy()
    df_day["date"] = df_day["fixtime"].dt.date

    # ✅ ONLY CHANGE (1): ensure each selected date renders (even if no rows for some bikes)
    for day in pd.date_range(start_date, end_date).date:
        day_df = df_day[df_day["date"] == day]
        daily_rows = []

        for deviceid in selected_devices:
            g = day_df[day_df["deviceid"] == deviceid].sort_values("fixtime")

            if g.empty:
                daily_rows.append({
                    "Date": day,
                    "Chassis number": deviceid_to_chassis.get(int(deviceid), str(int(deviceid))),
                    "Avg speed (km/h) [zeros ignored]": np.nan,
                    "Max speed (km/h)": np.nan,
                    "Total distance (km)": np.nan,
                    "Points": 0,
                })
                continue

            nz = g.loc[g["speed_kmh"] > 0, "speed_kmh"].dropna()
            avg_speed = float(nz.mean()) if len(nz) else np.nan
            max_speed = float(g["speed_kmh"].max()) if g["speed_kmh"].notna().any() else np.nan

            dd = g["distance"].dropna()
            total_dist = float(dd.sum()) if len(dd) else np.nan  # meters

            daily_rows.append({
                "Date": day,
                "Chassis number": deviceid_to_chassis.get(int(deviceid), str(int(deviceid))),
                "Avg speed (km/h) [zeros ignored]": avg_speed,
                "Max speed (km/h)": max_speed,
                "Total distance (km)": (total_dist / 1000.0) if not np.isnan(total_dist) else np.nan,
                "Points": int(len(g)),
            })

        st.markdown(f"### {day}")
        st.dataframe(
            pd.DataFrame(daily_rows).sort_values("Chassis number"),
            use_container_width=True,
        )

    card_close()


# =============================
# GRAPHS TAB
# =============================
with tab_graphs:
    card_open()
    st.subheader("Graphs")

    st.markdown("### 1) Speed over time (km/h)")
    st.altair_chart(
        alt_line(df.dropna(subset=["fixtime", "speed_kmh"]), "fixtime:T", "speed_kmh:Q", "deviceid:N",
                 "Speed over time", "Speed (km/h)"),
        use_container_width=True,
    )

    # --- UPDATED GRAPH (USING distance increments in meters -> cumulative km) ---
    st.markdown("### 2) Distance travelled over time")
    st.caption(
        "This is now computed using attributes.distance (treated as meters travelled since the previous point). "
        "We convert meters to km and plot a cumulative increasing curve."
    )
    st.altair_chart(
        alt_line(
            dist_time_df.dropna(subset=["fixtime", "distance_travelled_km"]),
            "fixtime:T",
            "distance_travelled_km:Q",
            "deviceid:N",
            "Distance travelled over time (cumulative from attributes.distance)",
            "Distance travelled (km)",
        ),
        use_container_width=True,
    )

    st.markdown("### 3) Temp1 over time")
    st.altair_chart(
        alt_line(df.dropna(subset=["fixtime", "temp1"]), "fixtime:T", "temp1:Q", "deviceid:N",
                 "Temp1 over time", "Temp1"),
        use_container_width=True,
    )
    card_close()


# =============================
# POPULAR LOCATIONS TAB
# =============================
with tab_popular:
    card_open()
    st.subheader("Popular locations (hotspots)")
    st.caption("Hotspots are computed by binning lat/lon into grid cells and counting visits.")

    pop = df.dropna(subset=["latitude", "longitude"]).copy()
    if pop.empty:
        st.warning("No valid points to compute hotspots.")
    else:
        precision = 3
        top_n = 500

        pop["lat_bin"] = pop["latitude"].round(precision)
        pop["lon_bin"] = pop["longitude"].round(precision)

        hotspot = (
            pop.groupby(["lat_bin", "lon_bin"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .head(top_n)
        )

        # ✅ ONLY CHANGE: consider popular only if visits > 20
        hotspot = hotspot[hotspot["count"] > 200].copy()

        hotspot["label"] = hotspot.apply(
            lambda r: f"Visits: {int(r['count'])}\nLat: {r['lat_bin']}\nLon: {r['lon_bin']}",
            axis=1
        )

        # ---- ORANGE PIN + BIG LIGHT ORANGE HALO ----
        hotspot["count"] = pd.to_numeric(hotspot["count"], errors="coerce").fillna(1).astype(int)

        DOT_RADIUS = 22      # pinpoint
        HALO_RADIUS = 260    # big halo

        # ✅ ONLY CHANGE (Popular locations button): store / use focus location for map
        if "hotspot_focus" not in st.session_state:
            st.session_state["hotspot_focus"] = None

        focus = st.session_state.get("hotspot_focus")
        if focus is not None:
            focus_lat, focus_lon = float(focus["lat"]), float(focus["lon"])
            center_lat, center_lon = focus_lat, focus_lon
            zoom_level = 16
        else:
            center_lat, center_lon = float(pop["latitude"].mean()), float(pop["longitude"].mean())
            zoom_level = 12

        halo_layer = pdk.Layer(
            "ScatterplotLayer",
            data=hotspot,
            get_position="[lon_bin, lat_bin]",
            get_radius=HALO_RADIUS,
            stroked=False,
            filled=True,
            pickable=False,
            # light orange halo
            get_fill_color=[255, 165, 0, 40],
        )

        dot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=hotspot,
            get_position="[lon_bin, lat_bin]",
            get_radius=DOT_RADIUS,
            stroked=True,
            filled=True,
            pickable=True,
            line_width_min_pixels=1,
            # solid orange point
            get_fill_color=[255, 140, 0, 220],
            get_line_color=[180, 90, 0, 230],
        )

        # Optional focus ring so selected hotspot stands out even when zoomed in
        focus_layer = None
        if focus is not None:
            focus_df = pd.DataFrame([{"lat": focus_lat, "lon": focus_lon, "label": focus.get("label", "")}])
            focus_layer = pdk.Layer(
                "ScatterplotLayer",
                data=focus_df,
                get_position="[lon, lat]",
                get_radius=520,
                stroked=True,
                filled=False,
                pickable=False,
                line_width_min_pixels=3,
                get_line_color=[255, 140, 0, 210],
            )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom_level,
            pitch=0,
        )

        layers = [halo_layer, dot_layer] + ([focus_layer] if focus_layer is not None else [])

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=view_state,
                layers=layers,
                tooltip={"text": "{label}"},
            )
        )

        # ✅ ONLY CHANGE (Popular locations button): buttons next to each hotspot row
        st.markdown("### Hotspot table (click **View on map**)")
        hotspot_tbl = hotspot[["lat_bin", "lon_bin", "count"]].copy()
        hotspot_tbl = hotspot_tbl.sort_values("count", ascending=False).reset_index(drop=True)

        header_cols = st.columns([2.2, 2.2, 1.2, 1.6])
        header_cols[0].markdown("**Latitude**")
        header_cols[1].markdown("**Longitude**")
        header_cols[2].markdown("**Visits**")
        header_cols[3].markdown("**Action**")

        for i, r in hotspot_tbl.iterrows():
            c0, c1, c2, c3 = st.columns([2.2, 2.2, 1.2, 1.6])
            c0.write(float(r["lat_bin"]))
            c1.write(float(r["lon_bin"]))
            c2.write(int(r["count"]))
            if c3.button("View on map", key=f"hotspot_view_{i}"):
                st.session_state["hotspot_focus"] = {
                    "lat": float(r["lat_bin"]),
                    "lon": float(r["lon_bin"]),
                    "label": f"Visits: {int(r['count'])}\nLat: {float(r['lat_bin'])}\nLon: {float(r['lon_bin'])}",
                }
                st.rerun()

    card_close()


# =============================
# ROUTE MAP TAB
# =============================
with tab_route:
    card_open()
    st.subheader("Route map (polyline)")

    # ✅ ONLY CHANGE (3): choose chassis number (names) + choose date from selected range
    device_name_to_id = dict(zip(device_map_df["device_name"], device_map_df["deviceid"])) if not device_map_df.empty else {}
    route_device_name = st.selectbox("Choose one bike (chassis number)", options=selected_device_names)
    one_device = int(device_name_to_id[route_device_name])

    selected_route_date = st.selectbox(
        "Choose date for route map",
        options=list(pd.date_range(start_date, end_date).date),
    )

    route = df[df["deviceid"] == one_device].sort_values("fixtime").copy()
    route = route.dropna(subset=["latitude", "longitude", "fixtime"])

    # filter route to selected date
    route = route[route["fixtime"].dt.date == selected_route_date].copy()

    if route.empty:
        st.warning("No route points found for this device on the selected date.")
    else:
        path = route[["longitude", "latitude"]].values.tolist()
        path_df = pd.DataFrame([{"deviceid": int(one_device), "path": path}])

        # ✅ ONLY CHANGE: decrease line width to better match street width (keep neon blue)
        path_layer = pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            get_width=2,               # thinner
            width_min_pixels=2,        # still visible at different zoom levels
            rounded=True,
            get_color=[0, 255, 255],   # neon blue (cyan)
            pickable=False,
        )

        start_pt = route.iloc[0]
        end_pt = route.iloc[-1]

        markers = pd.DataFrame([
            {"name": "Start", "lon": float(start_pt["longitude"]), "lat": float(start_pt["latitude"])},
            {"name": "End", "lon": float(end_pt["longitude"]), "lat": float(end_pt["latitude"])},
        ])

        marker_layer = pdk.Layer(
            "ScatterplotLayer",
            data=markers,
            get_position="[lon, lat]",
            get_radius=75,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=float(route["latitude"].mean()),
            longitude=float(route["longitude"].mean()),
            zoom=12,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=view_state,
                layers=[path_layer, marker_layer],
                tooltip={"text": "{name}"},
            )
        )

        st.markdown("### Route points preview")
        st.dataframe(
            route[["fixtime", "latitude", "longitude", "speed_kmh", "fuel1", "door", "ignition"]].head(300),
            use_container_width=True,
        )
    card_close()
