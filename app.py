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
st.set_page_config(page_title="🚲 Bike GPS Analytics (Traccar)", layout="wide")
st.title("🚲 Bike GPS Analytics (Traccar)")
st.caption("Keyed by tc_positions.deviceid. Timeline uses fixtime. Noise points are excluded by default.")


# =============================
# CONSTANTS
# =============================
DB_SCHEMA = "traccar_new"
POSITIONS_TABLE = f"{DB_SCHEMA}.tc_positions"
DEVICES_TABLE = f"{DB_SCHEMA}.tc_devices"
MAX_BIKES = 5
NOISE_CUTOFF = datetime(2000, 1, 1)


# =============================
# HELPERS
# =============================
def dt_range_inclusive(start_d: date, end_d: date):
    return datetime.combine(start_d, time.min), datetime.combine(end_d, time.max)


def knots_to_kmh(knots):
    try:
        return float(knots) * 1.852 if knots is not None else np.nan
    except Exception:
        return np.nan


def safe_json_load(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def to_bool(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return str(x).lower() in ("true", "1", "yes")


def format_hms(seconds):
    if not seconds or seconds <= 0:
        return "0:00:00"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def split_session_by_day(start_ts, end_ts):
    out = []
    cur = start_ts
    while cur.date() < end_ts.date():
        day_end = datetime.combine(cur.date(), time.max)
        out.append((cur.date(), (day_end - cur).total_seconds()))
        cur = datetime.combine(cur.date() + timedelta(days=1), time.min)
    out.append((cur.date(), (end_ts - cur).total_seconds()))
    return out


def alt_line(df, x, y, color, title, y_title=None):
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(x, title="Time"),
            y=alt.Y(y, title=y_title or y),
            color=alt.Color(color, legend=alt.Legend(title="Device")),
            tooltip=[color, x, y],
        )
        .properties(height=360, title=title)
        .interactive()
    )


# =============================
# DB CONNECTION
# =============================
import pymysql

@st.cache_resource
def get_conn():
    cfg = st.secrets["mysql"]
    return pymysql.connect(
        host=cfg["host"],
        port=int(cfg.get("port", 3306)),
        user=cfg["username"],
        password=cfg["password"],
        database=cfg["database"],
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def fetch_df(query, params):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(query, params)
        return pd.DataFrame(cur.fetchall())


# =============================
# DEVICE MAP
# =============================
@st.cache_data(ttl=60)
def fetch_device_map():
    q = f"""
        SELECT DISTINCT d.id AS deviceid, d.name AS device_name
        FROM {DEVICES_TABLE} d
        JOIN {POSITIONS_TABLE} p ON p.deviceid = d.id
        WHERE p.fixtime >= %s
        ORDER BY d.name
    """
    return fetch_df(q, (NOISE_CUTOFF,))


device_map_df = fetch_device_map()
device_name_options = device_map_df["device_name"].tolist()


# =============================
# SIDEBAR
# =============================
st.sidebar.header("Filters")

start_date = st.sidebar.date_input("Start date", date.today() - timedelta(days=7))
end_date = st.sidebar.date_input("End date", date.today())

selected_names = st.sidebar.multiselect(
    "Select up to 5 bike names",
    device_name_options,
    default=device_name_options[:1]
)

name_to_id = dict(zip(device_map_df["device_name"], device_map_df["deviceid"]))
selected_devices = [name_to_id[n] for n in selected_names]

start_dt, end_dt = dt_range_inclusive(start_date, end_date)
num_days = (end_date - start_date).days + 1


# =============================
# LOAD POSITIONS
# =============================
@st.cache_data(ttl=60)
def fetch_positions(devices, start_dt, end_dt):
    placeholders = ",".join(["%s"] * len(devices))
    q = f"""
        SELECT
            deviceid, fixtime, latitude, longitude, speed,
            JSON_UNQUOTE(JSON_EXTRACT(attributes,'$.distance')) AS distance,
            JSON_UNQUOTE(JSON_EXTRACT(attributes,'$.fuel1')) AS fuel1,
            JSON_UNQUOTE(JSON_EXTRACT(attributes,'$.door')) AS door
        FROM {POSITIONS_TABLE}
        WHERE deviceid IN ({placeholders})
          AND fixtime BETWEEN %s AND %s
          AND fixtime >= %s
          AND valid = 1
          AND latitude <> 0 AND longitude <> 0
        ORDER BY deviceid, fixtime
    """
    params = tuple(devices) + (start_dt, end_dt, NOISE_CUTOFF)
    df = fetch_df(q, params)
    df["fixtime"] = pd.to_datetime(df["fixtime"])
    df["speed_kmh"] = df["speed"].apply(knots_to_kmh)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce").fillna(0)
    return df


df = fetch_positions(selected_devices, start_dt, end_dt)


# =============================
# DISTANCE OVER TIME (FINAL LOGIC)
# =============================
def build_distance_over_time(df):
    d = df.copy()
    d["distance_step_km"] = d["distance"] / 1000.0
    d["distance_travelled_km"] = (
        d.groupby("deviceid")["distance_step_km"].cumsum()
    )
    return d


dist_time_df = build_distance_over_time(df)


# =============================
# TABS
# =============================
tab_overview, tab_graphs = st.tabs(["Overview", "Graphs"])


# =============================
# OVERVIEW TAB (FINAL LOGIC)
# =============================
with tab_overview:
    rows = []
    for deviceid, g in df.groupby("deviceid"):
        total_km = g["distance"].sum() / 1000.0
        avg_daily = total_km / num_days if num_days else np.nan
        rows.append({
            "Chassis number": device_map_df.set_index("deviceid").loc[deviceid, "device_name"],
            "Avg daily distance (km)": avg_daily,
            "Total distance in range (km)": total_km,
            "Avg speed (km/h)": g.loc[g["speed_kmh"] > 0, "speed_kmh"].mean(),
            "Max speed (km/h)": g["speed_kmh"].max(),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =============================
# GRAPHS TAB (FINAL LOGIC)
# =============================
with tab_graphs:
    st.markdown("### Distance travelled over time")
    st.altair_chart(
        alt_line(
            dist_time_df,
            "fixtime:T",
            "distance_travelled_km:Q",
            "deviceid:N",
            "Distance travelled over time",
            "Distance (km)"
        ),
        use_container_width=True
    )
