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
    return (
        datetime.combine(start_d, time.min),
        datetime.combine(end_d, time.max),
    )


def knots_to_kmh(knots):
    try:
        return float(knots) * 1.852
    except Exception:
        return np.nan


def safe_json_load(x):
    try:
        return json.loads(x) if isinstance(x, str) else {}
    except Exception:
        return {}


def to_bool(x):
    if x is None:
        return None
    s = str(x).lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    return None


def alt_line(df, x, y, color, title, y_title):
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(x, title="Time"),
            y=alt.Y(y, title=y_title),
            color=alt.Color(color, legend=alt.Legend(title="Bike")),
            tooltip=[color, x, y],
        )
        .properties(height=360, title=title)
        .interactive()
    )


# =============================
# DATABASE CONNECTION
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
name_to_id = dict(zip(device_map_df.device_name, device_map_df.deviceid))


# =============================
# SIDEBAR
# =============================
st.sidebar.header("Filters")

start_date = st.sidebar.date_input("Start date", date.today() - timedelta(days=7))
end_date = st.sidebar.date_input("End date", date.today())

selected_names = st.sidebar.multiselect(
    "Select up to 5 bikes",
    options=device_map_df.device_name.tolist(),
    default=device_map_df.device_name.head(1).tolist(),
)

selected_devices = [name_to_id[n] for n in selected_names]

start_dt, end_dt = dt_range_inclusive(start_date, end_date)
num_days = (end_date - start_date).days + 1


# =============================
# LOAD POSITIONS
# =============================
@st.cache_data(ttl=60)
def fetch_positions(device_ids, start_dt, end_dt):
    placeholders = ",".join(["%s"] * len(device_ids))
    q = f"""
        SELECT
            deviceid,
            fixtime,
            latitude,
            longitude,
            speed,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.distance')) AS distance,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.fuel1')) AS fuel1,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.door')) AS door
        FROM {POSITIONS_TABLE}
        WHERE deviceid IN ({placeholders})
          AND fixtime BETWEEN %s AND %s
          AND valid = 1
        ORDER BY deviceid, fixtime
    """
    df = fetch_df(q, tuple(device_ids) + (start_dt, end_dt))
    df["fixtime"] = pd.to_datetime(df["fixtime"])
    df["speed_kmh"] = df["speed"].apply(knots_to_kmh)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce").fillna(0)
    return df


df = fetch_positions(selected_devices, start_dt, end_dt)
df = df.merge(device_map_df, on="deviceid", how="left")


# =============================
# ✅ CLEAN DISTANCE–TIME LOGIC (FIXED)
# =============================
def build_distance_time(df):
    """
    Correct distance–time curve:
    - uses ONLY attributes.distance
    - meters → km
    - cumulative sum per device
    """
    out = []

    for deviceid, g in df.groupby("deviceid"):
        g = g.sort_values("fixtime").copy()
        g["distance_km"] = g["distance"] / 1000.0
        g["distance_travelled_km"] = g["distance_km"].cumsum()
        out.append(g)

    return pd.concat(out, ignore_index=True)


dist_time_df = build_distance_time(df)


# =============================
# TABS
# =============================
tab_overview, tab_graphs = st.tabs(["Overview", "Graphs"])


# =============================
# OVERVIEW
# =============================
with tab_overview:
    rows = []

    for deviceid, g in df.groupby("deviceid"):
        total_km = g["distance"].sum() / 1000.0
        rows.append({
            "Bike": g.device_name.iloc[0],
            "Total distance (km)": total_km,
            "Avg daily distance (km)": total_km / num_days if num_days else 0,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =============================
# GRAPHS
# =============================
with tab_graphs:
    st.markdown("### Distance travelled over time")

    st.altair_chart(
        alt_line(
            dist_time_df,
            "fixtime:T",
            "distance_travelled_km:Q",
            "device_name:N",
            "Distance travelled over time",
            "Distance (km)",
        ),
        use_container_width=True,
    )
