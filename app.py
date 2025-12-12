import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, date

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.exc import OperationalError
import pydeck as pdk


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="🚲 Bike GPS Analytics (Traccar)", layout="wide")
st.title("🚲 Bike GPS Analytics (Traccar)")
st.caption("Distance, speed, trips, charging, daily active bikes, popular locations (tc_positions).")


# -----------------------------
# Helpers
# -----------------------------
def safe_json(x):
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

def knots_to_kmh(knots):
    return float(knots) * 1.852

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def pick_time_field(df):
    for c in ["fixtime", "devicetime", "servertime"]:
        if c in df.columns:
            return c
    return None

def sanitize_err(e: Exception) -> str:
    s = str(e)
    try:
        pwd = st.secrets["traccar"]["password"]
        if pwd:
            s = s.replace(pwd, "***")
    except Exception:
        pass
    return s


# -----------------------------
# Secrets check
# -----------------------------
if "traccar" not in st.secrets:
    st.error("Missing [traccar] in secrets. Add it in .streamlit/secrets.toml or Streamlit Cloud Secrets.")
    st.stop()

cfg = dict(st.secrets["traccar"])
for k in ["host", "port", "user", "password", "database"]:
    if k not in cfg:
        st.error(f"Missing '{k}' in [traccar] secrets.")
        st.stop()

HOST = cfg["host"]
PORT = int(cfg["port"])
USER = cfg["user"]
PASSWORD = cfg["password"]
DB_NAME = cfg["database"]  # ✅ should be traccar_new


# -----------------------------
# Engine
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_engine():
    url = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}?charset=utf8mb4"
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": 10},
    )

def test_connection():
    try:
        eng = get_engine()
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        return True, "Connected ✅"
    except OperationalError as e:
        return False, sanitize_err(getattr(e, "orig", e))


# -----------------------------
# Sidebar: Connection + Filters
# -----------------------------
with st.sidebar:
    st.header("Database")
    st.write(f"Schema: **{DB_NAME}**")

    ok, info = test_connection()
    if ok:
        st.success(info)
    else:
        st.error("Connection failed ❌")
        st.code(info)
        st.stop()

    st.divider()
    st.header("Filters")

    # Load devices
    try:
        with get_engine().connect() as c:
            devices_df = pd.read_sql(text("SELECT id, name, uniqueid FROM tc_devices ORDER BY name"), c)
    except Exception as e:
        st.error("Connected, but failed to query tc_devices.")
        st.code(sanitize_err(e))
        st.stop()

    if devices_df.empty:
        st.warning("No devices found in tc_devices.")
        st.stop()

    name_to_id = dict(zip(devices_df["name"], devices_df["id"]))
    id_to_name = dict(zip(devices_df["id"], devices_df["name"]))

    selected_names = st.multiselect(
        "Select bikes/devices",
        options=list(name_to_id.keys()),
        default=list(name_to_id.keys())[: min(5, len(name_to_id))],
    )
    device_ids = [int(name_to_id[n]) for n in selected_names]

    today = date.today()
    start_date = st.date_input("Start date", value=today - timedelta(days=7))
    end_date = st.date_input("End date (inclusive)", value=today)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

if not device_ids:
    st.info("Select at least one device.")
    st.stop()


# -----------------------------
# Load positions (correct IN expanding)
# -----------------------------
q_positions = (
    text("""
        SELECT
            id, deviceid,
            servertime, devicetime, fixtime,
            latitude, longitude, altitude,
            speed, course,
            valid, accuracy,
            attributes
        FROM tc_positions
        WHERE deviceid IN :device_ids
          AND fixtime >= :start_dt
          AND fixtime < :end_dt
        ORDER BY deviceid, fixtime
    """)
    .bindparams(bindparam("device_ids", expanding=True))
)

try:
    with get_engine().connect() as c:
        positions = pd.read_sql(
            q_positions,
            c,
            params={"device_ids": device_ids, "start_dt": start_dt, "end_dt": end_dt},
        )
except Exception as e:
    st.error("Failed to query tc_positions.")
    st.code(sanitize_err(e))
    st.stop()

for col in ["servertime", "devicetime", "fixtime"]:
    if col in positions.columns:
        positions[col] = pd.to_datetime(positions[col], errors="coerce")

time_col = pick_time_field(positions)
if positions.empty or time_col is None:
    st.warning("No position data found for the selected devices/date range.")
    st.stop()

positions = positions.dropna(subset=["latitude", "longitude", time_col]).copy()
positions["speed_kmh"] = pd.to_numeric(positions["speed"], errors="coerce").fillna(0.0).apply(knots_to_kmh)

positions = positions.sort_values(["deviceid", time_col]).copy()
positions["prev_lat"] = positions.groupby("deviceid")["latitude"].shift(1)
positions["prev_lon"] = positions.groupby("deviceid")["longitude"].shift(1)

seg = haversine_km(
    positions["prev_lat"].fillna(positions["latitude"]),
    positions["prev_lon"].fillna(positions["longitude"]),
    positions["latitude"],
    positions["longitude"],
)
positions["seg_km"] = np.where(positions["prev_lat"].isna(), 0.0, seg)
positions["day"] = positions[time_col].dt.date


# -----------------------------
# Metrics
# -----------------------------
per_bike = (
    positions.groupby("deviceid", as_index=False)
    .agg(
        points=("id", "count"),
        total_km=("seg_km", "sum"),
        max_kmh=("speed_kmh", "max"),
        avg_kmh=("speed_kmh", "mean"),
        first_time=(time_col, "min"),
        last_time=(time_col, "max"),
    )
)
per_bike["device_name"] = per_bike["deviceid"].map(lambda i: id_to_name.get(int(i), str(i)))

overall_avg_speed = float(per_bike["avg_kmh"].mean()) if len(per_bike) else 0.0
overall_avg_distance = float(per_bike["total_km"].mean()) if len(per_bike) else 0.0
overall_max_speed = float(per_bike["max_kmh"].max()) if len(per_bike) else 0.0

ACTIVE_KM_THRESHOLD = 0.2
daily_km = positions.groupby(["day", "deviceid"], as_index=False)["seg_km"].sum()
daily_km["active"] = daily_km["seg_km"] >= ACTIVE_KM_THRESHOLD
daily_active = daily_km.groupby("day", as_index=False)["active"].sum().rename(columns={"active": "active_bikes"})

# Popular locations (grid)
grid_decimals = 3
positions["cell_lat"] = np.round(positions["latitude"], grid_decimals)
positions["cell_lon"] = np.round(positions["longitude"], grid_decimals)
top_cells = (
    positions.groupby(["cell_lat", "cell_lon"], as_index=False)
    .agg(points=("id", "count"))
    .sort_values("points", ascending=False)
    .head(50)
)


# -----------------------------
# UI
# -----------------------------
tab_overview, tab_bike, tab_locations, tab_a_to_b = st.tabs(
    ["Overview", "Bike Metrics", "Popular Locations", "Distance A → B"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bikes selected", len(device_ids))
    c2.metric("Avg speed (all bikes)", f"{overall_avg_speed:.1f} km/h")
    c3.metric("Avg distance (per bike)", f"{overall_avg_distance:.2f} km")
    c4.metric("Max speed (any bike)", f"{overall_max_speed:.1f} km/h")

    st.subheader("Daily active bikes")
    st.dataframe(daily_active, use_container_width=True)
    st.line_chart(daily_active.set_index("day")["active_bikes"])

    st.subheader("Total distance per bike")
    st.dataframe(per_bike.sort_values("total_km", ascending=False), use_container_width=True)
    st.bar_chart(per_bike.set_index("device_name")["total_km"])

with tab_bike:
    st.subheader("Per-bike metrics")
    st.dataframe(per_bike.sort_values("device_name"), use_container_width=True)

    chosen_bike = st.selectbox("Choose a bike", options=per_bike["device_name"].tolist())
    chosen_id = int(per_bike.loc[per_bike["device_name"] == chosen_bike, "deviceid"].iloc[0])

    bike_df = positions[positions["deviceid"] == chosen_id].sort_values(time_col).copy()

    st.markdown("**Speed over time (km/h)**")
    st.line_chart(bike_df.set_index(time_col)["speed_kmh"])

    st.markdown("**Daily distance (km)**")
    bike_daily = bike_df.groupby("day", as_index=False)["seg_km"].sum().rename(columns={"seg_km": "distance_km"})
    st.bar_chart(bike_daily.set_index("day")["distance_km"])

with tab_locations:
    st.subheader("Popular location cells (top 50)")
    st.dataframe(top_cells, use_container_width=True)

    st.subheader("Latest positions map")
    latest = positions.sort_values(["deviceid", time_col]).groupby("deviceid", as_index=False).tail(1).copy()
    latest["device_name"] = latest["deviceid"].map(lambda i: id_to_name.get(int(i), str(i)))

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=latest,
        get_position="[longitude, latitude]",
        get_radius=40,
        pickable=True,
    )
    view = pdk.ViewState(
        latitude=float(latest["latitude"].mean()),
        longitude=float(latest["longitude"].mean()),
        zoom=11,
        pitch=0,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "{device_name}\nSpeed: {speed_kmh} km/h"},
        )
    )

with tab_a_to_b:
    st.subheader("Distance from point A → point B (route distance)")
    bike_name = st.selectbox("Bike", options=per_bike["device_name"].tolist(), key="a2b_bike")
    bike_id = int(per_bike.loc[per_bike["device_name"] == bike_name, "deviceid"].iloc[0])
    bike_df = positions[positions["deviceid"] == bike_id].sort_values(time_col).copy()

    min_t = bike_df[time_col].min().to_pydatetime()
    max_t = bike_df[time_col].max().to_pydatetime()

    colA, colB = st.columns(2)
    with colA:
        tA = st.datetime_input("Point A time", value=min_t, min_value=min_t, max_value=max_t)
    with colB:
        tB = st.datetime_input("Point B time", value=max_t, min_value=min_t, max_value=max_t)

    if tA > tB:
        st.error("Point A time must be <= Point B time.")
    else:
        segment = bike_df[(bike_df[time_col] >= pd.to_datetime(tA)) & (bike_df[time_col] <= pd.to_datetime(tB))].copy()
        if len(segment) < 2:
            st.warning("Not enough points between A and B to compute route distance.")
        else:
            dist = float(segment["seg_km"].sum())
            st.success(f"Route distance A → B: **{dist:.2f} km**")

st.caption("✅ Using schema traccar_new. Speed: knots→km/h. Distance: haversine between consecutive points by fixtime.")
