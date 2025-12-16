import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, date

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import URL
import pydeck as pdk


# =============================
# UI
# =============================
st.set_page_config(page_title="🚲 Bike GPS Analytics (Traccar)", layout="wide")
st.title("🏍️ Bike GPS Analytics (Traccar)")
st.caption("Bikes are uniquely identified by chassis number = tc_devices.name. Analytics is sourced from tc_positions.")


# =============================
# Helpers
# =============================
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
    """Vectorized haversine distance in KM."""
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

def to_plain_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def to_plain_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)

def clean_for_pydeck_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure pydeck only receives JSON-serializable values.
    Expects: latitude, longitude, chassis_no, speed_kmh
    """
    if df.empty:
        return df
    out = df.copy()

    keep = [c for c in ["latitude", "longitude", "chassis_no", "speed_kmh"] if c in out.columns]
    out = out[keep].copy()

    if "latitude" in out.columns:
        out["latitude"] = out["latitude"].apply(to_plain_float)
    if "longitude" in out.columns:
        out["longitude"] = out["longitude"].apply(to_plain_float)
    if "speed_kmh" in out.columns:
        out["speed_kmh"] = out["speed_kmh"].apply(to_plain_float)
    if "chassis_no" in out.columns:
        out["chassis_no"] = out["chassis_no"].apply(to_plain_str)

    out = out.dropna(subset=["latitude", "longitude"]).copy()
    out = out.replace([np.inf, -np.inf], None)
    return out

def mean_ignore_zeros(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    s = s[s > 0]
    return float(s.mean()) if len(s) else 0.0

def extract_attr_numeric_multi(series_attrs: pd.Series, keys: list[str]) -> pd.Series:
    attrs = series_attrs.apply(safe_json)

    def pick(a: dict):
        # exact keys first
        for k in keys:
            if k in a and a.get(k, None) is not None:
                return a.get(k, None)
        # fallback: case-insensitive match
        lower_map = {str(kk).lower(): kk for kk in a.keys()}
        for k in keys:
            lk = str(k).lower()
            if lk in lower_map:
                v = a.get(lower_map[lk], None)
                if v is not None:
                    return v
        return None

    out = attrs.apply(pick)
    return pd.to_numeric(out, errors="coerce")

def extract_attr_bool(series_attrs: pd.Series, key: str) -> pd.Series:
    attrs = series_attrs.apply(safe_json)
    raw = attrs.apply(lambda a: a.get(key, None))
    def to_bool(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return bool(int(v))
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ["true", "1", "yes", "y", "on"]:
                return True
            if vv in ["false", "0", "no", "n", "off"]:
                return False
        try:
            return bool(v)
        except Exception:
            return False
    return raw.apply(to_bool)


# =============================
# Secrets / Config
# =============================
if "traccar" not in st.secrets:
    st.error("Missing [traccar] in secrets.")
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
DB_NAME = cfg["database"]  # e.g. traccar_new


# =============================
# Engine
# =============================
@st.cache_resource(show_spinner=False)
def get_engine():
    url = URL.create(
        drivername="mysql+pymysql",
        username=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        database=DB_NAME,
        query={"charset": "utf8mb4"},
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": 10},
    )

def test_connection():
    try:
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True, "Connected ✅"
    except Exception as e:
        return False, str(getattr(e, "orig", e))

@st.cache_data(ttl=600, show_spinner=False)
def table_exists(table_name: str) -> bool:
    try:
        with get_engine().connect() as c:
            q = text("""
                SELECT COUNT(*) AS cnt
                FROM information_schema.tables
                WHERE table_schema = :db
                  AND table_name = :t
            """)
            r = c.execute(q, {"db": DB_NAME, "t": table_name}).mappings().first()
            return int(r["cnt"]) > 0
    except Exception:
        return False


# =============================
# DB Loads
# =============================
@st.cache_data(ttl=300, show_spinner=False)
def load_all_devices() -> pd.DataFrame:
    """
    ✅ Bike list comes from tc_devices (NOT tc_positions date-filtered).
    Chassis number = tc_devices.name
    """
    if not table_exists("tc_devices"):
        return pd.DataFrame(columns=["deviceid", "chassis_no"])
    with get_engine().connect() as c:
        df = pd.read_sql(text("SELECT id AS deviceid, name AS chassis_no FROM tc_devices ORDER BY name"), c)
    df["chassis_no"] = df["chassis_no"].fillna("").astype(str).str.strip()
    df = df[df["chassis_no"] != ""].copy()
    return df

@st.cache_data(ttl=120, show_spinner=True)
def load_positions(device_ids: list[int], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if not device_ids:
        return pd.DataFrame()
    q = (
        text("""
            SELECT
                id,
                deviceid,
                servertime,
                devicetime,
                fixtime,
                latitude,
                longitude,
                altitude,
                speed,
                course,
                valid,
                accuracy,
                attributes
            FROM tc_positions
            WHERE deviceid IN :device_ids
              AND fixtime >= :start_dt
              AND fixtime < :end_dt
            ORDER BY deviceid, fixtime
        """)
        .bindparams(bindparam("device_ids", expanding=True))
    )

    with get_engine().connect() as c:
        df = pd.read_sql(q, c, params={"device_ids": device_ids, "start_dt": start_dt, "end_dt": end_dt})

    for col in ["servertime", "devicetime", "fixtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["speed_kmh"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0).apply(knots_to_kmh)
    return df

@st.cache_data(ttl=120, show_spinner=False)
def load_geofences():
    if not table_exists("tc_geofences"):
        return pd.DataFrame(columns=["id", "name", "area", "attributes"])
    try:
        with get_engine().connect() as c:
            return pd.read_sql(text("SELECT id, name, area, attributes FROM tc_geofences ORDER BY name"), c)
    except Exception:
        return pd.DataFrame(columns=["id", "name", "area", "attributes"])

@st.cache_data(ttl=60, show_spinner=False)
def load_geofence_events(device_ids: list[int], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if (not device_ids) or (not table_exists("tc_events")):
        return pd.DataFrame(columns=["id", "deviceid", "type", "servertime", "geofenceid", "positionid", "attributes"])
    try:
        q = (
            text("""
                SELECT
                    id,
                    deviceid,
                    type,
                    servertime,
                    geofenceid,
                    positionid,
                    attributes
                FROM tc_events
                WHERE deviceid IN :device_ids
                  AND servertime >= :start_dt
                  AND servertime < :end_dt
                  AND type IN ('geofenceEnter','geofenceExit')
                ORDER BY servertime DESC
                LIMIT 500
            """)
            .bindparams(bindparam("device_ids", expanding=True))
        )

        with get_engine().connect() as c:
            df = pd.read_sql(q, c, params={"device_ids": device_ids, "start_dt": start_dt, "end_dt": end_dt})

        if not df.empty:
            df["servertime"] = pd.to_datetime(df["servertime"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=["id", "deviceid", "type", "servertime", "geofenceid", "positionid", "attributes"])


# =============================
# Charging analytics
# ✅ door=True => charging
# ✅ charge session = first True -> first False
# =============================
def detect_charging_door(df: pd.DataFrame, time_col: str):
    if df.empty or time_col is None:
        return pd.DataFrame(), pd.DataFrame()

    d = df.sort_values(["deviceid", time_col]).copy()
    d["is_charging"] = extract_attr_bool(d["attributes"], "door")

    d["prev_charge"] = d.groupby("deviceid")["is_charging"].shift(1).fillna(False)
    d["charge_start"] = (d["is_charging"] == True) & (d["prev_charge"] == False)
    d["charge_session_id"] = d["charge_start"].groupby(d["deviceid"]).cumsum()

    charging_rows = d[d["is_charging"] == True].copy()
    if charging_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    sessions = (
        charging_rows.groupby(["deviceid", "charge_session_id"], as_index=False)
        .agg(start_time=(time_col, "min"), end_time=(time_col, "max"), samples=("id", "count"))
    )
    sessions["duration_min"] = (sessions["end_time"] - sessions["start_time"]).dt.total_seconds() / 60.0
    sessions["day"] = sessions["start_time"].dt.date

    daily = (
        sessions.groupby(["deviceid", "day"], as_index=False)
        .agg(
            charging_minutes=("duration_min", "sum"),
            charges_in_day=("charge_session_id", "nunique"),
        )
    )
    daily["avg_daily_hours_of_charge"] = daily["charging_minutes"] / 60.0
    return sessions.sort_values(["deviceid", "start_time"]), daily.sort_values(["deviceid", "day"])


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Connection")
    st.write(f"Host: `{HOST}`  Port: `{PORT}`")
    st.write(f"Schema: **{DB_NAME}**")

    ok, msg = test_connection()
    if ok:
        st.success(msg)
    else:
        st.error("Connection failed ❌")
        st.code(msg)
        st.stop()

    st.divider()
    st.header("Bike selection (Chassis No.)")
    devices_df = load_all_devices()
    if devices_df.empty:
        st.warning("No devices found in tc_devices.")
        st.stop()

    chassis_list = devices_df["chassis_no"].tolist()
    chassis_to_deviceid = dict(zip(devices_df["chassis_no"], devices_df["deviceid"]))
    deviceid_to_chassis = dict(zip(devices_df["deviceid"], devices_df["chassis_no"]))

    selected_chassis = st.multiselect(
        "Select bikes (tc_devices.name)",
        options=sorted(chassis_list),
        default=sorted(chassis_list)[: min(5, len(chassis_list))],
    )
    selected_deviceids = [int(chassis_to_deviceid[ch]) for ch in selected_chassis if ch in chassis_to_deviceid]

    st.divider()
    st.header("Date range (analytics)")
    today = date.today()
    start_date = st.date_input("Start date", value=today - timedelta(days=7))
    end_date = st.date_input("End date (inclusive)", value=today)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

if not selected_deviceids:
    st.info("Select at least one bike.")
    st.stop()


# =============================
# Load positions (date range affects analytics only)
# =============================
positions = load_positions(selected_deviceids, start_dt, end_dt)
time_col = pick_time_field(positions)

if positions.empty or time_col is None:
    st.warning("No position data found for the selected bikes in this date range. Change the date range to see analytics.")
    # still show tabs, but they’ll be mostly empty
    positions = pd.DataFrame(columns=[
        "id","deviceid","servertime","devicetime","fixtime","latitude","longitude","altitude","speed","course","valid","accuracy","attributes","speed_kmh"
    ])
    time_col = "fixtime"


# =============================
# Enrich + derive signals
# =============================
if not positions.empty:
    positions = positions.dropna(subset=["latitude", "longitude", time_col]).copy()
    positions = positions.sort_values(["deviceid", time_col]).copy()

    # chassis id
    positions["chassis_no"] = positions["deviceid"].map(lambda x: str(deviceid_to_chassis.get(int(x), "")).strip())

    # Fuel 1 => SOC (%)
    positions["soc"] = extract_attr_numeric_multi(
        positions["attributes"],
        keys=["fuel1", "fuel_1", "fuel 1", "Fuel1", "FUEL1"]
    )

    # Fuel 2 => estimated time
    positions["fuel2_est_time"] = extract_attr_numeric_multi(
        positions["attributes"],
        keys=["fuel2", "fuel_2", "fuel 2", "Fuel2", "FUEL2"]
    )

    # Temp1 (kept)
    positions["temp1"] = extract_attr_numeric_multi(
        positions["attributes"],
        keys=["temp1", "Temp1", "temperature1", "temperature_1", "temperature 1"]
    )

    # Charging flag from door
    positions["is_charging"] = extract_attr_bool(positions["attributes"], "door")

    # Distances
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
else:
    positions["chassis_no"] = ""
    positions["soc"] = np.nan
    positions["fuel2_est_time"] = np.nan
    positions["temp1"] = np.nan
    positions["is_charging"] = False
    positions["seg_km"] = np.nan
    positions["day"] = pd.NaT


# =============================
# Metrics keyed by chassis number
# =============================
if not positions.empty:
    per_bike_raw = (
        positions.groupby("deviceid", as_index=False)
        .agg(
            chassis_no=("chassis_no", "first"),
            total_km=("seg_km", "sum"),
            max_kmh=("speed_kmh", "max"),
            first_time=(time_col, "min"),
            last_time=(time_col, "max"),
        )
    )

    avg_speed = (
        positions.groupby("deviceid")["speed_kmh"]
        .apply(mean_ignore_zeros)
        .reset_index(name="avg_kmh")
    )
    per_bike = per_bike_raw.merge(avg_speed, on="deviceid", how="left")

    per_bike_display = per_bike[["chassis_no", "total_km", "max_kmh", "avg_kmh", "first_time", "last_time"]].copy()
    per_bike_display = per_bike_display.sort_values("chassis_no")

    overall_avg_speed = float(per_bike["avg_kmh"].mean()) if len(per_bike) else 0.0
    overall_avg_distance = float(per_bike["total_km"].mean()) if len(per_bike) else 0.0
    overall_max_speed = float(per_bike["max_kmh"].max()) if len(per_bike) else 0.0

    # Popular cells
    grid_decimals = 3
    positions["cell_lat"] = np.round(positions["latitude"], grid_decimals)
    positions["cell_lon"] = np.round(positions["longitude"], grid_decimals)
    top_cells = (
        positions.groupby(["cell_lat", "cell_lon"], as_index=False)
        .agg(points=("id", "count"))
        .sort_values("points", ascending=False)
        .head(50)
    )
else:
    per_bike_display = pd.DataFrame(columns=["chassis_no", "total_km", "max_kmh", "avg_kmh", "first_time", "last_time"])
    overall_avg_speed = 0.0
    overall_avg_distance = 0.0
    overall_max_speed = 0.0
    grid_decimals = 3
    top_cells = pd.DataFrame(columns=["cell_lat", "cell_lon", "points"])


# Charging summaries (door-based)
charge_sessions, charge_daily = detect_charging_door(positions, time_col=time_col)
if not charge_sessions.empty:
    charge_sessions["chassis_no"] = charge_sessions["deviceid"].map(lambda x: str(deviceid_to_chassis.get(int(x), "")).strip())
if not charge_daily.empty:
    charge_daily["chassis_no"] = charge_daily["deviceid"].map(lambda x: str(deviceid_to_chassis.get(int(x), "")).strip())

# Geofence alerts
geofence_events = load_geofence_events(selected_deviceids, start_dt, end_dt)
geofences = load_geofences()
geofence_id_to_name = dict(zip(geofences.get("id", pd.Series(dtype=int)), geofences.get("name", pd.Series(dtype=str))))

if not geofence_events.empty:
    geofence_events["chassis_no"] = geofence_events["deviceid"].map(lambda x: str(deviceid_to_chassis.get(int(x), "")).strip())
    geofence_events["geofence_name"] = geofence_events["geofenceid"].map(
        lambda x: geofence_id_to_name.get(int(x), f"geofenceid={x}") if pd.notna(x) else ""
    )
    geofence_events["event"] = geofence_events["type"].map(lambda t: "ENTER" if t == "geofenceEnter" else ("EXIT" if t == "geofenceExit" else t))

    if "last_geofence_event_id" not in st.session_state:
        st.session_state["last_geofence_event_id"] = None

    last_seen = st.session_state["last_geofence_event_id"]
    new_events = geofence_events.copy()
    if last_seen is not None and not new_events.empty:
        new_events = new_events[new_events["id"] > last_seen]

    try:
        if not geofence_events.empty:
            st.session_state["last_geofence_event_id"] = int(pd.to_numeric(geofence_events["id"], errors="coerce").max())
    except Exception:
        pass

    if not new_events.empty:
        for _, r in new_events.sort_values("servertime").tail(10).iterrows():
            st.toast(f"Geofence {r['event']} | {r['chassis_no']} | {r['geofence_name']}", icon="📍")


# =============================
# Tabs
# =============================
tab_overview, tab_bike, tab_charging, tab_map, tab_geofence = st.tabs(
    ["Overview", "Bike Metrics", "Charging", "Map", "Geofence Alerts"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bikes selected", len(selected_chassis))
    c2.metric("Avg speed (zeros ignored)", f"{overall_avg_speed:.1f} km/h")
    c3.metric("Avg distance (per bike)", f"{overall_avg_distance:.2f} km")
    c4.metric("Max speed (any bike)", f"{overall_max_speed:.1f} km/h")

    st.subheader("Total distance per bike (keyed by chassis number)")
    st.dataframe(per_bike_display.sort_values("total_km", ascending=False), use_container_width=True)
    if not per_bike_display.empty:
        st.bar_chart(per_bike_display.set_index("chassis_no")["total_km"])

with tab_bike:
    st.subheader("Per-bike metrics (keyed by chassis number)")
    st.dataframe(per_bike_display, use_container_width=True)

    st.subheader("Multi-bike comparison (Speed, Temp1, Daily Distance)")
    st.caption("Selected bikes are shown on the same chart, labeled by chassis number.")

    if positions.empty:
        st.info("No analytics available for the selected date range.")
    else:
        df_sel = positions[positions["chassis_no"].isin(selected_chassis)].copy()

        # Speed overlay
        speed_pivot = (
            df_sel.pivot_table(index=time_col, columns="chassis_no", values="speed_kmh", aggfunc="mean")
            .sort_index()
        )
        st.markdown("**Speed over time (km/h)**")
        st.line_chart(speed_pivot)

        # Temp1 overlay
        temp_pivot = (
            df_sel.pivot_table(index=time_col, columns="chassis_no", values="temp1", aggfunc="mean")
            .sort_index()
        )
        st.markdown("**Temp1 over time**")
        st.line_chart(temp_pivot)

        # Daily distance overlay
        daily = (
            df_sel.groupby(["chassis_no", "day"], as_index=False)["seg_km"]
            .sum()
            .rename(columns={"seg_km": "distance_km"})
        )
        dist_pivot = (
            daily.pivot_table(index="day", columns="chassis_no", values="distance_km", aggfunc="sum")
            .sort_index()
        )
        st.markdown("**Daily distance (km)**")
        st.line_chart(dist_pivot)

with tab_charging:
    st.subheader("Charging summary (door=True means charging)")
    st.caption("Charge session is counted from first door=True to first door=False. SOC = Fuel 1. Estimated time = Fuel 2.")

    if positions.empty:
        st.info("No analytics available for the selected date range.")
    else:
        if charge_daily.empty:
            st.info("No charging detected (door=True not found/never true in this range).")
        else:
            daily_view = charge_daily[["chassis_no", "day", "charging_minutes", "charges_in_day", "avg_daily_hours_of_charge"]].copy()
            st.dataframe(daily_view.sort_values(["chassis_no", "day"]), use_container_width=True)

            st.subheader("Daily charging timestamps (sessions)")
            sess_view = charge_sessions[["chassis_no", "start_time", "end_time", "duration_min", "samples"]].copy()
            st.dataframe(sess_view.sort_values(["chassis_no", "start_time"]), use_container_width=True)

        st.subheader("SOC (Fuel 1) graph")
        df_sel = positions[positions["chassis_no"].isin(selected_chassis)].copy()
        if not df_sel["soc"].notna().any():
            st.info("Fuel 1 (SOC) not found in attributes for the selected bikes in this range.")
        else:
            soc_pivot = (
                df_sel.pivot_table(index=time_col, columns="chassis_no", values="soc", aggfunc="mean")
                .sort_index()
            )
            st.line_chart(soc_pivot)

        st.subheader("Estimated time (Fuel 2) graph")
        if not df_sel["fuel2_est_time"].notna().any():
            st.info("Fuel 2 (Estimated time) not found in attributes for the selected bikes in this range.")
        else:
            fuel2_pivot = (
                df_sel.pivot_table(index=time_col, columns="chassis_no", values="fuel2_est_time", aggfunc="mean")
                .sort_index()
            )
            st.line_chart(fuel2_pivot)

with tab_map:
    st.subheader("Map: Full travel paths + all points + popular locations highlighted")
    st.caption(f"Popular locations are based on a rounded grid of {grid_decimals} decimals.")

    if positions.empty:
        st.info("No map data available for the selected date range.")
    else:
        path_rows = []
        for did, g in positions.sort_values(time_col).groupby("deviceid"):
            coords = g[["longitude", "latitude"]].dropna().values.tolist()
            if len(coords) >= 2:
                ch = str(deviceid_to_chassis.get(int(did), "")).strip()
                path_rows.append({"chassis_no": ch, "path": coords})
        paths_df = pd.DataFrame(path_rows)

        points_df = clean_for_pydeck_points(positions[["latitude", "longitude", "chassis_no", "speed_kmh"]].copy())

        hot = top_cells.copy()
        hot = hot.rename(columns={"cell_lat": "latitude", "cell_lon": "longitude"})
        if not hot.empty:
            hot["radius"] = (hot["points"].astype(float).clip(lower=1.0) ** 0.5) * 120.0
            hot["label"] = hot.apply(lambda r: f"Popular cell\nPoints: {int(r['points'])}", axis=1)

        layers = []

        if not paths_df.empty:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=paths_df,
                    get_path="path",
                    get_width=4,
                    pickable=True,
                )
            )

        if not points_df.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=points_df,
                    get_position="[longitude, latitude]",
                    get_radius=10,
                    pickable=True,
                )
            )

        if not hot.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=hot,
                    get_position="[longitude, latitude]",
                    get_radius="radius",
                    get_fill_color="[255, 99, 71, 170]",
                    get_line_color="[255, 255, 255, 220]",
                    line_width_min_pixels=1,
                    pickable=True,
                )
            )

        if points_df.empty and hot.empty and paths_df.empty:
            st.info("No valid map data to display.")
        else:
            if not points_df.empty:
                center_lat = float(points_df["latitude"].mean())
                center_lon = float(points_df["longitude"].mean())
            else:
                center_lat = float(hot["latitude"].mean()) if not hot.empty else 0.0
                center_lon = float(hot["longitude"].mean()) if not hot.empty else 0.0

            view = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=11,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view,
                    tooltip={"text": "{chassis_no}\nSpeed: {speed_kmh} km/h\n{label}"},
                )
            )

        st.subheader("Popular location cells (top 50)")
        st.dataframe(top_cells, use_container_width=True)

with tab_geofence:
    st.subheader("Geofence enter/exit notifications")
    st.caption("Shows events when bikes enter or exit geofences (tc_events type = geofenceEnter/geofenceExit).")

    if geofence_events.empty:
        st.info("No geofence enter/exit events found (or tc_events not available).")
    else:
        view = geofence_events[["servertime", "event", "chassis_no", "geofence_name"]].copy()
        st.dataframe(view.sort_values("servertime", ascending=False), use_container_width=True)

st.caption("✅ Bike selection is ALWAYS sourced from tc_devices.name (chassis number). Date range only affects analytics.")
