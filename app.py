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
st.title("🚲 Bike GPS Analytics (Traccar)")
st.caption("All metrics are keyed by chassis number (usually starts with 'LH'). Data is primarily sourced from tc_positions.")


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

    out["latitude"] = out["latitude"].apply(to_plain_float)
    out["longitude"] = out["longitude"].apply(to_plain_float)
    if "speed_kmh" in out.columns:
        out["speed_kmh"] = out["speed_kmh"].apply(to_plain_float)
    if "chassis_no" in out.columns:
        out["chassis_no"] = out["chassis_no"].apply(to_plain_str)

    out = out.dropna(subset=["latitude", "longitude"]).copy()
    out = out.replace([np.inf, -np.inf], None)
    return out

def clamp_dt(x: datetime, lo: datetime, hi: datetime) -> datetime:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def mean_ignore_zeros(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    s = s[s > 0]
    return float(s.mean()) if len(s) else 0.0

def extract_attr_numeric(series_attrs: pd.Series, key: str) -> pd.Series:
    attrs = series_attrs.apply(safe_json)
    out = attrs.apply(lambda a: a.get(key, None))
    return pd.to_numeric(out, errors="coerce")

def extract_attr_str(series_attrs: pd.Series, key: str) -> pd.Series:
    attrs = series_attrs.apply(safe_json)
    out = attrs.apply(lambda a: a.get(key, None))
    return out.apply(lambda v: "" if v is None else str(v))

def pick_chassis_from_attrs(attrs_dict: dict) -> str:
    # Try common keys (you can extend this list anytime)
    keys = ["chassis", "chassisNo", "chassis_no", "chassisNumber", "vin", "VIN", "vehicleVin", "uniqueid", "uniqueId"]
    for k in keys:
        v = attrs_dict.get(k, None)
        if v is None:
            continue
        v = str(v).strip()
        if v:
            return v
    return ""


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
DB_NAME = cfg["database"]  # should be traccar_new


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
# DB Loads (positions-first)
# =============================
@st.cache_data(ttl=120, show_spinner=True)
def load_positions(device_ids: list[int], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
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
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["speed_kmh"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0).apply(knots_to_kmh)
    return df

@st.cache_data(ttl=120, show_spinner=False)
def load_device_ids_seen(start_dt: datetime, end_dt: datetime) -> list[int]:
    # Pull device list from tc_positions (NOT tc_devices)
    q = text("""
        SELECT DISTINCT deviceid
        FROM tc_positions
        WHERE fixtime >= :start_dt AND fixtime < :end_dt
        ORDER BY deviceid
    """)
    with get_engine().connect() as c:
        rows = c.execute(q, {"start_dt": start_dt, "end_dt": end_dt}).fetchall()
    return [int(r[0]) for r in rows]

@st.cache_data(ttl=120, show_spinner=False)
def load_chassis_index_from_positions(device_ids: list[int], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Build a chassis index primarily from tc_positions.attributes within the selected date range.
    Falls back to tc_devices.uniqueid only if needed.
    """
    # Sample latest few positions per device to find chassis in attributes
    q = (
        text("""
            SELECT p.deviceid, p.fixtime, p.attributes
            FROM tc_positions p
            WHERE p.deviceid IN :device_ids
              AND p.fixtime >= :start_dt
              AND p.fixtime < :end_dt
            ORDER BY p.deviceid, p.fixtime DESC
        """)
        .bindparams(bindparam("device_ids", expanding=True))
    )

    with get_engine().connect() as c:
        df = pd.read_sql(q, c, params={"device_ids": device_ids, "start_dt": start_dt, "end_dt": end_dt})

    if df.empty:
        return pd.DataFrame(columns=["deviceid", "chassis_no"])

    # Parse chassis from attributes
    df["attrs"] = df["attributes"].apply(safe_json)
    df["ch_from_attrs"] = df["attrs"].apply(pick_chassis_from_attrs)
    # Take first non-empty per device
    out = (
        df.sort_values(["deviceid", "fixtime"], ascending=[True, False])
          .groupby("deviceid", as_index=False)
          .agg(chassis_no=("ch_from_attrs", lambda s: next((x for x in s if str(x).strip()), "")))
    )

    # Fallback to tc_devices.uniqueid if present and attrs empty
    if table_exists("tc_devices"):
        try:
            with get_engine().connect() as c:
                dev = pd.read_sql(text("SELECT id AS deviceid, uniqueid FROM tc_devices"), c)
            dev["uniqueid"] = dev["uniqueid"].fillna("").astype(str).str.strip()
            out = out.merge(dev, on="deviceid", how="left")
            out["chassis_no"] = out["chassis_no"].fillna("").astype(str).str.strip()
            out["uniqueid"] = out["uniqueid"].fillna("").astype(str).str.strip()
            out["chassis_no"] = np.where(out["chassis_no"] == "", out["uniqueid"], out["chassis_no"])
            out = out.drop(columns=["uniqueid"])
        except Exception:
            pass

    out["chassis_no"] = out["chassis_no"].fillna("").astype(str).str.strip()
    return out

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
    """
    FIXED: robust against missing tc_events table / missing columns / permission issues.
    This prevents the OperationalError from crashing the app.
    """
    if not table_exists("tc_events"):
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
        # Do NOT crash the app; return empty so UI still works
        return pd.DataFrame(columns=["id", "deviceid", "type", "servertime", "geofenceid", "positionid", "attributes"])


# =============================
# Charging analytics (no sidebar controls)
# =============================
def detect_charging_fixed(df: pd.DataFrame, time_col: str):
    """
    Charging detection uses fixed keys from tc_positions.attributes:
      - charging (bool-ish)
    Produces:
      - per-day: charging_minutes, charges_in_day, avg_daily_hours_of_charge
      - session timestamps per bike
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = df.sort_values(["deviceid", time_col]).copy()
    attrs = d["attributes"].apply(safe_json)

    raw = attrs.apply(lambda a: a.get("charging", None))
    d["is_charging"] = raw.apply(lambda v: bool(v) if v is not None else False)

    d["prev_charge"] = d.groupby("deviceid")["is_charging"].shift(1)
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
    st.header("Date range")
    today = date.today()
    start_date = st.date_input("Start date", value=today - timedelta(days=7))
    end_date = st.date_input("End date (inclusive)", value=today)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    st.divider()
    st.header("Bike selection (Chassis No.)")

    # Device ids from tc_positions in the selected date range
    device_ids_in_range = load_device_ids_seen(start_dt, end_dt)
    if not device_ids_in_range:
        st.warning("No devices found in tc_positions for the selected date range.")
        st.stop()

    chassis_index = load_chassis_index_from_positions(device_ids_in_range, start_dt, end_dt)

    # Filter to chassis starting with LH (as requested)
    chassis_index["chassis_no"] = chassis_index["chassis_no"].fillna("").astype(str).str.strip()
    chassis_index = chassis_index[chassis_index["chassis_no"].str.startswith("LH", na=False)].copy()

    if chassis_index.empty:
        st.warning("No chassis numbers starting with 'LH' were found in tc_positions (attributes) for this date range.")
        st.stop()

    # Ensure uniqueness
    chassis_index = chassis_index.drop_duplicates(subset=["chassis_no"]).copy()

    chassis_list = sorted(chassis_index["chassis_no"].tolist())
    chassis_to_deviceid = dict(zip(chassis_index["chassis_no"], chassis_index["deviceid"]))

    selected_chassis = st.multiselect(
        "Select bikes (chassis number)",
        options=chassis_list,
        default=chassis_list[: min(5, len(chassis_list))],
    )
    selected_deviceids = [int(chassis_to_deviceid[ch]) for ch in selected_chassis]

if not selected_deviceids:
    st.info("Select at least one bike.")
    st.stop()


# =============================
# Load positions
# =============================
positions = load_positions(selected_deviceids, start_dt, end_dt)
time_col = pick_time_field(positions)

if positions.empty or time_col is None:
    st.warning("No position data found for the selected bikes/date range.")
    st.stop()

positions = positions.dropna(subset=["latitude", "longitude", time_col]).copy()
positions = positions.sort_values(["deviceid", time_col]).copy()

# Attach chassis_no from the index (positions-first)
deviceid_to_chassis = dict(zip(chassis_index["deviceid"], chassis_index["chassis_no"]))
positions["chassis_no"] = positions["deviceid"].map(lambda x: str(deviceid_to_chassis.get(int(x), "")).strip())

# Pull SOC + temp1 from tc_positions.attributes
positions["soc"] = extract_attr_numeric(positions["attributes"], "batteryLevel")
positions["temp1"] = extract_attr_numeric(positions["attributes"], "temp1")

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


# =============================
# Metrics keyed by chassis number
# (Removed: Points, deviceid, uniqueid)
# Avg speed ignores zeros
# =============================
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

# Charging summaries (fixed; no sidebar controls shown)
charge_sessions, charge_daily = detect_charging_fixed(positions, time_col=time_col)
if not charge_sessions.empty:
    charge_sessions["chassis_no"] = charge_sessions["deviceid"].map(lambda x: str(deviceid_to_chassis.get(int(x), "")).strip())
if not charge_daily.empty:
    charge_daily["chassis_no"] = charge_daily["deviceid"].map(lambda x: str(deviceid_to_chassis.get(int(x), "")).strip())

# Geofence alerts (robust, won’t crash if tc_events missing)
geofence_events = load_geofence_events(selected_deviceids, start_dt, end_dt)
geofences = load_geofences()
geofence_id_to_name = dict(zip(geofences.get("id", pd.Series(dtype=int)), geofences.get("name", pd.Series(dtype=str))))

if not geofence_events.empty:
    geofence_events["chassis_no"] = geofence_events["deviceid"].map(lambda x: str(deviceid_to_chassis.get(int(x), "")).strip())
    geofence_events["geofence_name"] = geofence_events["geofenceid"].map(
        lambda x: geofence_id_to_name.get(int(x), f"geofenceid={x}") if pd.notna(x) else ""
    )
    geofence_events["event"] = geofence_events["type"].map(lambda t: "ENTER" if t == "geofenceEnter" else ("EXIT" if t == "geofenceExit" else t))

    # Notifications (enter/exit)
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
# Removed: Trips tab (Trip detection removed)
# Removed: Daily active bikes from overview
# =============================
tab_overview, tab_bike, tab_charging, tab_map, tab_a_to_b, tab_geofence = st.tabs(
    ["Overview", "Bike Metrics", "Charging", "Map", "Distance A → B", "Geofence Alerts"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bikes selected", len(selected_chassis))
    c2.metric("Avg speed (all bikes, zeros ignored)", f"{overall_avg_speed:.1f} km/h")
    c3.metric("Avg distance (per bike)", f"{overall_avg_distance:.2f} km")
    c4.metric("Max speed (any bike)", f"{overall_max_speed:.1f} km/h")

    st.subheader("Total distance per bike (keyed by chassis number)")
    st.dataframe(per_bike_display.sort_values("total_km", ascending=False), use_container_width=True)
    st.bar_chart(per_bike_display.set_index("chassis_no")["total_km"])

with tab_bike:
    st.subheader("Per-bike metrics (keyed by chassis number)")
    st.dataframe(per_bike_display, use_container_width=True)

    chosen_chassis = st.selectbox(
        "Choose a bike (chassis number)",
        options=per_bike_display["chassis_no"].tolist(),
    )

    chosen_deviceid = chassis_to_deviceid.get(str(chosen_chassis), None)
    if chosen_deviceid is None:
        st.warning("Selected chassis number could not be mapped to a device.")
    else:
        bike_df = positions[positions["deviceid"] == int(chosen_deviceid)].sort_values(time_col).copy()

        st.markdown("**Speed over time (km/h)**")
        st.line_chart(bike_df.set_index(time_col)["speed_kmh"])

        st.markdown("**SOC over time (%)**")
        if bike_df["soc"].notna().any():
            st.line_chart(bike_df.set_index(time_col)["soc"])
        else:
            st.info("SOC (batteryLevel) not found in attributes for this bike.")

        st.markdown("**Temp1 over time**")
        if bike_df["temp1"].notna().any():
            st.line_chart(bike_df.set_index(time_col)["temp1"])
        else:
            st.info("temp1 not found in attributes for this bike.")

        st.markdown("**Daily distance (km)**")
        bike_daily = bike_df.groupby("day", as_index=False)["seg_km"].sum().rename(columns={"seg_km": "distance_km"})
        st.bar_chart(bike_daily.set_index("day")["distance_km"])

        st.markdown("**Max speed**")
        st.write(f"{bike_df['speed_kmh'].max():.1f} km/h")

with tab_charging:
    st.subheader("Daily charging summary")
    st.caption("Shown: Daily charging timestamps, number of charges in a day, and average daily hours of charge.")

    if charge_daily.empty:
        st.info("No charging detected (attribute 'charging' not found/false in the selected range).")
    else:
        daily_view = charge_daily[["chassis_no", "day", "charging_minutes", "charges_in_day", "avg_daily_hours_of_charge"]].copy()
        st.dataframe(daily_view.sort_values(["chassis_no", "day"]), use_container_width=True)

        st.subheader("Daily charging timestamps (sessions)")
        sess_view = charge_sessions[["chassis_no", "start_time", "end_time", "duration_min", "samples"]].copy()
        st.dataframe(sess_view.sort_values(["chassis_no", "start_time"]), use_container_width=True)

    st.subheader("SOC graph")
    chosen_chassis_charge = st.selectbox(
        "Choose a bike for SOC (chassis number)",
        options=sorted(selected_chassis),
        key="soc_bike_select",
    )
    chosen_deviceid_charge = chassis_to_deviceid.get(str(chosen_chassis_charge), None)
    if chosen_deviceid_charge is None:
        st.warning("Selected chassis number could not be mapped to a device.")
    else:
        bike_df = positions[positions["deviceid"] == int(chosen_deviceid_charge)].sort_values(time_col).copy()
        if bike_df["soc"].notna().any():
            st.line_chart(bike_df.set_index(time_col)["soc"])
        else:
            st.info("SOC (batteryLevel) not found in attributes for this bike in the selected range.")

with tab_map:
    st.subheader("Map: Full travel paths + all points + popular locations highlighted")
    st.caption(f"Popular locations are based on a rounded grid of {grid_decimals} decimals.")

    # Paths per selected bike (plot traveling)
    path_rows = []
    for did, g in positions.sort_values(time_col).groupby("deviceid"):
        coords = g[["longitude", "latitude"]].dropna().values.tolist()
        if len(coords) >= 2:
            ch = str(deviceid_to_chassis.get(int(did), "")).strip()
            path_rows.append({"chassis_no": ch, "path": coords})

    paths_df = pd.DataFrame(path_rows)

    # Every point (all datapoints)
    points_df = clean_for_pydeck_points(positions[["latitude", "longitude", "chassis_no", "speed_kmh"]].copy())

    # Hotspots (pinpoint/highlight popular)
    hot = top_cells.copy()
    hot = hot.rename(columns={"cell_lat": "latitude", "cell_lon": "longitude"})
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
                pickable=True,
            )
        )

    if points_df.empty and hot.empty and paths_df.empty:
        st.info("No valid map data to display.")
    else:
        # View center
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

with tab_a_to_b:
    st.subheader("Distance from point A → point B (route distance)")
    st.caption("Computed as sum of segment distances between consecutive points in the selected time range.")

    bike_chassis = st.selectbox(
        "Bike (chassis number)",
        options=sorted(selected_chassis),
        key="a2b_chassis",
    )
    bike_id = chassis_to_deviceid.get(str(bike_chassis), None)
    if bike_id is None:
        st.warning("Selected chassis number could not be mapped to a device.")
    else:
        bike_df = positions[positions["deviceid"] == int(bike_id)].sort_values(time_col).copy()

        min_dt = bike_df[time_col].min().to_pydatetime()
        max_dt = bike_df[time_col].max().to_pydatetime()

        colA, colB = st.columns(2)

        with colA:
            a_date = st.date_input("Point A date", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date(), key="a_date")
            a_time = st.time_input("Point A time", value=min_dt.time().replace(microsecond=0), key="a_time")
            tA = datetime.combine(a_date, a_time)

        with colB:
            b_date = st.date_input("Point B date", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date(), key="b_date")
            b_time = st.time_input("Point B time", value=max_dt.time().replace(microsecond=0), key="b_time")
            tB = datetime.combine(b_date, b_time)

        tA = clamp_dt(tA, min_dt, max_dt)
        tB = clamp_dt(tB, min_dt, max_dt)

        if tA > tB:
            st.error("Point A datetime must be <= Point B datetime.")
        else:
            segment = bike_df[(bike_df[time_col] >= pd.to_datetime(tA)) & (bike_df[time_col] <= pd.to_datetime(tB))].copy()
            if len(segment) < 2:
                st.warning("Not enough points between A and B to compute route distance.")
            else:
                dist = float(segment["seg_km"].sum())
                max_speed = float(segment["speed_kmh"].max())
                avg_speed = mean_ignore_zeros(segment["speed_kmh"])

                st.success(f"Route distance A → B: **{dist:.2f} km**")
                st.write(f"Max speed in range: **{max_speed:.1f} km/h**")
                st.write(f"Avg speed in range (zeros ignored): **{avg_speed:.1f} km/h**")

with tab_geofence:
    st.subheader("Geofence enter/exit notifications")
    st.caption("Shows events when bikes enter or exit geofences (tc_events type = geofenceEnter/geofenceExit).")

    if geofence_events.empty:
        st.info("No geofence enter/exit events found (or tc_events not available).")
    else:
        view = geofence_events[["servertime", "event", "chassis_no", "geofence_name"]].copy()
        st.dataframe(view.sort_values("servertime", ascending=False), use_container_width=True)

st.caption("✅ Bikes are identified and displayed using chassis number (prefers tc_positions.attributes; filtered to chassis starting with 'LH').")
