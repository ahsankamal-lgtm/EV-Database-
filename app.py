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
st.caption("All metrics are keyed by device ID (tc_positions.deviceid). Data is primarily sourced from tc_positions.")


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

def to_plain_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None

def clean_for_pydeck_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure pydeck only receives JSON-serializable values.
    Expects: latitude, longitude, deviceid, speed_kmh
    """
    if df.empty:
        return df
    out = df.copy()

    keep = [c for c in ["latitude", "longitude", "deviceid", "speed_kmh"] if c in out.columns]
    out = out[keep].copy()

    out["latitude"] = out["latitude"].apply(to_plain_float)
    out["longitude"] = out["longitude"].apply(to_plain_float)
    if "speed_kmh" in out.columns:
        out["speed_kmh"] = out["speed_kmh"].apply(to_plain_float)
    if "deviceid" in out.columns:
        out["deviceid"] = out["deviceid"].apply(to_plain_int)

    out = out.dropna(subset=["latitude", "longitude"]).copy()
    out = out.replace([np.inf, -np.inf], None)
    return out

def mean_ignore_zeros(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    s = s[s > 0]
    return float(s.mean()) if len(s) else 0.0

def extract_attr_numeric(series_attrs: pd.Series, key: str) -> pd.Series:
    attrs = series_attrs.apply(safe_json)
    out = attrs.apply(lambda a: a.get(key, None))
    return pd.to_numeric(out, errors="coerce")

def extract_attr_numeric_first(series_attrs: pd.Series, keys: list[str]) -> pd.Series:
    """
    Try multiple keys (in order) and return the first numeric series that has any non-null values.
    """
    best = pd.Series([np.nan] * len(series_attrs), index=series_attrs.index, dtype="float64")
    attrs = series_attrs.apply(safe_json)

    for k in keys:
        s = attrs.apply(lambda a: a.get(k, None))
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().any():
            best = s
            break
    return best


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
# DB Loads (FAST sidebar)
# =============================
@st.cache_data(ttl=600, show_spinner=False)
def load_all_device_ids_from_tc_devices() -> list[int]:
    if not table_exists("tc_devices"):
        return []
    try:
        with get_engine().connect() as c:
            rows = c.execute(text("SELECT id FROM tc_devices ORDER BY id")).fetchall()
        return [int(r[0]) for r in rows]
    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def load_positions_servertime_bounds() -> tuple[datetime | None, datetime | None]:
    """
    Used only to pick a sensible DEFAULT day (latest available servertime),
    not to force a range UI.
    """
    if not table_exists("tc_positions"):
        return None, None
    try:
        with get_engine().connect() as c:
            r = c.execute(text("SELECT MIN(servertime) AS mn, MAX(servertime) AS mx FROM tc_positions")).mappings().first()
        mn = pd.to_datetime(r["mn"], errors="coerce")
        mx = pd.to_datetime(r["mx"], errors="coerce")
        mn = None if pd.isna(mn) else mn.to_pydatetime()
        mx = None if pd.isna(mx) else mx.to_pydatetime()
        return mn, mx
    except Exception:
        return None, None

@st.cache_data(ttl=120, show_spinner=True)
def load_positions(device_ids: list[int], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Filter by servertime only (fixtime/devicetime can be invalid in your DB).
    """
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
              AND servertime >= :start_dt
              AND servertime < :end_dt
            ORDER BY deviceid, servertime
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
        return pd.DataFrame(columns=["id", "deviceid", "type", "servertime", "geofenceid", "positionid", "attributes"])


# =============================
# Charging analytics (door-based, drop incomplete sessions)
# =============================
def detect_charging_from_door_drop_incomplete(df: pd.DataFrame, time_col: str):
    """
    Charging definition:
      - attributes['door'] == True  => charging
      - attributes['door'] == False => not charging

    Session definition:
      - start: False -> True
      - end:   True  -> False  (end timestamp is the timestamp of the first False row after charging)

    Drop incomplete sessions:
      - If a start occurs but no end occurs within the selected range, exclude that session entirely.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = df.sort_values(["deviceid", time_col]).copy()
    attrs = d["attributes"].apply(safe_json)

    door_raw = attrs.apply(lambda a: a.get("door", None))
    d["is_charging"] = door_raw.apply(lambda v: bool(v) if v is not None else False)

    sessions_rows = []

    for did, g in d.groupby("deviceid", sort=False):
        g = g.sort_values(time_col).copy()
        isch = g["is_charging"].astype(bool).values
        times = g[time_col].values

        prev = np.concatenate([[False], isch[:-1]])
        starts = (~prev) & isch
        ends = prev & (~isch)

        start_idx = np.where(starts)[0].tolist()
        end_idx = np.where(ends)[0].tolist()

        ei = 0
        for si in start_idx:
            while ei < len(end_idx) and end_idx[ei] <= si:
                ei += 1
            if ei >= len(end_idx):
                break  # incomplete -> drop
            e = end_idx[ei]
            ei += 1

            start_time = pd.to_datetime(times[si])
            end_time = pd.to_datetime(times[e])
            samples = int(max(0, e - si))

            sessions_rows.append(
                {
                    "deviceid": int(did),
                    "start_time": start_time,
                    "end_time": end_time,
                    "samples": samples,
                }
            )

    if not sessions_rows:
        return pd.DataFrame(), pd.DataFrame()

    sessions = pd.DataFrame(sessions_rows)
    sessions["duration_min"] = (sessions["end_time"] - sessions["start_time"]).dt.total_seconds() / 60.0
    sessions = sessions[sessions["duration_min"] >= 0].copy()
    sessions["day"] = sessions["start_time"].dt.date

    daily = (
        sessions.groupby(["deviceid", "day"], as_index=False)
        .agg(
            charging_minutes=("duration_min", "sum"),
            charges_in_day=("start_time", "count"),
        )
    )
    daily["avg_daily_hours_of_charge"] = daily["charging_minutes"] / 60.0

    return sessions.sort_values(["deviceid", "start_time"]), daily.sort_values(["deviceid", "day"])


# =============================
# Sidebar (ONE DAY ONLY)
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
    st.header("Date (single day)")

    mn_dt, mx_dt = load_positions_servertime_bounds()
    if mx_dt is not None:
        default_day = mx_dt.date()
    else:
        default_day = date.today()

    selected_day = st.date_input("Select day", value=default_day)

    start_dt = datetime.combine(selected_day, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)

    st.divider()
    st.header("Bike selection (Device IDs)")

    device_list = load_all_device_ids_from_tc_devices()
    if not device_list:
        st.warning("No devices found in tc_devices.")
        st.stop()

    selected_deviceids = st.multiselect(
        "Select bikes (device IDs)",
        options=device_list,
        default=device_list[: min(5, len(device_list))],
    )

if not selected_deviceids:
    st.info("Select at least one bike.")
    st.stop()


# =============================
# Load positions
# =============================
positions = load_positions([int(x) for x in selected_deviceids], start_dt, end_dt)
time_col = pick_time_field(positions)

if positions.empty or time_col is None:
    st.warning("No position data found for the selected bikes on the selected day.")
    st.stop()

positions = positions.dropna(subset=["latitude", "longitude", time_col]).copy()
positions = positions.sort_values(["deviceid", time_col]).copy()

# Pull fuel1 (SOC graph) + temp1 from tc_positions.attributes
positions["fuel1"] = extract_attr_numeric_first(
    positions["attributes"],
    keys=["fuel1", "fuel 1", "fuel_1", "Fuel1", "Fuel 1", "Fuel_1"],
)
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
# Metrics keyed by deviceid
# Avg speed ignores zeros
# =============================
per_bike_raw = (
    positions.groupby("deviceid", as_index=False)
    .agg(
        deviceid=("deviceid", "first"),
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

per_bike_display = per_bike[["deviceid", "total_km", "max_kmh", "avg_kmh", "first_time", "last_time"]].copy()
per_bike_display = per_bike_display.sort_values("deviceid")

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

# Charging summaries (door-based; incomplete dropped)
charge_sessions, charge_daily = detect_charging_from_door_drop_incomplete(positions, time_col=time_col)

# Geofence alerts (robust)
geofence_events = load_geofence_events([int(x) for x in selected_deviceids], start_dt, end_dt)
geofences = load_geofences()
geofence_id_to_name = dict(zip(geofences.get("id", pd.Series(dtype=int)), geofences.get("name", pd.Series(dtype=str))))

if not geofence_events.empty:
    geofence_events["geofence_name"] = geofence_events["geofenceid"].map(
        lambda x: geofence_id_to_name.get(int(x), f"geofenceid={x}") if pd.notna(x) else ""
    )
    geofence_events["event"] = geofence_events["type"].map(
        lambda t: "ENTER" if t == "geofenceEnter" else ("EXIT" if t == "geofenceExit" else t)
    )

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
            st.toast(f"Geofence {r['event']} | deviceid={int(r['deviceid'])} | {r['geofence_name']}", icon="📍")


# =============================
# Tabs
# =============================
tab_overview, tab_bike, tab_charging, tab_map, tab_geofence = st.tabs(
    ["Overview", "Bike Metrics", "Charging", "Map", "Geofence Alerts"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bikes selected", len(selected_deviceids))
    c2.metric("Avg speed (all bikes, zeros ignored)", f"{overall_avg_speed:.1f} km/h")
    c3.metric("Avg distance (per bike)", f"{overall_avg_distance:.2f} km")
    c4.metric("Max speed (any bike)", f"{overall_max_speed:.1f} km/h")

    st.subheader("Total distance per bike (keyed by deviceid)")
    st.dataframe(per_bike_display.sort_values("total_km", ascending=False), use_container_width=True)
    st.bar_chart(per_bike_display.set_index("deviceid")["total_km"])

with tab_bike:
    st.subheader("Per-bike metrics (keyed by deviceid)")
    st.dataframe(per_bike_display, use_container_width=True)

    st.subheader("Multi-bike comparison (Speed, Temp1, Daily Distance)")
    st.caption("Shows selected bikes on the same chart, clearly labeled by device ID.")

    deviceids_selected_now = [int(x) for x in selected_deviceids]
    if not deviceids_selected_now:
        st.info("Select bikes from the sidebar to view comparison charts.")
    else:
        df_sel = positions[positions["deviceid"].isin(deviceids_selected_now)].copy()

        speed_wide = df_sel[["deviceid", time_col, "speed_kmh"]].dropna(subset=[time_col]).copy()
        speed_wide["deviceid"] = speed_wide["deviceid"].astype(int).astype(str)
        speed_pivot = speed_wide.pivot_table(index=time_col, columns="deviceid", values="speed_kmh", aggfunc="mean").sort_index()
        st.markdown("**Speed over time (km/h)**")
        st.line_chart(speed_pivot)

        temp_wide = df_sel[["deviceid", time_col, "temp1"]].dropna(subset=[time_col]).copy()
        temp_wide["deviceid"] = temp_wide["deviceid"].astype(int).astype(str)
        temp_pivot = temp_wide.pivot_table(index=time_col, columns="deviceid", values="temp1", aggfunc="mean").sort_index()
        st.markdown("**Temp1 over time**")
        st.line_chart(temp_pivot)

        daily = (
            df_sel.groupby(["deviceid", "day"], as_index=False)["seg_km"]
            .sum()
            .rename(columns={"seg_km": "distance_km"})
        )
        daily["deviceid"] = daily["deviceid"].astype(int).astype(str)
        dist_pivot = daily.pivot_table(index="day", columns="deviceid", values="distance_km", aggfunc="sum").sort_index()
        st.markdown("**Daily distance (km)**")
        st.line_chart(dist_pivot)

with tab_charging:
    st.subheader("Daily charging summary")
    st.caption("Charging is determined from attributes['door'] (True = charging, False = not charging). Incomplete sessions are dropped.")

    if charge_daily.empty:
        st.info("No charging detected (attribute 'door' not found/false in the selected day, or sessions were incomplete and dropped).")
    else:
        daily_view = charge_daily[["deviceid", "day", "charging_minutes", "charges_in_day", "avg_daily_hours_of_charge"]].copy()
        st.dataframe(daily_view.sort_values(["deviceid", "day"]), use_container_width=True)

        st.subheader("Daily charging timestamps (sessions)")
        sess_view = charge_sessions[["deviceid", "start_time", "end_time", "duration_min", "samples"]].copy()
        st.dataframe(sess_view.sort_values(["deviceid", "start_time"]), use_container_width=True)

    st.subheader("SOC graph (multi-bike)")
    df_sel = positions[positions["deviceid"].isin([int(x) for x in selected_deviceids])].copy()
    if df_sel.empty or not df_sel["fuel1"].notna().any():
        st.info("Fuel 1 not found in attributes for the selected bikes on the selected day.")
    else:
        soc_pivot = df_sel.pivot_table(index=time_col, columns="deviceid", values="fuel1", aggfunc="mean").sort_index()
        soc_pivot.columns = soc_pivot.columns.astype(int).astype(str)
        st.line_chart(soc_pivot)

with tab_map:
    st.subheader("Map: Full travel paths + all points + popular locations highlighted")
    st.caption(f"Popular locations are based on a rounded grid of {grid_decimals} decimals.")

    path_rows = []
    for did, g in positions.sort_values(time_col).groupby("deviceid"):
        coords = g[["longitude", "latitude"]].dropna().values.tolist()
        if len(coords) >= 2:
            path_rows.append({"deviceid": int(did), "path": coords})

    paths_df = pd.DataFrame(path_rows)

    points_df = clean_for_pydeck_points(positions[["latitude", "longitude", "deviceid", "speed_kmh"]].copy())

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
                tooltip={"text": "deviceid={deviceid}\nSpeed: {speed_kmh} km/h\n{label}"},
            )
        )

    st.subheader("Popular location cells (top 50)")
    st.dataframe(top_cells, use_container_width=True)

with tab_geofence:
    st.subheader("Geofence enter/exit notifications")
    st.caption("Shows events when bikes enter or exit geofences (tc_events type = geofenceEnter/geofenceExit).")

    if geofence_events.empty:
        st.info("No geofence enter/exit events found (or tc_events not available) for the selected day.")
    else:
        view = geofence_events[["servertime", "event", "deviceid", "geofence_name"]].copy()
        st.dataframe(view.sort_values("servertime", ascending=False), use_container_width=True)

st.caption("✅ Single-day mode enabled: you can only select ONE day. Data is filtered using tc_positions.servertime.")
