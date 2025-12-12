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


# -----------------------------
# Utilities
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
    return knots * 1.852


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine (km)."""
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


def build_grid(lat, lon, decimals=3):
    return np.round(lat, decimals), np.round(lon, decimals)


def pick_time_field(df):
    for c in ["fixtime", "devicetime", "servertime"]:
        if c in df.columns:
            return c
    return None


# -----------------------------
# DB / Engine
# -----------------------------
def _get_cfg():
    if "traccar" not in st.secrets:
        st.error("Missing [traccar] section in .streamlit/secrets.toml")
        st.stop()
    cfg = dict(st.secrets["traccar"])
    cfg.setdefault("port", 3306)
    cfg.setdefault("database", "traccar")
    return cfg


@st.cache_resource(show_spinner=False)
def get_engine(db_name: str):
    """
    Cache an engine per DB name.
    """
    cfg = _get_cfg()
    host = cfg["host"]
    port = int(cfg["port"])
    user = cfg["user"]
    password = cfg["password"]

    # Important: connect_timeout + pool_pre_ping helps with flaky networks
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset=utf8mb4"
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": 10},
    )
    return engine


def test_connection(db_name: str):
    """
    Returns (ok: bool, message: str).
    """
    try:
        eng = get_engine(db_name)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        return True, f"Connected successfully to database '{db_name}'."
    except OperationalError as e:
        # Don’t print secrets. Give actionable explanation.
        return False, (
            "Could not connect to MySQL. Most common reasons:\n"
            "- Wrong database name\n"
            "- MySQL user not allowed from remote host (user created only for 'localhost')\n"
            "- Port 3306 blocked by firewall / AWS security group\n"
            "- MySQL bind-address not public (127.0.0.1)\n\n"
            "Fix is server-side, but this app lets you validate DB name & connection."
        )


def list_tables(db_name: str):
    eng = get_engine(db_name)
    q = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :db
        ORDER BY table_name
    """)
    with eng.connect() as c:
        return pd.read_sql(q, c, params={"db": db_name})


# -----------------------------
# Data queries (safe + correct IN expansion)
# -----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_devices(db_name: str):
    eng = get_engine(db_name)
    q = text("""
        SELECT id, name, uniqueid
        FROM tc_devices
        ORDER BY name
    """)
    with eng.connect() as c:
        return pd.read_sql(q, c)


@st.cache_data(ttl=60, show_spinner=True)
def load_positions(db_name: str, device_ids: list[int], start_dt: datetime, end_dt: datetime):
    eng = get_engine(db_name)

    # Use SQLAlchemy expanding parameter for IN (...)
    q = (
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

    with eng.connect() as c:
        df = pd.read_sql(q, c, params={"device_ids": device_ids, "start_dt": start_dt, "end_dt": end_dt})

    # normalize datetime columns
    for col in ["servertime", "devicetime", "fixtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["speed_kmh"] = knots_to_kmh(pd.to_numeric(df["speed"], errors="coerce").fillna(0.0))
    return df


def detect_trips(df: pd.DataFrame, time_col: str, gap_minutes: int, ignition_key: str, motion_key: str):
    if df.empty:
        return df.assign(trip_id=pd.Series(dtype=int)), pd.DataFrame()

    d = df.sort_values(["deviceid", time_col]).copy()
    d["prev_time"] = d.groupby("deviceid")[time_col].shift(1)
    d["gap_min"] = (d[time_col] - d["prev_time"]).dt.total_seconds() / 60.0

    attrs = d["attributes"].apply(safe_json)
    d["_ign"] = attrs.apply(lambda a: a.get(ignition_key, None) if ignition_key else None)
    d["_mot"] = attrs.apply(lambda a: a.get(motion_key, None) if motion_key else None)
    d["_prev_ign"] = d.groupby("deviceid")["_ign"].shift(1)
    d["_prev_mot"] = d.groupby("deviceid")["_mot"].shift(1)

    new_trip = (
        d["prev_time"].isna()
        | (d["gap_min"] > gap_minutes)
        | ((d["_prev_ign"] == False) & (d["_ign"] == True))
        | ((d["_prev_mot"] == False) & (d["_mot"] == True))
    )
    d["trip_id"] = new_trip.groupby(d["deviceid"]).cumsum().astype(int)

    d["prev_lat"] = d.groupby(["deviceid", "trip_id"])["latitude"].shift(1)
    d["prev_lon"] = d.groupby(["deviceid", "trip_id"])["longitude"].shift(1)
    seg_km = haversine_km(d["prev_lat"].fillna(d["latitude"]), d["prev_lon"].fillna(d["longitude"]), d["latitude"], d["longitude"])
    d["seg_km_trip"] = np.where(d["prev_lat"].isna(), 0.0, seg_km)

    trips = (
        d.groupby(["deviceid", "trip_id"], as_index=False)
        .agg(
            start_time=(time_col, "min"),
            end_time=(time_col, "max"),
            points=("id", "count"),
            trip_km=("seg_km_trip", "sum"),
            max_kmh=("speed_kmh", "max"),
            avg_kmh=("speed_kmh", "mean"),
        )
    )
    trips["duration_min"] = (trips["end_time"] - trips["start_time"]).dt.total_seconds() / 60.0
    return d, trips.sort_values(["deviceid", "start_time"])


def detect_charging(df: pd.DataFrame, time_col: str,
                    mode: str, charging_key: str, power_key: str, power_threshold: float,
                    battery_level_key: str, capacity_kwh: float):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = df.sort_values(["deviceid", time_col]).copy()
    attrs = d["attributes"].apply(safe_json)

    if mode == "attribute_key":
        raw = attrs.apply(lambda a: a.get(charging_key, None))
        d["is_charging"] = raw.apply(lambda v: bool(v) if v is not None else False)
    else:
        p = attrs.apply(lambda a: a.get(power_key, None))
        p = pd.to_numeric(p, errors="coerce")
        d["power_val"] = p
        d["is_charging"] = d["power_val"].fillna(-1) >= power_threshold

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

    # Consumption estimate if batteryLevel exists
    batt = attrs.apply(lambda a: a.get(battery_level_key, None))
    batt = pd.to_numeric(batt, errors="coerce")
    d["battery_level"] = batt

    if d["battery_level"].notna().any():
        # For each session: first and last battery level
        def first_nonnull(g):
            s = g.sort_values(time_col)["battery_level"].dropna()
            return s.iloc[0] if len(s) else np.nan

        def last_nonnull(g):
            s = g.sort_values(time_col)["battery_level"].dropna()
            return s.iloc[-1] if len(s) else np.nan

        b1 = d.groupby(["deviceid", "charge_session_id"]).apply(first_nonnull).reset_index(name="start_battery")
        b2 = d.groupby(["deviceid", "charge_session_id"]).apply(last_nonnull).reset_index(name="end_battery")
        sessions = sessions.merge(b1, on=["deviceid", "charge_session_id"], how="left").merge(b2, on=["deviceid", "charge_session_id"], how="left")
        sessions["battery_gain_pct"] = sessions["end_battery"] - sessions["start_battery"]
        sessions["consumption_kwh_est"] = (sessions["battery_gain_pct"] / 100.0) * capacity_kwh
    else:
        sessions["consumption_kwh_est"] = np.nan

    daily = (
        sessions.groupby(["deviceid", "day"], as_index=False)
        .agg(
            charging_minutes=("duration_min", "sum"),
            sessions=("charge_session_id", "nunique"),
            consumption_kwh_est=("consumption_kwh_est", "sum"),
        )
    )

    return sessions.sort_values(["deviceid", "start_time"]), daily.sort_values(["deviceid", "day"])


# -----------------------------
# Sidebar: connection + filters
# -----------------------------
st.title("🚲 Bike GPS Analytics (Traccar)")
st.caption("Uses tc_positions. Shows distance, speed, trips, charging, daily active bikes, popular locations.")

cfg = _get_cfg()
default_db = cfg.get("database", "traccar")

with st.sidebar:
    st.header("Database")
    db_name = st.text_input("Database name", value=default_db)

    ok, msg = test_connection(db_name)
    if ok:
        st.success(msg)
        with st.expander("Show tables (quick check)"):
            try:
                tables = list_tables(db_name)
                st.dataframe(tables, use_container_width=True, height=300)
                if "tc_positions" not in set(tables["table_name"]):
                    st.warning("Table tc_positions not found in this database name.")
            except Exception:
                st.info("Could not list tables (permission-limited user). Connection still OK.")
    else:
        st.error(msg)
        st.stop()

    st.divider()
    st.header("Filters")

    # Now safe to load devices (won't crash without explanation)
    try:
        devices_df = load_devices(db_name)
    except OperationalError:
        st.error("Connected to DB, but failed to query tc_devices. Check permissions for this user.")
        st.stop()
    except Exception as e:
        st.error("Failed to load devices. Verify this is a Traccar database and tables exist.")
        st.stop()

    if devices_df.empty:
        st.warning("No devices found in tc_devices.")
        st.stop()

    device_name_map = dict(zip(devices_df["name"], devices_df["id"]))
    device_unique_map = dict(zip(devices_df["id"], devices_df["uniqueid"]))

    selected_names = st.multiselect(
        "Select bikes/devices",
        options=list(device_name_map.keys()),
        default=list(device_name_map.keys())[: min(5, len(device_name_map))],
    )
    selected_device_ids = [int(device_name_map[n]) for n in selected_names]

    today = date.today()
    start_date = st.date_input("Start date", value=today - timedelta(days=7))
    end_date = st.date_input("End date (inclusive)", value=today)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    st.divider()
    st.subheader("Trip detection")
    gap_minutes = st.slider("New trip if time gap exceeds (minutes)", 1, 120, 10)
    ignition_key = st.text_input("Ignition key in attributes", value="ignition")
    motion_key = st.text_input("Motion key in attributes", value="motion")

    st.divider()
    st.subheader("Charging detection")
    charging_mode = st.selectbox("Charging mode", ["power_threshold", "attribute_key"])
    charging_key = st.text_input("Charging boolean key (attribute_key mode)", value="charging")
    power_key = st.text_input("Power key (power_threshold mode)", value="power")
    power_threshold = st.number_input("Charging if power >= (volts)", value=13.0, step=0.1)

    st.divider()
    st.subheader("Consumption estimate")
    battery_level_key = st.text_input("Battery level key (0-100)", value="batteryLevel")
    capacity_kwh = st.number_input("Battery capacity (kWh)", value=2.0, step=0.1)

if not selected_device_ids:
    st.info("Select at least one device.")
    st.stop()


# -----------------------------
# Load positions
# -----------------------------
try:
    positions = load_positions(db_name, selected_device_ids, start_dt, end_dt)
except OperationalError:
    st.error("Connected, but failed to query tc_positions. Check permissions or table existence.")
    st.stop()
except Exception:
    st.error("Failed to load positions. Verify tc_positions exists and fixtime column is present.")
    st.stop()

time_col = pick_time_field(positions)
if positions.empty or time_col is None:
    st.warning("No position data found for selected devices/date range.")
    st.stop()

positions = positions.dropna(subset=["latitude", "longitude", time_col]).copy()
positions = positions.sort_values(["deviceid", time_col]).copy()

# Segment distances per device
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

# Per-bike summary
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
id_to_name = dict(zip(devices_df["id"], devices_df["name"]))
per_bike["device_name"] = per_bike["deviceid"].map(lambda i: id_to_name.get(int(i), str(i)))
per_bike["uniqueid"] = per_bike["deviceid"].map(lambda i: device_unique_map.get(int(i), ""))

overall_avg_speed = float(per_bike["avg_kmh"].mean()) if len(per_bike) else 0.0
overall_avg_distance = float(per_bike["total_km"].mean()) if len(per_bike) else 0.0
overall_max_speed = float(per_bike["max_kmh"].max()) if len(per_bike) else 0.0

# Daily active bikes (distance threshold)
ACTIVE_KM_THRESHOLD = 0.2
daily_km = positions.groupby(["day", "deviceid"], as_index=False)["seg_km"].sum()
daily_km["active"] = daily_km["seg_km"] >= ACTIVE_KM_THRESHOLD
daily_active = daily_km.groupby("day", as_index=False)["active"].sum().rename(columns={"active": "active_bikes"})

# Trips
pos_with_trips, trips = detect_trips(
    positions,
    time_col=time_col,
    gap_minutes=gap_minutes,
    ignition_key=ignition_key.strip(),
    motion_key=motion_key.strip(),
)

trips_per_bike = trips.groupby("deviceid", as_index=False).agg(total_trips=("trip_id", "nunique"), trips_km=("trip_km", "sum"))
trips_per_bike["device_name"] = trips_per_bike["deviceid"].map(lambda i: id_to_name.get(int(i), str(i)))

# Charging
charge_sessions, charge_daily = detect_charging(
    positions,
    time_col=time_col,
    mode=charging_mode,
    charging_key=charging_key.strip(),
    power_key=power_key.strip(),
    power_threshold=float(power_threshold),
    battery_level_key=battery_level_key.strip(),
    capacity_kwh=float(capacity_kwh),
)
if not charge_daily.empty:
    charge_daily["device_name"] = charge_daily["deviceid"].map(lambda i: id_to_name.get(int(i), str(i)))
if not charge_sessions.empty:
    charge_sessions["device_name"] = charge_sessions["deviceid"].map(lambda i: id_to_name.get(int(i), str(i)))

# Popular locations
grid_decimals = 3
cell_lat, cell_lon = build_grid(positions["latitude"].to_numpy(), positions["longitude"].to_numpy(), decimals=grid_decimals)
positions["cell_lat"] = cell_lat
positions["cell_lon"] = cell_lon
top_cells = (
    positions.groupby(["cell_lat", "cell_lon"], as_index=False)
    .agg(points=("id", "count"))
    .sort_values("points", ascending=False)
    .head(50)
)


# -----------------------------
# UI Tabs
# -----------------------------
tab_overview, tab_bike, tab_trips, tab_charging, tab_locations, tab_a_to_b = st.tabs(
    ["Overview", "Bike Metrics", "Trips", "Charging", "Popular Locations", "Distance A → B"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bikes selected", len(selected_device_ids))
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

    bike_df = positions[positions["deviceid"] == chosen_id].copy()
    bike_df = bike_df.sort_values(time_col)

    st.markdown("**Speed over time (km/h)**")
    st.line_chart(bike_df.set_index(time_col)["speed_kmh"])

    st.markdown("**Daily distance (km)**")
    bike_daily = bike_df.groupby("day", as_index=False)["seg_km"].sum().rename(columns={"seg_km": "distance_km"})
    st.bar_chart(bike_daily.set_index("day")["distance_km"])

    st.markdown("**Max speed**")
    st.write(f"{bike_df['speed_kmh'].max():.1f} km/h")

with tab_trips:
    st.subheader("Trips")
    if trips.empty:
        st.info("No trips detected. Try lowering the time-gap threshold in the sidebar.")
    else:
        view = trips.copy()
        view["device_name"] = view["deviceid"].map(lambda i: id_to_name.get(int(i), str(i)))
        st.dataframe(view.sort_values(["device_name", "start_time"]), use_container_width=True)

        st.subheader("Total trips per bike")
        st.dataframe(trips_per_bike.sort_values("total_trips", ascending=False), use_container_width=True)

with tab_charging:
    st.subheader("Daily charging time")
    if charge_daily.empty:
        st.info("No charging detected. Adjust charging mode/key/threshold in the sidebar.")
    else:
        st.dataframe(charge_daily.sort_values(["device_name", "day"]), use_container_width=True)

        st.subheader("Charging sessions (timestamps)")
        st.dataframe(charge_sessions.sort_values(["device_name", "start_time"]), use_container_width=True)

        st.subheader("Charging consumption (estimate)")
        st.caption("Only valid if battery level exists in attributes (e.g., batteryLevel 0-100).")
        cons = charge_daily.groupby("device_name", as_index=False).agg(
            charging_minutes=("charging_minutes", "sum"),
            consumption_kwh_est=("consumption_kwh_est", "sum"),
        )
        st.dataframe(cons.sort_values("consumption_kwh_est", ascending=False), use_container_width=True)

with tab_locations:
    st.subheader("Popular location cells")
    st.caption(f"Rounded grid: {grid_decimals} decimals (~110m).")
    st.dataframe(top_cells, use_container_width=True)

    st.subheader("Latest position map")
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
    st.caption("Computed as sum of segment distances between consecutive points in the selected time range.")

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
            max_speed = float(segment["speed_kmh"].max())
            avg_speed = float(segment["speed_kmh"].mean())

            st.success(f"Route distance A → B: **{dist:.2f} km**")
            st.write(f"Max speed in range: **{max_speed:.1f} km/h**")
            st.write(f"Avg speed in range: **{avg_speed:.1f} km/h**")

            route_layer = pdk.Layer(
                "PathLayer",
                data=pd.DataFrame({"path": [segment[["longitude", "latitude"]].values.tolist()], "name": [bike_name]}),
                get_path="path",
                get_width=4,
                pickable=True,
            )
            view = pdk.ViewState(
                latitude=float(segment["latitude"].mean()),
                longitude=float(segment["longitude"].mean()),
                zoom=12,
                pitch=0,
            )
            st.pydeck_chart(pdk.Deck(layers=[route_layer], initial_view_state=view, tooltip={"text": "{name}"}))

st.caption("Speed converted from knots→km/h. Distance via haversine between consecutive points ordered by fixtime.")
