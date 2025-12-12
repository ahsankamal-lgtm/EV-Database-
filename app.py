import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, date, time

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import URL
import pydeck as pdk


# =============================
# UI
# =============================
st.set_page_config(page_title="🚲 Bike GPS Analytics (Traccar)", layout="wide")
st.title("🚲 Bike GPS Analytics (Traccar)")
st.caption("All metrics are keyed by tc_positions.deviceid (device unique key).")


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

def clean_for_pydeck(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure pydeck only receives JSON-serializable values.
    """
    if df.empty:
        return df
    out = df.copy()

    # Keep only safe columns
    keep = [c for c in ["latitude", "longitude", "device_name", "speed_kmh"] if c in out.columns]
    out = out[keep].copy()

    if "latitude" in out.columns:
        out["latitude"] = out["latitude"].apply(to_plain_float)
    if "longitude" in out.columns:
        out["longitude"] = out["longitude"].apply(to_plain_float)
    if "speed_kmh" in out.columns:
        out["speed_kmh"] = out["speed_kmh"].apply(to_plain_float)
    if "device_name" in out.columns:
        out["device_name"] = out["device_name"].apply(to_plain_str)

    out = out.dropna(subset=["latitude", "longitude"]).copy()
    out = out.replace([np.inf, -np.inf], None)
    return out

def clamp_dt(x: datetime, lo: datetime, hi: datetime) -> datetime:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


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
# Engine (safe for @ in password)
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


# =============================
# DB Loads
# =============================
@st.cache_data(ttl=120, show_spinner=False)
def load_devices():
    # tc_devices.id is the device primary key that matches tc_positions.deviceid
    with get_engine().connect() as c:
        return pd.read_sql(
            text("SELECT id, name, uniqueid FROM tc_devices ORDER BY name"),
            c
        )

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


# =============================
# Analytics: Trips & Charging
# =============================
def detect_trips(df: pd.DataFrame, time_col: str, gap_minutes: int,
                 ignition_key: str = "ignition", motion_key: str = "motion"):
    """
    Trip detection keyed by deviceid.
    """
    if df.empty:
        return df.assign(trip_id=pd.Series(dtype=int)), pd.DataFrame()

    d = df.sort_values(["deviceid", time_col]).copy()
    d["prev_time"] = d.groupby("deviceid")[time_col].shift(1)
    d["gap_min"] = (d[time_col] - d["prev_time"]).dt.total_seconds() / 60.0

    attrs = d["attributes"].apply(safe_json)
    d["_ign"] = attrs.apply(lambda a: a.get(ignition_key, None))
    d["_mot"] = attrs.apply(lambda a: a.get(motion_key, None))
    d["_prev_ign"] = d.groupby("deviceid")["_ign"].shift(1)
    d["_prev_mot"] = d.groupby("deviceid")["_mot"].shift(1)

    new_trip = (
        d["prev_time"].isna()
        | (d["gap_min"] > gap_minutes)
        | ((d["_prev_ign"] == False) & (d["_ign"] == True))
        | ((d["_prev_mot"] == False) & (d["_mot"] == True))
    )
    d["trip_id"] = new_trip.groupby(d["deviceid"]).cumsum().astype(int)

    # trip distance segments
    d["prev_lat"] = d.groupby(["deviceid", "trip_id"])["latitude"].shift(1)
    d["prev_lon"] = d.groupby(["deviceid", "trip_id"])["longitude"].shift(1)
    seg = haversine_km(
        d["prev_lat"].fillna(d["latitude"]),
        d["prev_lon"].fillna(d["longitude"]),
        d["latitude"],
        d["longitude"],
    )
    d["trip_seg_km"] = np.where(d["prev_lat"].isna(), 0.0, seg)

    trips = (
        d.groupby(["deviceid", "trip_id"], as_index=False)
        .agg(
            start_time=(time_col, "min"),
            end_time=(time_col, "max"),
            points=("id", "count"),
            trip_km=("trip_seg_km", "sum"),
            max_kmh=("speed_kmh", "max"),
            avg_kmh=("speed_kmh", "mean"),
        )
    )
    trips["duration_min"] = (trips["end_time"] - trips["start_time"]).dt.total_seconds() / 60.0
    return d, trips.sort_values(["deviceid", "start_time"])


def detect_charging(df: pd.DataFrame, time_col: str,
                    mode: str,
                    charging_key: str,
                    power_key: str,
                    power_threshold: float,
                    battery_level_key: str,
                    capacity_kwh: float):
    """
    Charging detection keyed by deviceid.
    """
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

    # consumption estimate if batteryLevel exists
    batt = attrs.apply(lambda a: a.get(battery_level_key, None))
    batt = pd.to_numeric(batt, errors="coerce")
    d["battery_level"] = batt

    if d["battery_level"].notna().any():
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
    st.header("Device selection")

    devices_df = load_devices()
    if devices_df.empty:
        st.warning("No devices found in tc_devices.")
        st.stop()

    devices_df = devices_df.rename(columns={"id": "deviceid"})  # explicit key

    device_label = devices_df.apply(
        lambda r: f"{r['name']}  (deviceid={int(r['deviceid'])}, uniqueid={r['uniqueid']})",
        axis=1,
    )
    label_to_deviceid = dict(zip(device_label, devices_df["deviceid"]))

    selected_labels = st.multiselect(
        "Select bikes/devices",
        options=list(label_to_deviceid.keys()),
        default=list(label_to_deviceid.keys())[: min(5, len(label_to_deviceid))],
    )
    selected_deviceids = [int(label_to_deviceid[lbl]) for lbl in selected_labels]

    st.divider()
    st.header("Date range")
    today = date.today()
    start_date = st.date_input("Start date", value=today - timedelta(days=7))
    end_date = st.date_input("End date (inclusive)", value=today)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    st.divider()
    st.header("Trip detection")
    gap_minutes = st.slider("New trip if time gap exceeds (minutes)", 1, 120, 10)

    st.divider()
    st.header("Charging detection")
    charging_mode = st.selectbox("Charging mode", ["power_threshold", "attribute_key"])
    charging_key = st.text_input("Charging key (attribute_key mode)", value="charging")
    power_key = st.text_input("Power key (power_threshold mode)", value="power")
    power_threshold = st.number_input("Charging if power >= (volts)", value=13.0, step=0.1)

    st.divider()
    st.header("Consumption estimate")
    battery_level_key = st.text_input("Battery level key (0-100)", value="batteryLevel")
    capacity_kwh = st.number_input("Battery capacity (kWh)", value=2.0, step=0.1)

if not selected_deviceids:
    st.info("Select at least one device.")
    st.stop()


# =============================
# Load positions (keyed by deviceid)
# =============================
positions = load_positions(selected_deviceids, start_dt, end_dt)
time_col = pick_time_field(positions)

if positions.empty or time_col is None:
    st.warning("No position data found for the selected devices/date range.")
    st.stop()

positions = positions.dropna(subset=["latitude", "longitude", time_col]).copy()
positions = positions.sort_values(["deviceid", time_col]).copy()

deviceid_to_name = dict(zip(devices_df["deviceid"], devices_df["name"]))
deviceid_to_uniqueid = dict(zip(devices_df["deviceid"], devices_df["uniqueid"]))

positions["device_name"] = positions["deviceid"].map(lambda x: deviceid_to_name.get(int(x), str(x)))
positions["device_uniqueid"] = positions["deviceid"].map(lambda x: deviceid_to_uniqueid.get(int(x), ""))

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
# =============================
per_bike = (
    positions.groupby("deviceid", as_index=False)
    .agg(
        device_name=("device_name", "first"),
        uniqueid=("device_uniqueid", "first"),
        points=("id", "count"),
        total_km=("seg_km", "sum"),
        max_kmh=("speed_kmh", "max"),
        avg_kmh=("speed_kmh", "mean"),
        first_time=(time_col, "min"),
        last_time=(time_col, "max"),
    )
)

overall_avg_speed = float(per_bike["avg_kmh"].mean()) if len(per_bike) else 0.0
overall_avg_distance = float(per_bike["total_km"].mean()) if len(per_bike) else 0.0
overall_max_speed = float(per_bike["max_kmh"].max()) if len(per_bike) else 0.0

ACTIVE_KM_THRESHOLD = 0.2
daily_km = positions.groupby(["day", "deviceid"], as_index=False)["seg_km"].sum()
daily_km["active"] = daily_km["seg_km"] >= ACTIVE_KM_THRESHOLD
daily_active = daily_km.groupby("day", as_index=False)["active"].sum().rename(columns={"active": "active_bikes"})

pos_with_trips, trips = detect_trips(positions, time_col=time_col, gap_minutes=gap_minutes)
trips_per_bike = (
    trips.groupby("deviceid", as_index=False)
    .agg(total_trips=("trip_id", "nunique"), trips_km=("trip_km", "sum"))
)
trips_per_bike["device_name"] = trips_per_bike["deviceid"].map(lambda x: deviceid_to_name.get(int(x), str(x)))

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
if not charge_sessions.empty:
    charge_sessions["device_name"] = charge_sessions["deviceid"].map(lambda x: deviceid_to_name.get(int(x), str(x)))
if not charge_daily.empty:
    charge_daily["device_name"] = charge_daily["deviceid"].map(lambda x: deviceid_to_name.get(int(x), str(x)))

grid_decimals = 3
positions["cell_lat"] = np.round(positions["latitude"], grid_decimals)
positions["cell_lon"] = np.round(positions["longitude"], grid_decimals)
top_cells = (
    positions.groupby(["cell_lat", "cell_lon"], as_index=False)
    .agg(points=("id", "count"))
    .sort_values("points", ascending=False)
    .head(50)
)


# =============================
# Tabs
# =============================
tab_overview, tab_bike, tab_trips, tab_charging, tab_locations, tab_a_to_b = st.tabs(
    ["Overview", "Bike Metrics", "Trips", "Charging", "Popular Locations", "Distance A → B"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bikes selected", len(selected_deviceids))
    c2.metric("Avg speed (all bikes)", f"{overall_avg_speed:.1f} km/h")
    c3.metric("Avg distance (per bike)", f"{overall_avg_distance:.2f} km")
    c4.metric("Max speed (any bike)", f"{overall_max_speed:.1f} km/h")

    st.subheader("Daily active bikes")
    st.dataframe(daily_active, use_container_width=True)
    st.line_chart(daily_active.set_index("day")["active_bikes"])

    st.subheader("Total distance per bike (keyed by deviceid)")
    st.dataframe(per_bike.sort_values("total_km", ascending=False), use_container_width=True)
    st.bar_chart(per_bike.set_index("device_name")["total_km"])

with tab_bike:
    st.subheader("Per-bike metrics (deviceid is the key)")
    st.dataframe(per_bike.sort_values("device_name"), use_container_width=True)

    chosen_deviceid = st.selectbox(
        "Choose a bike (deviceid)",
        options=per_bike["deviceid"].tolist(),
        format_func=lambda did: f"{deviceid_to_name.get(int(did), did)} (deviceid={int(did)})",
    )

    bike_df = positions[positions["deviceid"] == int(chosen_deviceid)].sort_values(time_col).copy()

    st.markdown("**Speed over time (km/h)**")
    st.line_chart(bike_df.set_index(time_col)["speed_kmh"])

    st.markdown("**Daily distance (km)**")
    bike_daily = bike_df.groupby("day", as_index=False)["seg_km"].sum().rename(columns={"seg_km": "distance_km"})
    st.bar_chart(bike_daily.set_index("day")["distance_km"])

    st.markdown("**Max speed**")
    st.write(f"{bike_df['speed_kmh'].max():.1f} km/h")

with tab_trips:
    st.subheader("Trips (keyed by deviceid)")
    if trips.empty:
        st.info("No trips detected. Try lowering the time-gap threshold.")
    else:
        view = trips.copy()
        view["device_name"] = view["deviceid"].map(lambda x: deviceid_to_name.get(int(x), str(x)))
        st.dataframe(view.sort_values(["deviceid", "start_time"]), use_container_width=True)

        st.subheader("Total trips per bike")
        st.dataframe(trips_per_bike.sort_values("total_trips", ascending=False), use_container_width=True)

with tab_charging:
    st.subheader("Daily charging time (minutes) per bike (keyed by deviceid)")
    if charge_daily.empty:
        st.info("No charging detected. Adjust charging settings in the sidebar.")
    else:
        st.dataframe(charge_daily.sort_values(["deviceid", "day"]), use_container_width=True)

        st.subheader("Charging sessions (timestamps) per bike")
        st.dataframe(charge_sessions.sort_values(["deviceid", "start_time"]), use_container_width=True)

        st.subheader("Charging consumption (estimate)")
        st.caption("Only valid if your devices store batteryLevel (0-100) in attributes.")
        cons = charge_daily.groupby("device_name", as_index=False).agg(
            charging_minutes=("charging_minutes", "sum"),
            consumption_kwh_est=("consumption_kwh_est", "sum"),
        )
        st.dataframe(cons.sort_values("consumption_kwh_est", ascending=False), use_container_width=True)

with tab_locations:
    st.subheader("Popular location cells (top 50)")
    st.caption(f"Rounded grid: {grid_decimals} decimals.")
    st.dataframe(top_cells, use_container_width=True)

    st.subheader("Latest position map")
    latest = (
        positions.sort_values(["deviceid", time_col])
        .groupby("deviceid", as_index=False)
        .tail(1)
        .copy()
    )
    latest["device_name"] = latest["deviceid"].map(lambda x: deviceid_to_name.get(int(x), str(x)))

    map_df = clean_for_pydeck(latest)
    if map_df.empty:
        st.info("No valid map points to display.")
    else:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_radius=60,
            pickable=True,
        )
        view = pdk.ViewState(
            latitude=float(map_df["latitude"].mean()),
            longitude=float(map_df["longitude"].mean()),
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

    bike_id = st.selectbox(
        "Bike (deviceid)",
        options=per_bike["deviceid"].tolist(),
        format_func=lambda did: f"{deviceid_to_name.get(int(did), did)} (deviceid={int(did)})",
        key="a2b_deviceid",
    )

    bike_df = positions[positions["deviceid"] == int(bike_id)].sort_values(time_col).copy()

    min_dt = bike_df[time_col].min().to_pydatetime()
    max_dt = bike_df[time_col].max().to_pydatetime()

    colA, colB = st.columns(2)

    # ✅ FIX: use date_input + time_input instead of st.datetime_input
    with colA:
        a_date = st.date_input("Point A date", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date(), key="a_date")
        a_time = st.time_input("Point A time", value=min_dt.time().replace(microsecond=0), key="a_time")
        tA = datetime.combine(a_date, a_time)

    with colB:
        b_date = st.date_input("Point B date", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date(), key="b_date")
        b_time = st.time_input("Point B time", value=max_dt.time().replace(microsecond=0), key="b_time")
        tB = datetime.combine(b_date, b_time)

    # Clamp to available range (prevents empty range selection)
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
            avg_speed = float(segment["speed_kmh"].mean())

            st.success(f"Route distance A → B: **{dist:.2f} km**")
            st.write(f"Max speed in range: **{max_speed:.1f} km/h**")
            st.write(f"Avg speed in range: **{avg_speed:.1f} km/h**")

st.caption("✅ Device unique key used everywhere: tc_positions.deviceid. tc_devices.uniqueid is shown only for reference.")
