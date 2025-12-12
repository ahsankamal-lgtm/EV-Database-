import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, date
from dateutil import tz

from sqlalchemy import create_engine, text
import pydeck as pdk

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="🚲 Bike GPS Analytics (Traccar)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def get_engine():
    cfg = st.secrets["traccar"]
    # mysql+pymysql://user:pass@host:port/db
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg.get('port', 3306)}/{cfg['database']}"
        f"?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True)

def safe_json(x):
    """Parse attributes safely: can be dict already, JSON string, or None."""
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

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    R = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def knots_to_kmh(knots):
    return knots * 1.852

def day_floor(ts):
    return ts.dt.date

def pick_time_field(df, preferred="fixtime"):
    """Use fixtime if present; else devicetime; else servertime."""
    for col in [preferred, "devicetime", "servertime"]:
        if col in df.columns:
            return col
    return None

def build_grid_cell(lat, lon, decimals=3):
    """Simple 'popular locations' cell by rounding. decimals=3 ~ ~110m."""
    return (np.round(lat, decimals), np.round(lon, decimals))

def detect_trips(df, time_col, gap_minutes=10, ignition_key="ignition", motion_key="motion"):
    """
    Trip segmentation logic:
    - Sort by time
    - A new trip starts if time gap > gap_minutes OR ignition/motion indicates stop→move transition.
    Returns df with trip_id and a trips summary.
    """
    if df.empty:
        return df.assign(trip_id=pd.Series(dtype="int")), pd.DataFrame()

    d = df.sort_values([ "deviceid", time_col ]).copy()

    # time gap per device
    d["prev_time"] = d.groupby("deviceid")[time_col].shift(1)
    d["gap_min"] = (d[time_col] - d["prev_time"]).dt.total_seconds() / 60.0

    # ignition/motion states
    ign = d["attributes"].apply(lambda a: safe_json(a).get(ignition_key, None))
    mot = d["attributes"].apply(lambda a: safe_json(a).get(motion_key, None))
    d["_ign"] = ign
    d["_mot"] = mot

    # transitions (best effort)
    d["_prev_ign"] = d.groupby("deviceid")["_ign"].shift(1)
    d["_prev_mot"] = d.groupby("deviceid")["_mot"].shift(1)

    # new trip conditions:
    # 1) first record per device
    # 2) time gap > threshold
    # 3) ignition false->true or motion false->true (if present)
    new_trip = (
        d["prev_time"].isna()
        | (d["gap_min"] > gap_minutes)
        | ((d["_prev_ign"] == False) & (d["_ign"] == True))
        | ((d["_prev_mot"] == False) & (d["_mot"] == True))
    )

    d["trip_id"] = new_trip.groupby(d["deviceid"]).cumsum().astype(int)

    # compute trip distance from consecutive points within same trip
    d["prev_lat"] = d.groupby(["deviceid","trip_id"])["latitude"].shift(1)
    d["prev_lon"] = d.groupby(["deviceid","trip_id"])["longitude"].shift(1)
    seg_km = haversine_km(d["prev_lat"], d["prev_lon"], d["latitude"], d["longitude"])
    d["seg_km"] = np.where(d["prev_lat"].isna(), 0.0, seg_km)

    trips = (
        d.groupby(["deviceid","trip_id"], as_index=False)
          .agg(
              start_time=(time_col, "min"),
              end_time=(time_col, "max"),
              points=("id", "count"),
              trip_km=("seg_km", "sum"),
              max_kmh=("speed_kmh", "max"),
              avg_kmh=("speed_kmh", "mean"),
          )
    )
    trips["duration_min"] = (trips["end_time"] - trips["start_time"]).dt.total_seconds() / 60.0
    trips = trips.sort_values(["deviceid","start_time"])

    return d, trips

def detect_charging_sessions(df, time_col, charging_mode="power_threshold",
                             charging_key="charging", power_key="power",
                             power_threshold=13.0,
                             battery_level_key="batteryLevel",
                             capacity_kwh=2.0):
    """
    Charging sessions per device, per day.

    charging_mode options:
    - "attribute_key": uses attributes[charging_key] boolean
    - "power_threshold": uses attributes[power_key] >= threshold to mark charging

    Consumption:
    - If battery level key exists (0-100), consumption_kwh ~ delta% * capacity_kwh
      (Best-effort; depends on device sending batteryLevel)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = df.sort_values(["deviceid", time_col]).copy()
    attrs = d["attributes"].apply(safe_json)

    if charging_mode == "attribute_key":
        charging_state = attrs.apply(lambda a: a.get(charging_key, None))
        # normalize to bool where possible
        d["is_charging"] = charging_state.apply(lambda v: bool(v) if v is not None else False)
    else:
        power_vals = attrs.apply(lambda a: a.get(power_key, None))
        power_num = pd.to_numeric(power_vals, errors="coerce")
        d["power_val"] = power_num
        d["is_charging"] = d["power_val"].fillna(-1) >= power_threshold

    d["prev_charge"] = d.groupby("deviceid")["is_charging"].shift(1)
    d["charge_start"] = (d["is_charging"] == True) & (d["prev_charge"] == False)
    d["charge_end"] = (d["is_charging"] == False) & (d["prev_charge"] == True)

    # session id increments on every start
    d["charge_session_id"] = d["charge_start"].groupby(d["deviceid"]).cumsum()

    # Only rows within charging periods (including start rows)
    charging_rows = d[d["is_charging"] == True].copy()
    if charging_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    # session summary
    sessions = (
        charging_rows.groupby(["deviceid", "charge_session_id"], as_index=False)
        .agg(
            start_time=(time_col, "min"),
            end_time=(time_col, "max"),
            samples=("id", "count"),
        )
    )
    sessions["duration_min"] = (sessions["end_time"] - sessions["start_time"]).dt.total_seconds() / 60.0

    # consumption best-effort based on batteryLevel delta
    batt = attrs.apply(lambda a: a.get(battery_level_key, None))
    batt_num = pd.to_numeric(batt, errors="coerce")
    d["battery_level"] = batt_num

    # Merge start/end battery level per session if available
    if d["battery_level"].notna().any():
        start_batt = d.groupby(["deviceid","charge_session_id"])[["battery_level", time_col]].apply(
            lambda g: g.sort_values(time_col)["battery_level"].dropna().head(1)
        )
        end_batt = d.groupby(["deviceid","charge_session_id"])[["battery_level", time_col]].apply(
            lambda g: g.sort_values(time_col)["battery_level"].dropna().tail(1)
        )
        # flatten
        start_batt = start_batt.reset_index().rename(columns={"battery_level":"start_battery"})
        end_batt = end_batt.reset_index().rename(columns={"battery_level":"end_battery"})

        sessions = sessions.merge(start_batt[["deviceid","charge_session_id","start_battery"]], on=["deviceid","charge_session_id"], how="left")
        sessions = sessions.merge(end_batt[["deviceid","charge_session_id","end_battery"]], on=["deviceid","charge_session_id"], how="left")

        sessions["battery_gain_pct"] = sessions["end_battery"] - sessions["start_battery"]
        sessions["consumption_kwh_est"] = (sessions["battery_gain_pct"] / 100.0) * capacity_kwh
    else:
        sessions["consumption_kwh_est"] = np.nan

    # daily charging totals
    sessions["day"] = sessions["start_time"].dt.date
    daily = (
        sessions.groupby(["deviceid","day"], as_index=False)
        .agg(
            charging_minutes=("duration_min", "sum"),
            sessions=("charge_session_id", "nunique"),
            consumption_kwh_est=("consumption_kwh_est", "sum"),
        )
    )

    return sessions.sort_values(["deviceid","start_time"]), daily.sort_values(["deviceid","day"])

# -----------------------------
# Data access
# -----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_devices():
    eng = get_engine()
    # device table name in Traccar is typically tc_devices
    q = text("""
        SELECT id, name, uniqueid
        FROM tc_devices
        ORDER BY name
    """)
    with eng.connect() as c:
        df = pd.read_sql(q, c)
    return df

@st.cache_data(ttl=60, show_spinner=True)
def load_positions(device_ids, start_dt, end_dt):
    eng = get_engine()

    # Pull what we need from tc_positions
    # Note: attributes/network may be JSON or text depending on schema.
    q = text("""
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

    with eng.connect() as c:
        df = pd.read_sql(
            q,
            c,
            params={
                "device_ids": tuple(device_ids),
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        )

    # normalize datetimes
    for col in ["servertime", "devicetime", "fixtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # speed conversions
    df["speed_kmh"] = knots_to_kmh(pd.to_numeric(df["speed"], errors="coerce").fillna(0.0))

    return df

# -----------------------------
# UI
# -----------------------------
st.title("🚲 Bike GPS Analytics (Traccar)")
st.caption("Powered by tc_positions. Metrics: distance, speed, trips, charging, active bikes, popular locations.")

with st.sidebar:
    st.header("Filters")

    devices_df = load_devices()
    device_name_map = dict(zip(devices_df["name"], devices_df["id"]))
    device_unique_map = dict(zip(devices_df["id"], devices_df["uniqueid"]))

    selected_names = st.multiselect(
        "Select bikes/devices",
        options=list(device_name_map.keys()),
        default=list(device_name_map.keys())[:5] if len(device_name_map) else [],
    )
    selected_device_ids = [device_name_map[n] for n in selected_names]

    today = date.today()
    default_start = today - timedelta(days=7)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date (inclusive)", value=today)

    # Build datetime range [start, end+1)
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    st.divider()
    st.subheader("Trip detection")
    gap_minutes = st.slider("New trip if time gap exceeds (minutes)", 1, 120, 10)
    ignition_key = st.text_input("Ignition key in attributes (optional)", value="ignition")
    motion_key = st.text_input("Motion key in attributes (optional)", value="motion")

    st.divider()
    st.subheader("Charging detection")
    charging_mode = st.selectbox("Charging mode", ["power_threshold", "attribute_key"])
    charging_key = st.text_input("Charging boolean key (if using attribute_key)", value="charging")
    power_key = st.text_input("Power key (if using power_threshold)", value="power")
    power_threshold = st.number_input("Power threshold (charging if power >=)", value=13.0, step=0.1)

    st.divider()
    st.subheader("Consumption (estimate)")
    battery_level_key = st.text_input("Battery level key (0-100)", value="batteryLevel")
    capacity_kwh = st.number_input("Battery capacity (kWh) for estimate", value=2.0, step=0.1)

# Guard
if not selected_device_ids:
    st.info("Select at least one bike/device from the sidebar.")
    st.stop()

# Load data
positions = load_positions(selected_device_ids, start_dt, end_dt)
time_col = pick_time_field(positions, "fixtime")

if positions.empty:
    st.warning("No positions found for the selected devices and date range.")
    st.stop()

# Ensure required columns
positions = positions.dropna(subset=["latitude", "longitude", time_col]).copy()

# Compute segment distances per device (for total distance & daily distance)
positions = positions.sort_values(["deviceid", time_col]).copy()
positions["prev_lat"] = positions.groupby("deviceid")["latitude"].shift(1)
positions["prev_lon"] = positions.groupby("deviceid")["longitude"].shift(1)
seg_km = haversine_km(positions["prev_lat"], positions["prev_lon"], positions["latitude"], positions["longitude"])
positions["seg_km"] = np.where(positions["prev_lat"].isna(), 0.0, seg_km)
positions["day"] = positions[time_col].dt.date

# Per-bike totals
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
per_bike["device_name"] = per_bike["deviceid"].map(lambda i: devices_df.set_index("id").loc[i, "name"] if i in set(devices_df["id"]) else str(i))
per_bike["uniqueid"] = per_bike["deviceid"].map(lambda i: device_unique_map.get(i, ""))

# Overall averages
overall_avg_speed = float(per_bike["avg_kmh"].mean()) if len(per_bike) else 0.0
overall_avg_distance = float(per_bike["total_km"].mean()) if len(per_bike) else 0.0
overall_max_speed = float(per_bike["max_kmh"].max()) if len(per_bike) else 0.0

# Daily active bikes: a bike is "active" if it moved > 0.2km (configurable) OR has motion true
ACTIVE_KM_THRESHOLD = 0.2
daily_km = positions.groupby(["day", "deviceid"], as_index=False)["seg_km"].sum()
daily_km["active"] = daily_km["seg_km"] >= ACTIVE_KM_THRESHOLD
daily_active = daily_km.groupby("day", as_index=False)["active"].sum().rename(columns={"active": "active_bikes"})

# Trips
pos_with_trips, trips = detect_trips(
    positions.assign(attributes=positions["attributes"]),
    time_col=time_col,
    gap_minutes=gap_minutes,
    ignition_key=ignition_key.strip(),
    motion_key=motion_key.strip(),
)

# Total trips per bike
trips_per_bike = trips.groupby("deviceid", as_index=False).agg(total_trips=("trip_id", "nunique"), trips_km=("trip_km", "sum"))

# Charging
charge_sessions, charge_daily = detect_charging_sessions(
    positions.assign(attributes=positions["attributes"]),
    time_col=time_col,
    charging_mode=charging_mode,
    charging_key=charging_key.strip(),
    power_key=power_key.strip(),
    power_threshold=float(power_threshold),
    battery_level_key=battery_level_key.strip(),
    capacity_kwh=float(capacity_kwh),
)

# Popular locations (grid)
grid_decimals = 3
lat_round, lon_round = build_grid_cell(positions["latitude"].values, positions["longitude"].values, decimals=grid_decimals)
positions["cell_lat"] = lat_round
positions["cell_lon"] = lon_round
top_cells = (
    positions.groupby(["cell_lat", "cell_lon"], as_index=False)
    .agg(points=("id", "count"))
    .sort_values("points", ascending=False)
    .head(50)
)

# -----------------------------
# Tabs
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
    st.subheader("Per-bike speed & distance")
    st.dataframe(per_bike.sort_values("device_name"), use_container_width=True)

    chosen_bike = st.selectbox("Choose a bike", options=per_bike["device_name"].tolist())
    chosen_id = int(per_bike.loc[per_bike["device_name"] == chosen_bike, "deviceid"].iloc[0])

    bike_df = positions[positions["deviceid"] == chosen_id].copy()

    st.markdown("**Speed over time (km/h)**")
    speed_series = bike_df.set_index(time_col)["speed_kmh"].sort_index()
    st.line_chart(speed_series)

    st.markdown("**Daily distance (km)**")
    bike_daily = bike_df.groupby("day", as_index=False)["seg_km"].sum().rename(columns={"seg_km":"distance_km"})
    st.bar_chart(bike_daily.set_index("day")["distance_km"])

    st.markdown("**Max speed**")
    st.write(f"{bike_df['speed_kmh'].max():.1f} km/h")

with tab_trips:
    st.subheader("Trips summary (per bike)")
    if trips.empty:
        st.info("No trips detected for the current settings. Try lowering the time-gap threshold.")
    else:
        trips_view = trips.copy()
        trips_view["device_name"] = trips_view["deviceid"].map(lambda i: devices_df.set_index("id").loc[i, "name"] if i in set(devices_df["id"]) else str(i))
        st.dataframe(trips_view.sort_values(["device_name", "start_time"]), use_container_width=True)

        st.subheader("Total trips per bike")
        tpb = trips_per_bike.copy()
        tpb["device_name"] = tpb["deviceid"].map(lambda i: devices_df.set_index("id").loc[i, "name"] if i in set(devices_df["id"]) else str(i))
        st.dataframe(tpb.sort_values("total_trips", ascending=False), use_container_width=True)

with tab_charging:
    st.subheader("Daily charging time (minutes) per bike")
    if charge_daily.empty:
        st.info("No charging detected with current charging logic. Adjust charging mode / key / threshold in sidebar.")
    else:
        cd = charge_daily.copy()
        cd["device_name"] = cd["deviceid"].map(lambda i: devices_df.set_index("id").loc[i, "name"] if i in set(devices_df["id"]) else str(i))
        st.dataframe(cd.sort_values(["device_name", "day"]), use_container_width=True)

        st.subheader("Charging sessions (timestamps) per bike")
        cs = charge_sessions.copy()
        cs["device_name"] = cs["deviceid"].map(lambda i: devices_df.set_index("id").loc[i, "name"] if i in set(devices_df["id"]) else str(i))
        st.dataframe(cs.sort_values(["device_name","start_time"]), use_container_width=True)

        st.subheader("Charging consumption (estimate)")
        st.caption("Shown only if battery level is available in attributes (e.g., batteryLevel 0-100).")
        cons = cd.groupby("device_name", as_index=False).agg(
            charging_minutes=("charging_minutes", "sum"),
            consumption_kwh_est=("consumption_kwh_est", "sum"),
        )
        st.dataframe(cons.sort_values("consumption_kwh_est", ascending=False), use_container_width=True)

with tab_locations:
    st.subheader("Popular location cells (top 50)")
    st.caption(f"Grid rounding: {grid_decimals} decimals (~110m). Higher 'points' = more visits/samples.")
    st.dataframe(top_cells, use_container_width=True)

    st.subheader("Map (latest points)")
    latest = positions.sort_values([ "deviceid", time_col ]).groupby("deviceid", as_index=False).tail(1).copy()
    latest["device_name"] = latest["deviceid"].map(lambda i: devices_df.set_index("id").loc[i, "name"] if i in set(devices_df["id"]) else str(i))

    # Pydeck map
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
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{device_name}\nSpeed: {speed_kmh} km/h"}))

with tab_a_to_b:
    st.subheader("Distance from point A to point B (per bike)")
    st.caption("Select a bike and two timestamps. Distance is calculated along the route between those times (sum of segments).")

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
            seg_dist = float(segment["seg_km"].sum())
            max_speed = float(segment["speed_kmh"].max())
            avg_speed = float(segment["speed_kmh"].mean())

            st.success(f"Route distance A → B: **{seg_dist:.2f} km**")
            st.write(f"Max speed in range: **{max_speed:.1f} km/h**")
            st.write(f"Avg speed in range: **{avg_speed:.1f} km/h**")

            # map route
            route_layer = pdk.Layer(
                "PathLayer",
                data=pd.DataFrame({
                    "path": [segment[["longitude","latitude"]].values.tolist()],
                    "name": [bike_name]
                }),
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

# Footer
st.caption("Notes: Speed is converted from knots to km/h. Distance uses haversine between consecutive points (fixtime order). Charging & trips depend on available attributes and sidebar settings.")
