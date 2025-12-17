import json
from datetime import datetime, date, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="🚲 Bike GPS Analytics (Traccar)", layout="wide")
st.title("🚲 Bike GPS Analytics (Traccar)")
st.caption("All metrics are keyed by tc_positions.deviceid. Timestamps use fixtime. Noise points are excluded by default.")


# -----------------------------
# Constants / helpers
# -----------------------------
DB_SCHEMA = "traccar_new"
POSITIONS_TABLE = f"{DB_SCHEMA}.tc_positions"

NOISE_CUTOFF = datetime(2000, 1, 1, 0, 0, 0)  # anything earlier is treated as noise
MAX_BIKES = 5


def knots_to_kmh(knots: float) -> float:
    if knots is None:
        return np.nan
    return float(knots) * 1.852


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


def dt_range_inclusive(start_d: date, end_d: date):
    start_dt = datetime.combine(start_d, time.min)
    end_dt = datetime.combine(end_d, time.max)  # 23:59:59.999999
    return start_dt, end_dt


def format_hms(seconds: float) -> str:
    if seconds is None or np.isnan(seconds) or seconds <= 0:
        return "0:00:00"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def split_session_by_day(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    """
    Split [start_ts, end_ts] into per-day segments and return list of (day, seconds_in_day_segment).
    """
    if pd.isna(start_ts) or pd.isna(end_ts):
        return []
    if end_ts <= start_ts:
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


# -----------------------------
# DB connection
# -----------------------------
@st.cache_resource
def get_engine():
    cfg = st.secrets["mysql"]
    url = URL.create(
        drivername="mysql+pymysql",
        username=cfg["username"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg.get("port", 3306)),
        database=cfg["database"],
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


engine = get_engine()


# -----------------------------
# Device list for selector
# -----------------------------
@st.cache_data(ttl=60)
def fetch_device_ids():
    # deviceid is numeric in tc_positions; grab distinct recent-ish deviceids
    q = text(f"""
        SELECT DISTINCT deviceid
        FROM {POSITIONS_TABLE}
        WHERE fixtime >= :cutoff
        ORDER BY deviceid;
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"cutoff": NOISE_CUTOFF}).fetchall()
    return [int(r[0]) for r in rows if r and r[0] is not None]


device_ids = fetch_device_ids()


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

today = date.today()
default_start = today - timedelta(days=7)
default_end = today

start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)

if start_date > end_date:
    st.sidebar.error("Start date must be <= End date.")
    st.stop()

selected_devices = st.sidebar.multiselect(
    "Select up to 5 bike device IDs",
    options=device_ids,
    default=device_ids[:1] if device_ids else [],
)

if len(selected_devices) == 0:
    st.info("Select at least one deviceid to begin.")
    st.stop()

if len(selected_devices) > MAX_BIKES:
    st.sidebar.error(f"Please select at most {MAX_BIKES} bikes.")
    st.stop()

start_dt, end_dt = dt_range_inclusive(start_date, end_date)
num_days = (end_date - start_date).days + 1


# -----------------------------
# Load positions (optimized)
# -----------------------------
@st.cache_data(ttl=60)
def fetch_positions(device_list, start_dt, end_dt):
    """
    Pull only what we need; extract JSON keys in SQL for speed.
    Filters out:
      - invalid points
      - fixtime < 2000-01-01
      - lat/lon = 0
    """
    # Build IN clause safely
    device_tuple = tuple(int(x) for x in device_list)

    q = text(f"""
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

            -- JSON extracted fields (cast to numeric where appropriate)
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.event')) AS event,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.ignition')) AS CHAR) AS ignition_raw,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.door')) AS CHAR) AS door_raw,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.motion')) AS CHAR) AS motion_raw,

            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.fuel1')) AS DECIMAL(10,4)) AS fuel1,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.fuel2')) AS DECIMAL(18,4)) AS fuel2,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.temp1')) AS DECIMAL(18,0)) AS temp1,

            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.distance')) AS DECIMAL(18,8)) AS distance,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.totalDistance')) AS DECIMAL(18,8)) AS totalDistance

        FROM {POSITIONS_TABLE}
        WHERE deviceid IN :deviceids
          AND fixtime BETWEEN :start_dt AND :end_dt
          AND fixtime >= :cutoff
          AND valid = 1
          AND latitude <> 0 AND longitude <> 0
        ORDER BY deviceid, fixtime;
    """)

    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={
            "deviceids": device_tuple,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "cutoff": NOISE_CUTOFF
        })

    # Ensure dtypes
    for c in ["servertime", "devicetime", "fixtime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Convert speed
    df["speed_kmh"] = df["speed"].apply(knots_to_kmh)

    # Normalize boolean-like JSON fields (they may come as "true"/"false"/"1"/"0"/None)
    def to_bool(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        s = str(x).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        return None

    df["ignition"] = df["ignition_raw"].apply(to_bool)
    df["door"] = df["door_raw"].apply(to_bool)        # Charging ON/OFF per your mapping
    df["motion"] = df["motion_raw"].apply(to_bool)

    # If any JSON fields failed to extract (null), attempt parse from attributes as fallback
    # (useful if MySQL JSON is stored inconsistently)
    if df["fuel1"].isna().all() or df["door"].isna().all():
        attrs = df["attributes"].apply(safe_json_load)
        if df["fuel1"].isna().all():
            df["fuel1"] = attrs.apply(lambda a: a.get("fuel1", np.nan))
        if df["door"].isna().all():
            df["door"] = attrs.apply(lambda a: a.get("door", None))
        if df["ignition"].isna().all():
            df["ignition"] = attrs.apply(lambda a: a.get("ignition", None))
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


# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_graphs, tab_charging, tab_popular, tab_route = st.tabs(
    ["Overview", "Graphs", "Charging", "Popular locations", "Route map"]
)


# -----------------------------
# Overview tab
# -----------------------------
with tab_overview:
    st.subheader("Overview (per selected bike)")

    overview_rows = []
    for deviceid, g in df.groupby("deviceid"):
        g = g.sort_values("fixtime")

        # Speed avg excluding zeros
        nonzero_speeds = g.loc[g["speed_kmh"] > 0, "speed_kmh"].dropna()
        avg_speed = float(nonzero_speeds.mean()) if len(nonzero_speeds) else np.nan
        max_speed = float(g["speed_kmh"].max()) if g["speed_kmh"].notna().any() else np.nan

        # Distance: prefer totalDistance delta (more stable), fallback to sum(distance)
        total_dist = np.nan
        if g["totalDistance"].notna().any():
            td = g["totalDistance"].dropna()
            if len(td) >= 2:
                total_dist = float(td.max() - td.min())

        if np.isnan(total_dist):
            # fallback
            total_dist = float(g["distance"].dropna().sum()) if g["distance"].notna().any() else np.nan

        avg_daily_dist = total_dist / num_days if (not np.isnan(total_dist) and num_days > 0) else np.nan

        overview_rows.append({
            "deviceid": int(deviceid),
            "Avg daily distance (km)": avg_daily_dist,
            "Avg speed (km/h) [zeros ignored]": avg_speed,
            "Max speed (km/h)": max_speed,
            "Total distance in range (km)": total_dist,
            "Points": int(len(g)),
        })

    overview_df = pd.DataFrame(overview_rows).sort_values("deviceid")
    st.dataframe(overview_df, use_container_width=True)

    st.caption("Notes: Avg speed excludes 0 km/h. Distance uses (max(totalDistance)-min(totalDistance)) when available; otherwise sums distance.")


# -----------------------------
# Graphs tab
# -----------------------------
with tab_graphs:
    st.subheader("Graphs")

    # Speed over time
    st.markdown("### 1) Speed over time (km/h)")
    fig_speed = px.line(
        df,
        x="fixtime",
        y="speed_kmh",
        color="deviceid",
        markers=False,
        hover_data=["latitude", "longitude"]
    )
    st.plotly_chart(fig_speed, use_container_width=True)

    # Distance graph
    st.markdown("### 2) Distance travelled (distance field)")
    fig_dist = px.line(
        df,
        x="fixtime",
        y="distance",
        color="deviceid",
        markers=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Temp1 graph
    st.markdown("### 3) Temp1 over time")
    fig_temp = px.line(
        df,
        x="fixtime",
        y="temp1",
        color="deviceid",
        markers=False
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    st.caption("Temp1 is your encoded mode/range value (EENNSS).")


# -----------------------------
# Charging tab
# -----------------------------
with tab_charging:
    st.subheader("Charging")

    st.caption("Charging ON/OFF is taken from attributes.door (per your mapping). SOC is attributes.fuel1.")

    charging_sessions_all = []

    # Build charging sessions per device: door False->True start, True->False end
    for deviceid, g in df.groupby("deviceid"):
        g = g.sort_values("fixtime").reset_index(drop=True)

        # If door is missing, skip
        if g["door"].isna().all():
            continue

        # Normalize door to bool where possible; treat None as False for transition logic
        door = g["door"].fillna(False).astype(bool)

        start_idxs = []
        end_idxs = []

        # Detect transitions
        prev = door.iloc[0]
        for i in range(1, len(door)):
            cur = door.iloc[i]
            if (prev is False) and (cur is True):
                start_idxs.append(i)
            if (prev is True) and (cur is False):
                end_idxs.append(i)
            prev = cur

        # If it starts in charging state, assume session begins at first row
        if door.iloc[0] is True:
            start_idxs = [0] + start_idxs

        # If it ends still charging, end at last row
        if door.iloc[-1] is True:
            end_idxs = end_idxs + [len(g) - 1]

        # Pair them safely
        pairs = []
        si = 0
        ei = 0
        while si < len(start_idxs) and ei < len(end_idxs):
            s = start_idxs[si]
            e = end_idxs[ei]
            if e <= s:
                ei += 1
                continue
            pairs.append((s, e))
            si += 1
            ei += 1

        for s, e in pairs:
            start_ts = g.loc[s, "fixtime"]
            end_ts = g.loc[e, "fixtime"]
            if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
                continue

            # Plug-in/plug-out SOC
            soc_in = g.loc[s, "fuel1"]
            soc_out = g.loc[e, "fuel1"]

            # Charging location = plug-in location
            lat_in = g.loc[s, "latitude"]
            lon_in = g.loc[s, "longitude"]

            charging_sessions_all.append({
                "deviceid": int(deviceid),
                "start": start_ts,
                "end": end_ts,
                "duration_sec": (end_ts - start_ts).total_seconds(),
                "SOC_in": soc_in,
                "SOC_out": soc_out,
                "lat_in": lat_in,
                "lon_in": lon_in,
            })

    sessions_df = pd.DataFrame(charging_sessions_all)

    if sessions_df.empty:
        st.warning("No charging sessions detected in the selected date range (after noise filtering).")
    else:
        sessions_df["start"] = pd.to_datetime(sessions_df["start"])
        sessions_df["end"] = pd.to_datetime(sessions_df["end"])
        sessions_df["duration"] = sessions_df["duration_sec"].apply(format_hms)

        st.markdown("### Charging sessions (plug-in / plug-out)")
        show_sessions = sessions_df.sort_values(["deviceid", "start"])[
            ["deviceid", "start", "end", "duration", "SOC_in", "SOC_out", "lat_in", "lon_in"]
        ]
        st.dataframe(show_sessions, use_container_width=True)

        # Daily charging totals
        st.markdown("### Daily charging totals (per device)")
        daily_rows = []
        for _, r in sessions_df.iterrows():
            for d, secs in split_session_by_day(pd.Timestamp(r["start"]), pd.Timestamp(r["end"])):
                daily_rows.append({
                    "deviceid": r["deviceid"],
                    "date": d,
                    "charging_seconds": secs
                })

        daily_df = pd.DataFrame(daily_rows)
        if not daily_df.empty:
            daily_summary = (
                daily_df.groupby(["deviceid", "date"], as_index=False)["charging_seconds"]
                .sum()
                .sort_values(["deviceid", "date"])
            )
            daily_summary["charging_time"] = daily_summary["charging_seconds"].apply(format_hms)
            st.dataframe(daily_summary[["deviceid", "date", "charging_time"]], use_container_width=True)

            fig_charge = px.bar(
                daily_summary,
                x="date",
                y="charging_seconds",
                color="deviceid",
                barmode="group",
                title="Daily charging duration (seconds)"
            )
            st.plotly_chart(fig_charge, use_container_width=True)

        # Charging locations map
        st.markdown("### Charging locations (plug-in points)")
        locs = sessions_df.dropna(subset=["lat_in", "lon_in"]).copy()
        if locs.empty:
            st.info("No valid lat/lon found for charging plug-in points.")
        else:
            locs["label"] = locs.apply(
                lambda x: f"Device {x['deviceid']} | SOC_in={x['SOC_in']} | SOC_out={x['SOC_out']}",
                axis=1
            )
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=locs,
                get_position="[lon_in, lat_in]",
                get_radius=40,
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=float(locs["lat_in"].mean()),
                longitude=float(locs["lon_in"].mean()),
                zoom=12,
                pitch=0,
            )
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "{label}"}
            ))

        # SOC graph
        st.markdown("### SOC (fuel1) over time")
        soc_df = df.dropna(subset=["fuel1"]).copy()
        fig_soc = px.line(
            soc_df,
            x="fixtime",
            y="fuel1",
            color="deviceid",
            markers=False
        )
        st.plotly_chart(fig_soc, use_container_width=True)


# -----------------------------
# Popular locations tab
# -----------------------------
with tab_popular:
    st.subheader("Popular locations (hotspots)")

    st.caption("Hotspots are computed by binning lat/lon into small grid cells and counting visits.")

    pop = df[["deviceid", "fixtime", "latitude", "longitude"]].dropna().copy()
    if pop.empty:
        st.warning("No valid points to compute hotspots.")
    else:
        # Grid binning: adjust precision as needed (3 decimals ≈ 100m; 4 decimals ≈ 10m)
        precision = st.slider("Hotspot grid precision (decimal places)", min_value=2, max_value=5, value=3)
        pop["lat_bin"] = pop["latitude"].round(precision)
        pop["lon_bin"] = pop["longitude"].round(precision)

        hotspot = (
            pop.groupby(["lat_bin", "lon_bin"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )

        top_n = st.slider("Show top N hotspots", min_value=10, max_value=500, value=100)
        hotspot = hotspot.head(top_n)

        # Prepare map
        hotspot["label"] = hotspot["count"].apply(lambda c: f"Visits: {int(c)}")

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=hotspot,
            get_position="[lon_bin, lat_bin]",
            get_radius="count * 4",
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=float(pop["latitude"].mean()),
            longitude=float(pop["longitude"].mean()),
            zoom=12,
            pitch=0,
        )

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "{label}"}
        ))

        st.markdown("### Hotspot table")
        st.dataframe(hotspot, use_container_width=True)


# -----------------------------
# Route map tab
# -----------------------------
with tab_route:
    st.subheader("Route map (polyline)")

    one_device = st.selectbox("Choose one bike (deviceid)", options=selected_devices)

    route = df[df["deviceid"] == one_device].sort_values("fixtime").copy()
    route = route.dropna(subset=["latitude", "longitude", "fixtime"])

    if route.empty:
        st.warning("No route points found for this device in the selected range.")
    else:
        # Build path
        path = route[["longitude", "latitude"]].values.tolist()

        path_df = pd.DataFrame([{
            "deviceid": int(one_device),
            "path": path
        }])

        layer = pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            get_width=4,
            pickable=False,
        )

        # start/end markers
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
            get_radius=60,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=float(route["latitude"].mean()),
            longitude=float(route["longitude"].mean()),
            zoom=12,
            pitch=0,
        )

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer, marker_layer],
            tooltip={"text": "{name}"}
        ))

        st.markdown("### Route points (preview)")
        st.dataframe(route[["fixtime", "latitude", "longitude", "speed_kmh", "fuel1", "door", "ignition"]].head(200),
                     use_container_width=True)
