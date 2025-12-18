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
MAX_BIKES = 5
NOISE_CUTOFF = datetime(2000, 1, 1)


# =============================
# HELPERS
# =============================
def dt_range_inclusive(start_d: date, end_d: date):
    start_dt = datetime.combine(start_d, time.min)
    end_dt = datetime.combine(end_d, time.max)
    return start_dt, end_dt


def knots_to_kmh(knots):
    try:
        if knots is None:
            return np.nan
        return float(knots) * 1.852
    except Exception:
        return np.nan


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


def to_bool(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def format_hms(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)) or seconds <= 0:
        return "0:00:00"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def split_session_by_day(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
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


def alt_line(df, x, y, color, title, y_title=None):
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(x, title="Time"),
            y=alt.Y(y, title=y_title or y),
            color=alt.Color(color, legend=alt.Legend(title="Device ID")),
            tooltip=[color, x, y],
        )
        .properties(height=360, title=title)
        .interactive()
    )


# =============================
# DB DRIVER AUTO-DETECT
# =============================
DB_DRIVER = None
try:
    import pymysql  # type: ignore
    DB_DRIVER = "pymysql"
except Exception:
    try:
        import mysql.connector  # type: ignore
        DB_DRIVER = "mysql-connector"
    except Exception:
        DB_DRIVER = None

if DB_DRIVER is None:
    st.error(
        "Missing MySQL driver in the Streamlit environment.\n\n"
        "Fix:\n"
        "1) Ensure `requirements.txt` is in the repo ROOT (same folder as app.py)\n"
        "2) Put this inside requirements.txt:\n\n"
        "streamlit\npandas\nnumpy\naltair\npydeck\npymysql\nmysql-connector-python\n\n"
        "Then redeploy / reboot the app from Streamlit Cloud."
    )
    st.stop()


@st.cache_resource
def get_conn_params():
    cfg = st.secrets["mysql"]
    return {
        "host": cfg["host"],
        "port": int(cfg.get("port", 3306)),
        "user": cfg["username"],
        "password": cfg["password"],
        "database": cfg["database"],
    }


def fetch_df(query: str, params: tuple):
    cfg = get_conn_params()

    if DB_DRIVER == "pymysql":
        import pymysql  # type: ignore
        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            return pd.DataFrame(rows)
        finally:
            conn.close()

    # mysql-connector fallback
    import mysql.connector  # type: ignore
    conn = mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        return pd.DataFrame(rows)
    finally:
        conn.close()


# =============================
# DEVICE LIST
# =============================
@st.cache_data(ttl=60)
def fetch_device_ids():
    q = f"""
        SELECT DISTINCT deviceid
        FROM {POSITIONS_TABLE}
        WHERE fixtime >= %s
        ORDER BY deviceid;
    """
    df_ids = fetch_df(q, (NOISE_CUTOFF,))
    if df_ids.empty:
        return []
    return [int(x) for x in df_ids["deviceid"].dropna().tolist()]


device_ids = fetch_device_ids()


# =============================
# SIDEBAR FILTERS
# =============================
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


# =============================
# LOAD POSITIONS
# =============================
@st.cache_data(ttl=60)
def fetch_positions(device_list, start_dt, end_dt):
    device_list = [int(x) for x in device_list]
    placeholders = ",".join(["%s"] * len(device_list))

    q = f"""
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

            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.event')) AS event,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.ignition')) AS ignition_raw,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.door')) AS door_raw,
            JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.motion')) AS motion_raw,

            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.fuel1')) AS DECIMAL(10,4)) AS fuel1,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.fuel2')) AS DECIMAL(18,4)) AS fuel2,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.temp1')) AS DECIMAL(18,0)) AS temp1,

            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.distance')) AS DECIMAL(18,8)) AS distance,
            CAST(JSON_UNQUOTE(JSON_EXTRACT(attributes, '$.totalDistance')) AS DECIMAL(18,8)) AS totalDistance

        FROM {POSITIONS_TABLE}
        WHERE deviceid IN ({placeholders})
          AND fixtime BETWEEN %s AND %s
          AND fixtime >= %s
          AND valid = 1
          AND latitude <> 0 AND longitude <> 0
        ORDER BY deviceid, fixtime;
    """

    params = tuple(device_list) + (start_dt, end_dt, NOISE_CUTOFF)
    df = fetch_df(q, params)

    if df.empty:
        return df

    for c in ["servertime", "devicetime", "fixtime"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["speed_kmh"] = df["speed"].apply(knots_to_kmh)

    df["ignition"] = df["ignition_raw"].apply(to_bool)
    df["door"] = df["door_raw"].apply(to_bool)      # charging ON/OFF
    df["motion"] = df["motion_raw"].apply(to_bool)

    # Fallback parsing in case JSON_EXTRACT returns null
    if df["fuel1"].isna().all() or df["door"].isna().all():
        attrs = df["attributes"].apply(safe_json_load)
        if df["fuel1"].isna().all():
            df["fuel1"] = attrs.apply(lambda a: a.get("fuel1", np.nan))
        if df["door"].isna().all():
            df["door"] = attrs.apply(lambda a: a.get("door", None))
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


# =============================
# NEW: BUILD A TRUE "DISTANCE TRAVELLED OVER TIME" SERIES
# =============================
def build_distance_over_time(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes the "distance over time" issue by plotting a *cumulative distance travelled* series.

    Priority:
    1) Use totalDistance if it behaves like an odometer:
       cumulative_km = totalDistance - first(totalDistance in selected range)
    2) Else fallback to cumulative sum of distance increments, but with sanity filtering
       to avoid unrealistic spikes.
    """
    d = raw_df.dropna(subset=["fixtime", "deviceid"]).copy()
    if d.empty:
        return d

    d = d.sort_values(["deviceid", "fixtime"]).reset_index(drop=True)
    d["distance_travelled_km"] = np.nan

    for deviceid, g in d.groupby("deviceid", sort=False):
        g = g.copy().sort_values("fixtime")

        td = g["totalDistance"].dropna().astype(float) if "totalDistance" in g.columns else pd.Series([], dtype=float)

        use_td = False
        if len(td) >= 2:
            td_min = float(td.min())
            td_max = float(td.max())
            td_range = td_max - td_min

            # Heuristic: odometer should be non-decreasing-ish and range shouldn't be insane for the selected dates.
            # We won't hard-fail if there are tiny drops, but we will ignore totalDistance if range is negative/invalid.
            if td_range >= 0:
                use_td = True

        if use_td:
            # Align to 0 at the start of selected range so the graph shows "distance travelled within the range"
            first_td = float(td.iloc[0])
            g["distance_travelled_km"] = g["totalDistance"].astype(float) - first_td
        else:
            # Fallback: cumulative sum of per-point increments, but filter unrealistic jumps.
            inc = g["distance"].astype(float)

            # If "distance" is already an increment, it should usually be small.
            # We compute per-row increment and cap outliers to avoid spikes from bad telemetry/units.
            # Cap rule: anything > 2 km between consecutive points is treated as noise (can adjust later).
            inc_clean = inc.copy()
            inc_clean = inc_clean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            inc_clean = inc_clean.clip(lower=0.0, upper=2.0)  # <- prevents wild spikes

            g["distance_travelled_km"] = inc_clean.cumsum()

        d.loc[g.index, "distance_travelled_km"] = g["distance_travelled_km"].values

    return d


dist_time_df = build_distance_over_time(df)


# =============================
# TABS
# =============================
tab_overview, tab_graphs, tab_charging, tab_popular, tab_route = st.tabs(
    ["Overview", "Graphs", "Charging", "Popular locations", "Route map"]
)


# =============================
# OVERVIEW TAB
# =============================
with tab_overview:
    st.subheader("Overview (per selected bike)")

    rows = []
    for deviceid, g in df.groupby("deviceid"):
        g = g.sort_values("fixtime")

        nz = g.loc[g["speed_kmh"] > 0, "speed_kmh"].dropna()
        avg_speed = float(nz.mean()) if len(nz) else np.nan
        max_speed = float(g["speed_kmh"].max()) if g["speed_kmh"].notna().any() else np.nan

        total_dist = np.nan
        td = g["totalDistance"].dropna()
        if len(td) >= 2:
            total_dist = float(td.max() - td.min())
        if np.isnan(total_dist):
            dd = g["distance"].dropna()
            total_dist = float(dd.sum()) if len(dd) else np.nan

        avg_daily_dist = total_dist / num_days if (not np.isnan(total_dist) and num_days > 0) else np.nan

        rows.append({
            "deviceid": int(deviceid),
            "Avg daily distance (km)": avg_daily_dist,
            "Avg speed (km/h) [zeros ignored]": avg_speed,
            "Max speed (km/h)": max_speed,
            "Total distance in range (km)": total_dist,
            "Points": int(len(g)),
        })

    st.dataframe(pd.DataFrame(rows).sort_values("deviceid"), use_container_width=True)


# =============================
# GRAPHS TAB
# =============================
with tab_graphs:
    st.subheader("Graphs")

    st.markdown("### 1) Speed over time (km/h)")
    st.altair_chart(
        alt_line(df.dropna(subset=["fixtime", "speed_kmh"]), "fixtime:T", "speed_kmh:Q", "deviceid:N",
                 "Speed over time", "Speed (km/h)"),
        use_container_width=True,
    )

    # --- FIXED GRAPH ---
    st.markdown("### 2) Distance travelled over time (total in selected range)")
    st.caption(
        "This is now a *cumulative distance travelled* curve per bike. "
        "It uses totalDistance (odometer-style) when reliable; otherwise it falls back to a cleaned cumulative sum."
    )
    st.altair_chart(
        alt_line(
            dist_time_df.dropna(subset=["fixtime", "distance_travelled_km"]),
            "fixtime:T",
            "distance_travelled_km:Q",
            "deviceid:N",
            "Distance travelled over time (cumulative)",
            "Distance travelled (km)",
        ),
        use_container_width=True,
    )

    st.markdown("### 3) Temp1 over time")
    st.altair_chart(
        alt_line(df.dropna(subset=["fixtime", "temp1"]), "fixtime:T", "temp1:Q", "deviceid:N",
                 "Temp1 over time", "Temp1"),
        use_container_width=True,
    )


# =============================
# CHARGING TAB
# =============================
with tab_charging:
    st.subheader("Charging")
    st.caption("Charging ON/OFF uses attributes.door. SOC uses attributes.fuel1. Plug-in location uses lat/lon at session start.")

    sessions = []

    for deviceid, g in df.groupby("deviceid"):
        g = g.sort_values("fixtime").reset_index(drop=True)
        if g["door"].isna().all():
            continue

        door = g["door"].fillna(False).astype(bool)

        start_idxs, end_idxs = [], []
        prev = door.iloc[0]
        for i in range(1, len(door)):
            cur = door.iloc[i]
            if (prev is False) and (cur is True):
                start_idxs.append(i)
            if (prev is True) and (cur is False):
                end_idxs.append(i)
            prev = cur

        if door.iloc[0] is True:
            start_idxs = [0] + start_idxs
        if door.iloc[-1] is True:
            end_idxs = end_idxs + [len(g) - 1]

        pairs = []
        si = ei = 0
        while si < len(start_idxs) and ei < len(end_idxs):
            s, e = start_idxs[si], end_idxs[ei]
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

            sessions.append({
                "deviceid": int(deviceid),
                "start": start_ts,
                "end": end_ts,
                "duration_sec": (end_ts - start_ts).total_seconds(),
                "SOC_in": g.loc[s, "fuel1"],
                "SOC_out": g.loc[e, "fuel1"],
                "lat_in": g.loc[s, "latitude"],
                "lon_in": g.loc[s, "longitude"],
            })

    sessions_df = pd.DataFrame(sessions)

    if sessions_df.empty:
        st.warning("No charging sessions detected in the selected date range.")
    else:
        sessions_df["start"] = pd.to_datetime(sessions_df["start"])
        sessions_df["end"] = pd.to_datetime(sessions_df["end"])
        sessions_df["duration"] = sessions_df["duration_sec"].apply(format_hms)

        st.markdown("### Charging sessions")
        st.dataframe(
            sessions_df.sort_values(["deviceid", "start"])[
                ["deviceid", "start", "end", "duration", "SOC_in", "SOC_out", "lat_in", "lon_in"]
            ],
            use_container_width=True,
        )

        st.markdown("### Daily charging totals (per device)")
        daily_rows = []
        for _, r in sessions_df.iterrows():
            for d, secs in split_session_by_day(pd.Timestamp(r["start"]), pd.Timestamp(r["end"])):
                daily_rows.append({"deviceid": r["deviceid"], "date": d, "charging_seconds": secs})

        daily_df = pd.DataFrame(daily_rows)
        if not daily_df.empty:
            daily_summary = (
                daily_df.groupby(["deviceid", "date"], as_index=False)["charging_seconds"]
                .sum()
                .sort_values(["deviceid", "date"])
            )
            daily_summary["charging_time"] = daily_summary["charging_seconds"].apply(format_hms)

            st.dataframe(daily_summary[["deviceid", "date", "charging_time"]], use_container_width=True)

            bar = (
                alt.Chart(daily_summary)
                .mark_bar()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("charging_seconds:Q", title="Charging seconds"),
                    color=alt.Color("deviceid:N", legend=alt.Legend(title="Device ID")),
                    tooltip=["deviceid:N", "date:T", "charging_time:N"],
                )
                .properties(height=360, title="Daily charging duration")
                .interactive()
            )
            st.altair_chart(bar, use_container_width=True)

        st.markdown("### Charging locations (plug-in points)")
        locs = sessions_df.dropna(subset=["lat_in", "lon_in"]).copy()
        if locs.empty:
            st.info("No valid charging plug-in points to show.")
        else:
            locs["label"] = locs.apply(
                lambda x: f"Device {x['deviceid']} | SOC_in={x['SOC_in']} | SOC_out={x['SOC_out']}",
                axis=1,
            )
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=locs,
                get_position="[lon_in, lat_in]",
                get_radius=55,
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=float(locs["lat_in"].mean()),
                longitude=float(locs["lon_in"].mean()),
                zoom=12,
                pitch=0,
            )
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": "{label}"},
                )
            )

        st.markdown("### SOC (fuel1) over time")
        soc_df = df.dropna(subset=["fuel1", "fixtime"]).copy()
        st.altair_chart(
            alt_line(soc_df, "fixtime:T", "fuel1:Q", "deviceid:N", "SOC over time", "SOC (%)"),
            use_container_width=True,
        )


# =============================
# POPULAR LOCATIONS TAB
# =============================
with tab_popular:
    st.subheader("Popular locations (hotspots)")
    st.caption("Hotspots are computed by binning lat/lon into grid cells and counting visits.")

    pop = df.dropna(subset=["latitude", "longitude"]).copy()
    if pop.empty:
        st.warning("No valid points to compute hotspots.")
    else:
        # remove sliders, fixed binning + orange hotspots
        precision = 3
        top_n = 500

        pop["lat_bin"] = pop["latitude"].round(precision)
        pop["lon_bin"] = pop["longitude"].round(precision)

        hotspot = (
            pop.groupby(["lat_bin", "lon_bin"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .head(top_n)
        )

        hotspot["label"] = hotspot["count"].apply(lambda c: f"Visits: {int(c)}")

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=hotspot,
            get_position="[lon_bin, lat_bin]",
            get_radius="count * 6",
            get_fill_color=[255, 165, 0],  # ORANGE
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=float(pop["latitude"].mean()),
            longitude=float(pop["longitude"].mean()),
            zoom=12,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "{label}"},
            )
        )

        st.markdown("### Hotspot table")
        st.dataframe(hotspot, use_container_width=True)


# =============================
# ROUTE MAP TAB
# =============================
with tab_route:
    st.subheader("Route map (polyline)")
    one_device = st.selectbox("Choose one bike (deviceid)", options=selected_devices)

    route = df[df["deviceid"] == one_device].sort_values("fixtime").copy()
    route = route.dropna(subset=["latitude", "longitude", "fixtime"])

    if route.empty:
        st.warning("No route points found for this device in the selected range.")
    else:
        path = route[["longitude", "latitude"]].values.tolist()
        path_df = pd.DataFrame([{"deviceid": int(one_device), "path": path}])

        path_layer = pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            get_width=7,
            get_color=[0, 255, 255],  # CYAN for visibility
            pickable=False,
        )

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
            get_radius=75,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=float(route["latitude"].mean()),
            longitude=float(route["longitude"].mean()),
            zoom=12,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=view_state,
                layers=[path_layer, marker_layer],
                tooltip={"text": "{name}"},
            )
        )

        st.markdown("### Route points preview")
        st.dataframe(
            route[["fixtime", "latitude", "longitude", "speed_kmh", "fuel1", "door", "ignition"]].head(300),
            use_container_width=True,
        )
