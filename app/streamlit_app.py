from pathlib import Path

import folium
import numpy as np
import streamlit as st
from predictor import load_model
import pandas as pd
from streamlit_folium import st_folium
import geopandas as gpd
from shapely import wkt


# -----------------------------------------
# LOAD AND CACHE DATA
# -----------------------------------------
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

@st.cache_data
def load_segments(path):
    df = pd.read_csv(path)
    df["geometry"] = df["geometry"].apply(wkt.loads)

    return gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs="EPSG:26916"
    )

segments = load_segments(
    PROJECT_ROOT
    / "data"
    / "interim"
    / "road_segments.csv"
)

@st.cache_data
def load_traffic_stats(path):
    return pd.read_parquet(path)

traffic_stats = load_traffic_stats(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "traffic_crashes_hourly.parquet"
)

@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# -----------------------------------------
# STYLES AND CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Road Segment Accident Risk Assessment in Chicago",
    layout="wide"
)

st.title("Road Segment Accident Risk Assessment in Chicago")

left_col, right_col = st.columns(
    [1, 2]
)

if "proba" not in st.session_state:
    st.session_state.proba = None

if "risk_color" not in st.session_state:
    st.session_state.risk_color = "#333333"


# -----------------------------------------
# LEFT COLUMNS WITH CONTROLS
# -----------------------------------------
def clear_prediction():
    st.session_state.proba = None
    st.session_state.risk_color = "#333333"

with left_col:
    segment_id = st.selectbox(
        "Road Segment",
        sorted(
            segments["segment_id"].unique()
        ),
        index=800,
        on_change=clear_prediction,
    )
    hour = st.slider(
        "Hour",
        0,
        23,
        12,
        on_change=clear_prediction,
    )

    day_of_week = st.selectbox(
        "Day of Week",
        range(1, 8),
        on_change=clear_prediction,
    )

    month = st.selectbox(
        "Month",
        range(1, 13),
        on_change=clear_prediction,
    )

    traffic_row = traffic_stats[
        (traffic_stats["segment_id"] == segment_id)
        & (traffic_stats["hour"] == hour)
        & (traffic_stats["day_of_week"] == day_of_week)
        & (traffic_stats["month"] == month)
    ]

    if not traffic_row.empty:
        avg_speed = float(
            traffic_row.iloc[0]["avg_speed"]
        )

        avg_congestion = float(
            traffic_row.iloc[0]["avg_congestion"]
        )
    else:
        avg_speed = 0
        avg_congestion = 0

    st.subheader("Historical Traffic Conditions")

    st.metric(
        "Average Speed",
        f"{avg_speed:.1f} mph"
    )

    st.metric(
        "Average Congestion",
        f"{avg_congestion:.3f}"
    )

    segment = segments[segments["segment_id"] == segment_id]

    segment_lat = (
        segment.iloc[0]["start_latitude"]
        + segment.iloc[0]["end_latitude"]
    ) / 2

    segment_lon = (
        segment.iloc[0]["start_longitude"]
        + segment.iloc[0]["end_longitude"]
    ) / 2

    hour_sin = np.sin(2 * np.pi * hour / 24)

    hour_cos = np.cos(2 * np.pi * hour / 24)

    dow_sin = np.sin(2 * np.pi * day_of_week / 7)

    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    month_sin = np.sin(2 * np.pi * month / 12)

    month_cos = np.cos(2 * np.pi * month / 12)

    color = st.session_state.risk_color

    if st.button("Assess Risk"):
        X = pd.DataFrame(
            [
                {
                    "avg_speed": avg_speed,
                    "avg_free_flow_speed": avg_speed,
                    "avg_congestion": avg_congestion,
                    "segment_lat": segment_lat,
                    "segment_lon": segment_lon,
                    "hour_sin": hour_sin,
                    "hour_cos": hour_cos,
                    "dow_sin": dow_sin,
                    "dow_cos": dow_cos,
                    "month_sin": month_sin,
                    "month_cos": month_cos,
                    "segment_id": segment_id,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "month": month,
                }
            ]
        )

        proba = model.predict_proba(X)[0, 1]

        st.session_state.proba = proba

        # percentiles of predicted values are used for ranging
        # 50% - 0.369724 , 80% - 0.579984
        if proba < 0.37:
            st.session_state.risk_color = "green"

        elif proba < 0.58:
            st.session_state.risk_color = "orange"

        else:
            st.session_state.risk_color = "red"


    if st.session_state.proba is not None:
        st.metric(
            "Accident Risk Score",
            f"{st.session_state.proba:.2f}"
        )

        # percentiles of predicted values are used for ranging
        # 50% - 0.369724 , 80% - 0.579984
        if st.session_state.proba < 0.37:
            st.success("LOW RISK LEVEL")

        elif st.session_state.proba < 0.58:
            st.warning("MEDIUM RISK LEVEL")

        else:
            st.error("HIGH RISK LEVEL")

# -----------------------------------------
# RIGHT COLUMN WITH MAP
# -----------------------------------------
segment_wgs = segment.to_crs("EPSG:4326")

bounds = segment_wgs.total_bounds # [minx, miny, maxx, maxy]

m = folium.Map(
    location=[segment_lat, segment_lon],
    zoom_start=15,
    tiles="CartoDB Voyager",
)

folium.GeoJson(
    segment_wgs,
    style_function=lambda x: {
        "color": st.session_state.risk_color,
        "weight": 6,
        "opacity": 0.5,
    }
).add_to(m)

with right_col:
    st_folium(
        m,
        width=None,
        height=600
    )