import gc
from pathlib import Path
import pandas as pd

from src.features.spatial import (
    create_road_segments,
    create_gdf_road_segments,
    filter_crashes_by_street,
    assign_segments_to_crashes,
)

from src.features.temporal import (
    calculate_free_flow_speed,
    calculate_congestion,
    create_traffic_hourly,
    create_full_traffic_hourly,
    fill_empty_features,
)

from src.features.aggregation import (
    aggregate_crashes,
    merge_traffic_and_target,
    data_augmentation,
)

# ---------------------------------------
# DATA READING AND PREPROCESSING
# ---------------------------------------
print("Reading crashes file...")
df_crashes = pd.read_csv('../../data/raw/chicago_crashes_05012025_04302026.csv')

# leave relevant data about crashes
df_crashes = df_crashes[
    [
        'CRASH_RECORD_ID',
        'POSTED_SPEED_LIMIT',
        'STREET_NAME',
        'CRASH_HOUR',
        'CRASH_DAY_OF_WEEK',
        'CRASH_MONTH',
        'LATITUDE',
        'LONGITUDE',
    ]
]

# remove crash records without geolocation
df_crashes.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

print("Reading traffic file...")
df_traffic = pd.read_parquet('../../data/raw/chicago_traffic_tracker_05012025_04302026.parquet')

# leave relevant data about traffic
df_traffic = df_traffic[
    [
        'time',
        'segment_id',
        'speed',
        'street',
        'hour',
        'day_of_week',
        'month',
        'start_latitude',
        'start_longitude',
        'end_latitude',
        'end_longitude',
    ]
]


# SPATIAL FEATURES
road_segments = (
    create_road_segments(df_traffic)
)

df_crashes_filtered = (
    filter_crashes_by_street(
        df_crashes,
        road_segments,
    )
)

gdf_road_segments = (
    create_gdf_road_segments(road_segments)
)

crashes_with_segments = (
    assign_segments_to_crashes(
        df_crashes_filtered,
        gdf_road_segments,
    )
)

# create file for visualization
print('Saving file with crashes and road segments...')
crashes_with_segments.to_csv('../../data/interim/crashes_with_segments.csv', index=False)

print('Saving file with geolocation of road segments...')
gdf_road_segments.to_csv('../../data/interim/road_segments.csv', index=False)


# TEMPORAL FEATURES
df_traffic = (
    calculate_free_flow_speed(
        df_traffic
    )
)

df_traffic = (
    calculate_congestion(
        df_traffic
    )
)

traffic_hourly = (
    create_traffic_hourly(
        df_traffic
    )
)

traffic_hourly_full_grid = (
    create_full_traffic_hourly(
        traffic_hourly
    )
)

traffic_hourly_full = (
    fill_empty_features(
        traffic_hourly,
        traffic_hourly_full_grid
    )
)

# TARGET
crash_counts = (
    aggregate_crashes(
        crashes_with_segments
    )
)

# MERGE
dataset_with_target = (
    merge_traffic_and_target(
        traffic_hourly_full,
        crash_counts,
    )
)

final_dataset = (
    data_augmentation(
        dataset_with_target
    )
)

# save final dataset
print('Saving final file with traffic and crashes hourly data...')
final_dataset.to_parquet(
    '../../data/processed/traffic_crashes_hourly.parquet',
    engine="pyarrow",
    compression="snappy",
    index=False
)