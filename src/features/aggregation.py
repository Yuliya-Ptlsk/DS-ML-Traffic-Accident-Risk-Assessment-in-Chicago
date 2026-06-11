import numpy as np

from src.utils.mappers import congestion_level_mapper, time_period_mapper


def aggregate_crashes(
    crashes_with_segments
):
    crashes_aggregated = (
        crashes_with_segments
        .groupby(
            [
                'segment_id',
                'CRASH_HOUR',
                'CRASH_DAY_OF_WEEK',
                'CRASH_MONTH'
            ]
        )
        .size()
        .reset_index(name='crash_count')
    )

    return crashes_aggregated

def merge_traffic_and_target(
    traffic_hourly_full,
    crash_counts
):
    traffic_crashes_hourly = (
        traffic_hourly_full
        .merge(
            crash_counts,
            left_on=[
                'segment_id',
                'hour',
                'day_of_week',
                'month'
            ],
            right_on=[
                'segment_id',
                'CRASH_HOUR',
                'CRASH_DAY_OF_WEEK',
                'CRASH_MONTH'
            ],
            how='left'
        )
    )

    # remove duplicated columns
    traffic_crashes_hourly = traffic_crashes_hourly.drop(
        columns=[
            'CRASH_HOUR',
            'CRASH_DAY_OF_WEEK',
            'CRASH_MONTH'
        ],
        errors='ignore'
    )

    # fill records without crashes with 0 value
    traffic_crashes_hourly['crash_count'] = (
        traffic_crashes_hourly['crash_count']
        .fillna(0)
        .astype(int)
    )

    # create target column
    traffic_crashes_hourly['accident'] = (
        traffic_crashes_hourly['crash_count'] > 0
    ).astype(int)

    return traffic_crashes_hourly


def data_augmentation(traffic_crashes_hourly):
    traffic_crashes_hourly["hour_sin"] = np.sin(
        2 * np.pi * traffic_crashes_hourly["hour"] / 24
    )

    traffic_crashes_hourly["hour_cos"] = np.cos(
        2 * np.pi * traffic_crashes_hourly["hour"] / 24
    )

    traffic_crashes_hourly["dow_sin"] = np.sin(
        2 * np.pi * traffic_crashes_hourly["day_of_week"] / 7
    )

    traffic_crashes_hourly["dow_cos"] = np.cos(
        2 * np.pi * traffic_crashes_hourly["day_of_week"] / 7
    )

    traffic_crashes_hourly["month_sin"] = np.sin(
        2 * np.pi * traffic_crashes_hourly["month"] / 12
    )

    traffic_crashes_hourly["month_cos"] = np.cos(
        2 * np.pi * traffic_crashes_hourly["month"] / 12
    )

    traffic_crashes_hourly["time_period"] = (
        traffic_crashes_hourly["hour"]
        .apply(time_period_mapper)
    )

    segment_risk = (
        traffic_crashes_hourly
        .groupby("segment_id")["accident"]
        .mean()
        .reset_index(name="segment_risk")
    )

    traffic_crashes_hourly = traffic_crashes_hourly.merge(
        segment_risk,
        on="segment_id",
        how="left"
    )

    traffic_crashes_hourly["speed_ratio"] = (
            traffic_crashes_hourly["avg_speed"]
            / traffic_crashes_hourly["avg_free_flow_speed"]
    )

    traffic_crashes_hourly["congestion_level"] = (
        traffic_crashes_hourly["avg_congestion"]
        .apply(congestion_level_mapper)
    )

    return traffic_crashes_hourly
