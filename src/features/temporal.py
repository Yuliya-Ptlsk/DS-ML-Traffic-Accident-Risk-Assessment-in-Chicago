import pandas as pd


def calculate_free_flow_speed(
    df_traffic,
    percentile=0.95,
):
    free_flow_speed = (
        df_traffic.groupby('segment_id')['speed']
        .quantile(percentile)
        .reset_index()
    )
    free_flow_speed.columns = ['segment_id', 'free_flow_speed']

    df_traffic = df_traffic.merge(free_flow_speed, on='segment_id', how='left')

    return df_traffic


def calculate_congestion(
    df_traffic
):
    # calculate congestion with formula: 1 - speed/free_flow_speed
    df_traffic['congestion'] = (
            1 - df_traffic['speed'] / df_traffic['free_flow_speed']
    )

    # normalize congestion value
    df_traffic["congestion"] = (
        df_traffic["congestion"]
        .clip(0, 1)
    )

    return df_traffic


def create_traffic_hourly(df_traffic):
    # aggregate traffic hourly
    traffic_hourly = (
        df_traffic
        .groupby(
            [
                'segment_id',
                'hour',
                'day_of_week',
                'month',
            ]
        )
        .aggregate(
            avg_speed=('speed', 'mean'),
            avg_free_flow_speed=('free_flow_speed', 'mean'),
            avg_congestion=('congestion', 'mean'),
            start_latitude=('start_latitude', 'first'),
            start_longitude=('start_longitude', 'first'),
            end_latitude=('end_latitude', 'first'),
            end_longitude=('end_longitude', 'first')
        )
        .reset_index()
    )

    traffic_hourly['segment_lat'] = (traffic_hourly['start_latitude'] + traffic_hourly['end_latitude']) / 2

    traffic_hourly['segment_lon'] = (traffic_hourly['start_longitude'] + traffic_hourly['end_longitude']) / 2

    return traffic_hourly


def create_full_traffic_hourly(traffic_hourly):
    segments = traffic_hourly["segment_id"].unique()

    full_grid = pd.MultiIndex.from_product(
        [
            segments,
            range(24),  # hour
            range(1, 8),  # day_of_week
            range(1, 13),  # month
        ],
        names=[
            "segment_id",
            "hour",
            "day_of_week",
            "month"
        ]
    ).to_frame(index=False)

    traffic_hourly_full = full_grid.merge(
        traffic_hourly,
        on=[
            "segment_id",
            "hour",
            "day_of_week",
            "month"
        ],
        how="left"
    )

    # fill missing coordinates
    segment_geo = (
        traffic_hourly
        .groupby("segment_id")
        .agg(
            start_latitude=("start_latitude", "first"),
            start_longitude=("start_longitude", "first"),
            end_latitude=("end_latitude", "first"),
            end_longitude=("end_longitude", "first"),
            segment_lat=("segment_lat", "first"),
            segment_lon=("segment_lon", "first")
        )
        .reset_index()
    )

    traffic_hourly_full = traffic_hourly_full.drop(
        columns=[
            "start_latitude",
            "start_longitude",
            "end_latitude",
            "end_longitude",
            "segment_lat",
            "segment_lon"
        ],
        errors="ignore"
    )

    traffic_hourly_full = traffic_hourly_full.merge(
        segment_geo,
        on="segment_id",
        how="left"
    )

    return traffic_hourly_full


def fill_empty_features(
    traffic_hourly,
    traffic_hourly_full
):
    traffic_hourly_full["is_weekend"] = (
        traffic_hourly_full["day_of_week"] >= 6
    )

    for col in [
        "avg_speed",
        "avg_free_flow_speed",
        "avg_congestion"
    ]:
        # ==================================================
        # LEVEL 1
        # segment_id + hour + weekday/weekend
        # ==================================================

        fill_lvl1 = (
            traffic_hourly
            .assign(
                is_weekend=lambda x:
                x["day_of_week"] >= 6
            )
            .groupby(
                [
                    "segment_id",
                    "hour",
                    "is_weekend"
                ]
            )[col]
            .mean()
            .reset_index()
        )

        traffic_hourly_full = traffic_hourly_full.merge(
            fill_lvl1,
            on=[
                "segment_id",
                "hour",
                "is_weekend"
            ],
            how="left",
            suffixes=("", "_lvl1")
        )

        traffic_hourly_full[col] = (
            traffic_hourly_full[col]
            .fillna(
                traffic_hourly_full[f"{col}_lvl1"]
            )
        )

        traffic_hourly_full.drop(
            columns=[f"{col}_lvl1"],
            inplace=True
        )

        # ==================================================
        # LEVEL 2
        # hour + weekday/weekend
        # ==================================================

        fill_lvl2 = (
            traffic_hourly
            .assign(
                is_weekend=lambda x:
                x["day_of_week"] >= 6
            )
            .groupby(
                [
                    "hour",
                    "is_weekend"
                ]
            )[col]
            .mean()
            .reset_index()
        )

        traffic_hourly_full = traffic_hourly_full.merge(
            fill_lvl2,
            on=[
                "hour",
                "is_weekend"
            ],
            how="left",
            suffixes=("", "_lvl2")
        )

        traffic_hourly_full[col] = (
            traffic_hourly_full[col]
            .fillna(
                traffic_hourly_full[f"{col}_lvl2"]
            )
        )

        traffic_hourly_full.drop(
            columns=[f"{col}_lvl2"],
            inplace=True
        )

        # ==================================================
        # LEVEL 3
        # GLOBAL MEAN
        # ==================================================

        traffic_hourly_full[col] = (
            traffic_hourly_full[col]
            .fillna(
                traffic_hourly[col].mean()
            )
        )

    traffic_hourly_full['avg_speed'] = traffic_hourly_full['avg_speed'].round().astype(int)
    traffic_hourly_full['avg_free_flow_speed'] = traffic_hourly_full['avg_free_flow_speed'].round().astype(int)
    traffic_hourly_full['avg_congestion'] = traffic_hourly_full['avg_congestion'].round(4)

    return traffic_hourly_full
