import geopandas as gpd
from shapely.geometry import Point, LineString
from src.utils.normalizer import normalize_street, street_matches


def create_road_segments(df_traffic):
    # create df with unique road segments id
    road_segments = df_traffic[
        [
            'segment_id',
            'street',
            'start_latitude',
            'start_longitude',
            'end_latitude',
            'end_longitude'
        ]
    ].drop_duplicates(subset='segment_id')

    # add normalized street name to each road segment
    road_segments["street_norm"] = (
        road_segments["street"]
        .fillna("")
        .apply(normalize_street)
    )

    return road_segments


def filter_crashes_by_street(
    df_crashes,
    road_segments,
):
    # create streets set
    traffic_streets = set(road_segments["street_norm"])

    # add normalized street name to each crash point
    df_crashes["street_norm"] = (
        df_crashes["STREET_NAME"]
        .fillna("")
        .apply(normalize_street)
    )

    # filter dataset with crashes by streets existing in traffic dataset
    mask = (
        df_crashes["street_norm"]
        .apply(
            lambda street: street_matches(street, traffic_streets)
        )
    )
    df_crashes_filtered = df_crashes[mask]

    return df_crashes_filtered


def create_gdf_road_segments(road_segments):
    #  create GIS LineString for road segments
    road_segments["geometry"] = road_segments.apply(
        lambda row: LineString([
            (row['start_longitude'], row['start_latitude']),
            (row['end_longitude'], row['end_latitude']),
        ]),
        axis=1
    )

    # convert road segments df into GEO df
    gdf_road_segments = gpd.GeoDataFrame(
        road_segments,
        geometry="geometry",
        crs="EPSG:4326"
        # code for the WGS84 geographic coordinate reference system (CRS) which uses latitude and longitude in decimal degrees
    )
    # convert to EPSG:26916 (projected coordinate system) to find distance in meters
    gdf_road_segments = gdf_road_segments.to_crs("EPSG:26916")

    return gdf_road_segments


# For mapping crash point with road segment_id:
# - add geometry object to crash and traffic dataframes
# - calculate nearest distance between crash point and road segment id via geo coordinates
def assign_segments_to_crashes(
    df_crashes_filtered,
    gdf_road_segments,
    max_distance=100,
):
    # create GIS Point for crashes df
    df_crashes_filtered['geometry'] = df_crashes_filtered.apply(
        lambda row: Point(
            row['LONGITUDE'],
            row['LATITUDE']
        ),
        axis=1
    )

    # convert crashes df into GEO df
    gdf_crashes = gpd.GeoDataFrame(
        df_crashes_filtered,
        geometry="geometry",
        crs="EPSG:4326"
        # code for the WGS84 geographic coordinate reference system (CRS) which uses latitude and longitude in decimal degrees
    )
    # convert to EPSG:26916 (projected coordinate system) to find distance in meters
    gdf_crashes = gdf_crashes.to_crs("EPSG:26916")

    # assign segment_id to each crash point by nearest distance
    crashes_with_segments = gpd.sjoin_nearest(
        gdf_crashes,
        gdf_road_segments[['segment_id', 'geometry']],
        how="left",
        distance_col="distance",
    )

    # getting rid of emissions
    crashes_with_segments = crashes_with_segments[
        crashes_with_segments["distance"] <= max_distance
        ]

    return crashes_with_segments

