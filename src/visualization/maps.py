import geopandas as gpd
import folium
from shapely import wkt, Point, LineString
from branca.colormap import LinearColormap


def create_crashes_segments_map(crashes, roads):
    crashes_sample = crashes.sample(1500, random_state=42)
    crashes_sample = crashes_sample.dropna(subset=['LONGITUDE', 'LATITUDE'])

    is_gis_crashes = "geometry" in crashes.columns

    if not is_gis_crashes:
        crashes_sample["geometry"] = crashes_sample.apply(
            lambda row: Point(
                row['LONGITUDE'],
                row['LATITUDE']
            ),
            axis=1
        )
    else:
        crashes_sample["geometry"] = (
            crashes_sample["geometry"]
            .apply(wkt.loads)
        )

    roads["geometry"] = (
        roads["geometry"]
        .apply(wkt.loads)
    )

    gdf_crashes = gpd.GeoDataFrame(
        crashes_sample,
        geometry="geometry",
        crs=f"{'EPSG:26916' if is_gis_crashes else 'EPSG:4326'}"
    )

    gdf_roads = gpd.GeoDataFrame(
        roads,
        geometry="geometry",
        crs="EPSG:26916"
    )

    # add segments
    segments_wgs = gdf_roads.to_crs("EPSG:4326")

    bounds = segments_wgs.total_bounds
    # [minx, miny, maxx, maxy]

    m = folium.Map(
        tiles="CartoDB Voyager",
    )

    # center the map
    m.fit_bounds([
        [bounds[1], bounds[0]],  # south-west
        [bounds[3], bounds[2]]   # north-east
    ])

    folium.GeoJson(
        segments_wgs,
        style_function=lambda feature: {
            "color": "#3236a8",
            "weight": 2,
            "opacity": 0.5,
        }
    ).add_to(m)

    # add crash points
    crashes = gdf_crashes.to_crs("EPSG:4326")

    for _, row in crashes.iterrows():
        folium.CircleMarker(
            location=[
                row.geometry.y,
                row.geometry.x
            ],
            radius=2,
            color="#d62728",
            weight=1,
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.5,
        ).add_to(m)

    return m

def create_risk_segments_map(crashes_hourly):
    crashes_hourly["geometry"] = crashes_hourly.apply(
        lambda row: LineString([
            (row['start_longitude'], row['start_latitude']),
            (row['end_longitude'], row['end_latitude']),
        ]),
        axis=1
    )

    segment_stats = (
        crashes_hourly.groupby("segment_id", as_index=False)
        .agg(
            segment_risk=("segment_risk", "max"),
            crash_count=("crash_count", "max"),
            geometry=("geometry", "first")
        )
    )

    roads = gpd.GeoDataFrame(
        segment_stats,
        geometry="geometry",
        crs="EPSG:4326"
    )

    # Create map
    bounds = roads.total_bounds
    # [minx, miny, maxx, maxy]

    m = folium.Map(
        tiles="CartoDB Positron"
    )

    m.fit_bounds([
        [bounds[1], bounds[0]],
        [bounds[3], bounds[2]]
    ])

    # Color scale
    colormap = LinearColormap(
        colors=[
            "#2ca25f",  # green
            "#fee08b",  # yellow
            "#f46d43",  # orange
            "#d73027"  # red
        ],
        vmin=roads["segment_risk"].min(),
        vmax=roads["segment_risk"].max()
    )

    colormap.caption = "Segment Risk"

   # Style function
    def style_function(feature):
        risk = feature["properties"]["segment_risk"]

        return {
            "color": colormap(risk),
            "weight": 4,
            "opacity": 0.8
        }

    # Tooltip
    tooltip = folium.GeoJsonTooltip(
        fields=[
            "segment_id",
            "segment_risk",
            "crash_count"
        ],
        aliases=[
            "Segment ID:",
            "Risk:",
            "Max Crash Count per Hour:"
        ],
        localize=True,
        sticky=False
    )

    # select top 50 roads with highest risk
    top_100_roads = (
        roads
        .sort_values(
            by="segment_risk",
            ascending=False
        )
        .drop_duplicates(
            subset=["segment_id"]
        )
        .head(100)
    )

    # Add roads
    folium.GeoJson(
        top_100_roads,
        style_function=style_function,
        tooltip=tooltip
    ).add_to(m)

    colormap.add_to(m)

    return m