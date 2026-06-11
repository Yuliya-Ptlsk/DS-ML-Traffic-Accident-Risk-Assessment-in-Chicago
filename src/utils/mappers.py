def time_period_mapper(hour):
    if 6 <= hour < 12:
        return "morning"

    elif 12 <= hour < 18:
        return "afternoon"

    elif 18 <= hour < 22:
        return "evening"

    return "night"

def congestion_level_mapper(x):

    if x < 0.2:
        return "low"

    elif x < 0.5:
        return "medium"

    return "high"

# unique:
# CLEAR
# CLOUDY
# RAIN
# SNOW
# FOG
# UNKNOWN
# OTHER
weather_mapping = {
    'CLEAR': 'CLEAR',
    'CLOUDY/OVERCAST': 'CLOUDY',
    'RAIN': 'RAIN',
    'FREEZING RAIN/DRIZZLE': 'RAIN',
    'SNOW': 'SNOW',
    'BLOWING SNOW': 'SNOW',
    'SLEET/HAIL': 'SNOW',
    'FOG/SMOKE/HAZE': 'FOG',
    'UNKNOWN': 'UNKNOWN',
    'OTHER': 'OTHER',
    'SEVERE CROSS WIND GATE': 'OTHER',
    'BLOWING SAND, SOIL, DIRT': 'OTHER'
}

# unique:
# DRY
# WET
# SNOW_ICE
# OTHER
# UNKNOWN
surface_mapping = {
    'DRY': 'DRY',
    'WET': 'WET',
    'SNOW OR SLUSH': 'SNOW_ICE',
    'ICE': 'SNOW_ICE',
    'SAND, MUD, DIRT': 'OTHER',
    'OTHER': 'OTHER',
    'UNKNOWN': 'UNKNOWN'
}

# unique:
# ROAD
# INTERSECTION
# PARKING
# ALLEY
# RAMP
# DRIVEWAY
# UNKNOWN
# OTHER
trafficway_mapping = {
    'NOT DIVIDED': 'ROAD',
    'DIVIDED - W/MEDIAN (NOT RAISED)': 'ROAD',
    'DIVIDED - W/MEDIAN BARRIER': 'ROAD',
    'ONE-WAY': 'ROAD',
    'FOUR WAY': 'INTERSECTION',
    'T-INTERSECTION': 'INTERSECTION',
    'Y-INTERSECTION': 'INTERSECTION',
    'L-INTERSECTION': 'INTERSECTION',
    'FIVE POINT, OR MORE': 'INTERSECTION',
    'ROUNDABOUT': 'INTERSECTION',
    'PARKING LOT': 'PARKING',
    'ALLEY': 'ALLEY',
    'RAMP': 'RAMP',
    'DRIVEWAY': 'DRIVEWAY',
    'CENTER TURN LANE': 'ROAD',
    'TRAFFIC ROUTE': 'ROAD',
    'UNKNOWN': 'UNKNOWN',
    'UNKNOWN INTERSECTION TYPE': 'UNKNOWN',
    'NOT REPORTED': 'UNKNOWN',
    'OTHER': 'OTHER'
}
