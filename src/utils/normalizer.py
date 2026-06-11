import re
import pandas as pd


def normalize_street(name):
    if pd.isna(name):
        return ""

    name = str(name).upper()

    # remove direction
    name = re.sub(r"\b(N|S|E|W)\b", "", name)

    # remove type of roads
    name = re.sub(
        r"\b(STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR|BOULEVARD|BLVD|PLACE|PL)\b",
        "",
        name
    )

    # remove white spaces
    name = re.sub(r"\s+", " ", name)

    return name.strip()

def street_matches(crash_street, traffic_streets):
    if not crash_street:
        return False

    for traffic_street in traffic_streets:
        if traffic_street in crash_street:
            return True

    return False