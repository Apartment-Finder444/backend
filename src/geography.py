import math

import numpy as np
import pandas as pd


EARTH_RADIUS_MILES = 3958.7613
CAMPUS_LATITUDE = 41.7959
CAMPUS_LONGITUDE = -88.0855
US_BOUNDS = (24.3, 49.4, -125.0, -66.5)
MIDWEST_STATES = {"IL", "IN", "IA", "MI", "MN", "MO", "OH", "WI", "ND", "SD", "NE", "KS"}
CHICAGO_METRO_CITIES = {
    "aurora", "berwyn", "blue island", "bolingbrook", "burbank", "calumet city", "chicago",
    "cicero", "des plaines", "downers grove", "elgin", "elmhurst", "evanston", "glenview",
    "hanover park", "highland park", "hoffman estates", "joliet", "lisle", "mchenry", "naperville",
    "oak lawn", "oak park", "orland park", "palatine", "plainfield", "romeoville", "schaumburg",
    "skokie", "tinley park", "waukegan", "wheaton", "woodridge",
}
STATE_BOUNDS = {
    "AL": (30.1, 35.1, -88.5, -84.8), "AZ": (31.3, 37.1, -114.9, -109.0),
    "AR": (33.0, 36.5, -94.7, -89.6), "CA": (32.5, 42.1, -124.5, -114.1),
    "CO": (36.9, 41.1, -109.1, -102.0), "CT": (40.9, 42.1, -73.8, -71.7),
    "DE": (38.4, 39.9, -75.8, -75.0), "FL": (24.3, 31.1, -87.7, -80.0),
    "GA": (30.3, 35.1, -85.7, -80.7), "ID": (41.9, 49.1, -117.3, -111.0),
    "IL": (36.9, 42.6, -91.6, -87.0), "IN": (37.7, 41.9, -88.2, -84.7),
    "IA": (40.3, 43.6, -96.7, -90.1), "KS": (37.0, 40.1, -102.1, -94.5),
    "KY": (36.4, 39.3, -89.6, -81.9), "LA": (28.8, 33.1, -94.1, -89.0),
    "ME": (43.0, 47.6, -71.1, -66.8), "MD": (37.9, 39.8, -79.6, -75.0),
    "MA": (41.2, 42.9, -73.6, -69.8), "MI": (41.7, 48.3, -90.5, -82.1),
    "MN": (43.4, 49.5, -97.3, -89.4), "MS": (30.1, 35.1, -91.7, -88.0),
    "MO": (35.9, 40.7, -95.8, -89.0), "MT": (44.3, 49.1, -116.1, -104.0),
    "NE": (39.9, 43.1, -104.1, -95.3), "NV": (34.9, 42.1, -120.1, -114.0),
    "NH": (42.7, 45.4, -72.6, -70.6), "NJ": (38.9, 41.4, -75.6, -73.8),
    "NM": (31.3, 37.1, -109.1, -103.0), "NY": (40.4, 45.1, -79.8, -71.7),
    "NC": (33.8, 36.7, -84.4, -75.4), "ND": (45.9, 49.1, -104.1, -96.5),
    "OH": (38.4, 42.1, -84.9, -80.5), "OK": (33.6, 37.1, -103.1, -94.4),
    "OR": (41.9, 46.4, -124.7, -116.4), "PA": (39.7, 42.4, -80.6, -74.6),
    "RI": (41.1, 42.1, -71.9, -71.1), "SC": (32.0, 35.3, -83.4, -78.5),
    "SD": (42.4, 45.9, -104.1, -96.4), "TN": (34.9, 36.8, -90.4, -81.6),
    "TX": (25.8, 36.6, -106.7, -93.4), "UT": (36.9, 42.1, -114.1, -109.0),
    "VT": (42.7, 45.1, -73.5, -71.4), "VA": (36.5, 39.6, -83.7, -75.1),
    "WA": (45.5, 49.1, -124.9, -116.8), "WV": (37.1, 40.7, -82.7, -77.7),
    "WI": (42.4, 47.4, -92.9, -86.7), "WY": (40.9, 45.1, -111.1, -104.0),
    "AK": (51.0, 72.0, -180.0, -129.0), "HI": (18.8, 22.3, -160.5, -154.7),
}


def haversine_miles(latitude_a, longitude_a, latitude_b, longitude_b):
    lat_a, lon_a, lat_b, lon_b = map(
        math.radians, (float(latitude_a), float(longitude_a), float(latitude_b), float(longitude_b))
    )
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(min(1.0, value)))


def flag_coordinate_mismatches(frame):
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    state = frame["state"].fillna("").astype(str).str.strip().str.upper()
    mismatch = pd.Series(False, index=frame.index)
    valid = latitude.notna() & longitude.notna()
    for abbreviation, (minimum_lat, maximum_lat, minimum_lon, maximum_lon) in STATE_BOUNDS.items():
        state_rows = state.eq(abbreviation) & valid
        mismatch.loc[state_rows] = (
            latitude.loc[state_rows].lt(minimum_lat - 0.5)
            | latitude.loc[state_rows].gt(maximum_lat + 0.5)
            | longitude.loc[state_rows].lt(minimum_lon - 0.5)
            | longitude.loc[state_rows].gt(maximum_lon + 0.5)
        )
    city_keys = pd.DataFrame({"city": frame["cityname"].fillna("").astype(str).str.casefold(), "state": state})
    valid_rows = frame.loc[valid]
    for key, indexes in city_keys.loc[valid].groupby(["city", "state"]).groups.items():
        if not key[0] or len(indexes) < 3:
            continue
        center_latitude = latitude.loc[indexes].median()
        center_longitude = longitude.loc[indexes].median()
        distances = [
            haversine_miles(center_latitude, center_longitude, latitude.loc[index], longitude.loc[index])
            for index in indexes
        ]
        outliers = pd.Series(distances, index=indexes).gt(60)
        mismatch.loc[outliers.index[outliers]] = True
    return mismatch.astype(bool)


def campus_distances(frame):
    latitudes = pd.to_numeric(frame["latitude"], errors="coerce")
    longitudes = pd.to_numeric(frame["longitude"], errors="coerce")
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid = latitudes.notna() & longitudes.notna()
    result.loc[valid] = [
        haversine_miles(CAMPUS_LATITUDE, CAMPUS_LONGITUDE, latitude, longitude)
        for latitude, longitude in zip(latitudes.loc[valid], longitudes.loc[valid])
    ]
    return result

def chicago_metro_mask(frame):
    state = frame["state"].fillna("").astype(str).str.strip().str.upper()
    city = frame["cityname"].fillna("").astype(str).str.strip().str.casefold()
    return state.eq("IL") & city.isin(CHICAGO_METRO_CITIES)
