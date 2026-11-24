from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.geography import CAMPUS_LATITUDE, CAMPUS_LONGITUDE, EARTH_RADIUS_MILES, campus_distances


ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = ROOT / "data" / "clean" / "apartments_clean.csv"
REPORT_FILE = ROOT / "reports" / "exploration.txt"
CAMPUS = (41.7959, -88.0855)
CHICAGO_METRO_CITIES = {
    "aurora", "berwyn", "blue island", "bolingbrook", "burbank", "calumet city", "chicago",
    "cicero", "des plaines", "downers grove", "elgin", "elmhurst", "evanston", "glenview",
    "hanover park", "highland park", "hoffman estates", "joliet", "lisle", "mchenry", "naperville",
    "oak lawn", "oak park", "orland park", "palatine", "plainfield", "romeoville", " schaumburg",
    "skokie", "tinley park", "waukegan", "wheaton", "woodridge",
}
ILLINOIS_STATES = {"IL"}
MIDWEST_STATES = {"IL", "IN", "IA", "MI", "MN", "MO", "OH", "WI", "ND", "SD", "NE", "KS"}
FLAG_COLUMNS = [
    "flag_bedroom_conflict",
    "flag_placeholder_coords",
    "flag_coord_mismatch",
    "flag_suspect_price",
    "flag_suspect_size",
]
USABLE_COLUMNS = ["price", "bedrooms", "bathrooms", "square_feet", "latitude", "longitude"]
SCOPE_CITIES = {
    "Chicago metro": CHICAGO_METRO_CITIES,
    "All of Illinois": ILLINOIS_STATES,
    "Midwest": MIDWEST_STATES,
    "National": None,
}


def main():
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CLEAN_FILE, low_memory=False)
    frame["state"] = frame["state"].fillna("").astype(str).str.upper()
    frame["city_normalized"] = frame["cityname"].fillna("").astype(str).str.casefold().str.strip()
    complete = frame[USABLE_COLUMNS].notna().all(axis=1) & frame["price"].gt(0)
    flagged = frame[FLAG_COLUMNS].fillna(False).astype(bool).any(axis=1)
    lines = [
        f"Campus: {CAMPUS[0]:.4f}, {CAMPUS[1]:.4f}",
        f"Usable rows require positive price and complete {', '.join(USABLE_COLUMNS)}.",
        "",
        "Scope counts:",
    ]
    for scope, members in SCOPE_CITIES.items():
        if scope == "Chicago metro":
            in_scope = frame["state"].eq("IL") & frame["city_normalized"].isin(
                {city.strip() for city in members}
            )
        elif members is None:
            in_scope = pd.Series(True, index=frame.index)
        else:
            in_scope = frame["state"].isin(members)
        lines.append(
            f"{scope}: including flagged {int((in_scope & complete).sum()):,}; "
            f"excluding flagged {int((in_scope & complete & ~flagged).sum()):,}"
        )

    missing_percent = frame.drop(columns=["city_normalized"]).isna().mean().mul(100).sort_values(ascending=False)
    lines.extend(["", "Missing data by column:"])
    lines.extend(f"{column}: {percentage:.2f}%" for column, percentage in missing_percent.items())
    for column in ("pets_allowed", "amenities"):
        missing_count = int(frame[column].isna().sum())
        lines.append(f"Rows lost by filtering on {column}: {missing_count:,} ({missing_count / len(frame):.2%})")

    distances = campus_distances(frame)
    frame["distance_from_campus_miles"] = distances
    lines.extend([
        "",
        f"Rows with campus distance: {int(distances.notna().sum()):,}",
        f"Median campus distance: {distances.median():.2f} miles",
    ])

    coordinate_rows = frame.loc[frame[["latitude", "longitude"]].notna().all(axis=1), ["latitude", "longitude"]]
    unique_coordinates = coordinate_rows.drop_duplicates()
    lines.append(f"Distinct coordinate pairs: {len(unique_coordinates):,} across {len(coordinate_rows):,} rows")
    if len(unique_coordinates) > 1:
        coordinates_radians = np.radians(unique_coordinates[["latitude", "longitude"]].to_numpy(dtype=float))
        neighbor_model = NearestNeighbors(n_neighbors=2, metric="haversine", algorithm="ball_tree")
        neighbor_model.fit(coordinates_radians)
        neighbor_distances, _ = neighbor_model.kneighbors(coordinates_radians)
        nearest_miles = neighbor_distances[:, 1] * EARTH_RADIUS_MILES
        lines.append(
            "Nearest distinct coordinate spacing: "
            f"median {float(np.median(nearest_miles)):.2f} miles, "
            f"mean {float(np.mean(nearest_miles)):.2f} miles"
        )
    else:
        lines.append("Nearest distinct coordinate spacing: unavailable")

    chicago_count = int(
        (frame["state"].eq("IL") & frame["city_normalized"].isin({city.strip() for city in CHICAGO_METRO_CITIES}) & complete & ~flagged).sum()
    )
    midwest_count = int((frame["state"].isin(MIDWEST_STATES) & complete & ~flagged).sum())
    if not 400 <= chicago_count <= 1_200:
        lines.append(f"WARNING: Chicago metro count {chicago_count:,} differs from the expected roughly 700.")
    if not 8_000 <= midwest_count <= 16_000:
        lines.append(f"WARNING: Midwest count {midwest_count:,} differs from the expected roughly 12,000.")
    if len(unique_coordinates) > 1:
        median_spacing = float(np.median(nearest_miles))
        if not 0.5 <= median_spacing <= 3.5:
            lines.append(f"WARNING: nearest coordinate spacing {median_spacing:.2f} miles differs from roughly 1.5.")

    report = "\n".join(lines) + "\n"
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()