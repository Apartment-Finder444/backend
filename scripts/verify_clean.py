import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "apartments_for_rent_classified_100K" / "apartments_for_rent_classified_100K.csv"
CLEAN_FILE = ROOT / "data" / "clean" / "apartments_clean.csv"
SLIM_FILE = ROOT / "data" / "clean" / "apartments_slim.csv"
REJECTED_FILE = ROOT / "data" / "clean" / "rejected_rows.csv"
REPORT_FILE = ROOT / "reports" / "verification.txt"
FLAG_COLUMNS = [
    "flag_bedroom_conflict",
    "flag_placeholder_coords",
    "flag_coord_mismatch",
    "flag_suspect_price",
    "flag_suspect_size",
]
SLIM_COLUMNS = [
    "id", "price", "bedrooms", "bathrooms", "square_feet", "cityname", "state",
    "latitude", "longitude", "pets_allowed", "amenities", "has_photo", "source", *FLAG_COLUMNS,
]
PETS_VALUES = {"Cats", "Dogs", "Cats,Dogs", "Cats,Dogs,None", "none_allowed"}
ALLOWED_REASONS = {
    "exact_duplicate", "repost_duplicate", "ambiguous_price_period", "missing_price",
    "body_price_contradiction", "implausible_price",
}
UNIT_COLUMNS = ["price", "bedrooms", "bathrooms", "square_feet", "latitude", "longitude"]
US_BOUNDS = [(24.3, 49.4, -125.0, -66.5), (51.0, 72.0, -180.0, -129.0), (18.8, 22.3, -160.5, -154.7)]


def main():
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        RAW_FILE,
        sep=";",
        encoding="cp1252",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    cleaned = pd.read_csv(CLEAN_FILE, low_memory=False)
    slim = pd.read_csv(SLIM_FILE, low_memory=False)
    rejected = pd.read_csv(REJECTED_FILE, low_memory=False)
    checks = []

    def check(name, passed, detail=""):
        checks.append((name, bool(passed), detail))

    check("Raw dataset has exactly 100,000 rows", len(raw) == 100_000, f"{len(raw):,} rows")
    check(
        "Every raw row is kept or rejected",
        len(cleaned) + len(rejected) == len(raw),
        f"{len(cleaned):,} kept + {len(rejected):,} rejected = {len(cleaned) + len(rejected):,}",
    )
    check("No duplicate IDs remain", not cleaned["id"].duplicated().any())
    unit_complete = cleaned[UNIT_COLUMNS].notna().all(axis=1)
    reposts = int((unit_complete & (cleaned.duplicated(UNIT_COLUMNS + ["title"]) | cleaned.duplicated(UNIT_COLUMNS + ["body"]))).sum())
    check("No reposted listings of the same unit remain", reposts == 0, f"{reposts:,} found")
    reasons = set(rejected["reject_reason"].dropna().unique())
    check("Every rejected row has an allowed reason", reasons.issubset(ALLOWED_REASONS) and rejected["reject_reason"].notna().all(), str(rejected["reject_reason"].value_counts().to_dict()))
    check("Rents are between $200 and $20,000", cleaned["price"].between(200, 20_000).all(), f"min {cleaned['price'].min():,.0f}, max {cleaned['price'].max():,.0f}")
    check(
        "Prices are positive and monthly",
        cleaned["price"].gt(0).all() and cleaned["price_type"].astype(str).str.casefold().eq("monthly").all(),
        f"price types: {cleaned['price_type'].value_counts(dropna=False).to_dict()}",
    )
    bedrooms = pd.to_numeric(cleaned["bedrooms"], errors="coerce")
    bathrooms = pd.to_numeric(cleaned["bathrooms"], errors="coerce")
    check("Bedrooms and bathrooms are within sane bounds", bedrooms.dropna().between(0, 10).all() and bathrooms.dropna().between(0, 10).all())

    latitude = pd.to_numeric(cleaned["latitude"], errors="coerce")
    longitude = pd.to_numeric(cleaned["longitude"], errors="coerce")
    coordinates_blank = latitude.isna() & longitude.isna()
    inside_us = pd.Series(False, index=cleaned.index)
    for minimum_lat, maximum_lat, minimum_lon, maximum_lon in US_BOUNDS:
        inside_us |= latitude.between(minimum_lat, maximum_lat) & longitude.between(minimum_lon, maximum_lon)
    coordinates_valid = inside_us & latitude.ne(0) & longitude.ne(0)
    outside_count = int((~(coordinates_blank | coordinates_valid)).sum())
    check(
        "Coordinates are inside US bounds (contiguous, Alaska, Hawaii) or blank, and never zero",
        outside_count == 0,
        f"{outside_count:,} rows outside",
    )
    has_null_text = cleaned.astype("string").apply(
        lambda column: column.str.fullmatch(r"\s*null\s*", case=False, na=False).any()
    ).any()
    check("No string 'null' survives", not has_null_text)
    pet_values = set(cleaned["pets_allowed"].dropna().astype(str).unique())
    check("Pets column contains only expected values", pet_values.issubset(PETS_VALUES), f"observed: {sorted(pet_values)}")

    numeric_columns = ["id", "price", "bedrooms", "bathrooms", "square_feet", "latitude", "longitude"]
    numeric_types_ok = all(pd.api.types.is_numeric_dtype(cleaned[column]) for column in numeric_columns)
    flag_types_ok = all(pd.api.types.is_bool_dtype(cleaned[column]) for column in FLAG_COLUMNS)
    check("Numeric and flag column types are correct", numeric_types_ok and flag_types_ok)

    matching_rows = len(slim) == len(cleaned)
    sample_matches = False
    if matching_rows and set(SLIM_COLUMNS).issubset(cleaned.columns):
        sample_indexes = cleaned.sample(min(100, len(cleaned)), random_state=555).index
        full_sample = cleaned.loc[sample_indexes, SLIM_COLUMNS].reset_index(drop=True)
        slim_sample = slim.loc[sample_indexes, SLIM_COLUMNS].reset_index(drop=True)
        sample_matches = full_sample.equals(slim_sample)
    check("Slim file row count matches full file", matching_rows, f"{len(slim):,} slim / {len(cleaned):,} full")
    check("Slim values match full file on random sample", sample_matches)

    flag_counts = {column: int(cleaned[column].fillna(False).astype(bool).sum()) for column in FLAG_COLUMNS}
    check("Flag counts are reported and at least one flag is present", sum(flag_counts.values()) > 0, str(flag_counts))

    lines = [
        f"{'PASS' if passed else 'FAIL'}: {name}{f' ({detail})' if detail else ''}"
        for name, passed, detail in checks
    ]
    lines.append("Flag counts:")
    lines.extend(f"  {column}: {count:,}" for column, count in flag_counts.items())
    report = "\n".join(lines) + "\n"
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")
    if not all(passed for _, passed, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()