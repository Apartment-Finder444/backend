import csv
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.geography import flag_coordinate_mismatches, haversine_miles


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "apartments_for_rent_classified_100K" / "apartments_for_rent_classified_100K.csv"
CLEAN_DIR = ROOT / "data" / "clean"
EXPECTED_ROWS = 100_000
FLAG_COLUMNS = [
    "flag_bedroom_conflict",
    "flag_placeholder_coords",
    "flag_coord_mismatch",
    "flag_suspect_price",
    "flag_suspect_size",
]
SLIM_COLUMNS = [
    "id",
    "price",
    "bedrooms",
    "bathrooms",
    "square_feet",
    "cityname",
    "state",
    "latitude",
    "longitude",
    "pets_allowed",
    "amenities",
    "has_photo",
    "source",
    *FLAG_COLUMNS,
]
US_CENTER = (39.8333, -98.5855)
PRICE_BOUNDS = {
    0: (250, 12_000),
    1: (300, 15_000),
    2: (400, 20_000),
    3: (500, 25_000),
    4: (600, 30_000),
}
SIZE_BOUNDS = {
    0: (120, 3_000),
    1: (180, 4_000),
    2: (300, 6_000),
    3: (450, 8_000),
    4: (600, 10_000),
}


def read_raw_data(path=RAW_FILE):
    frame = pd.read_csv(
        path,
        sep=";",
        encoding="cp1252",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    expected_columns = {"price_type", "square_feet", "cityname", "pets_allowed"}
    if not expected_columns.issubset(frame.columns):
        raise ValueError(f"Wrong dataset schema; required columns: {sorted(expected_columns)}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} raw rows, found {len(frame):,}")
    return frame


def numeric_column(frame, name):
    frame[name] = pd.to_numeric(frame[name], errors="coerce")


def body_price_contradicts(price, body):
    if pd.isna(price) or pd.isna(body):
        return False
    body_text = str(body)
    body_text = re.split(r"\bmore units available\s*:", body_text, maxsplit=1, flags=re.IGNORECASE)[0]
    patterns = (
        r"\b(?:monthly\s+)?rent\s*(?:is\s*|[:=]\s*)?\$\s*([\d,]+(?:\.\d{1,2})?)",
        r"\$\s*([\d,]+(?:\.\d{1,2})?)\s*(?:/\s*mo\b|per\s+month\b|monthly\b)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, body_text, flags=re.IGNORECASE):
            normalized_amount = match.group(1).replace(",", "")
            if not normalized_amount:
                continue
            context_before = body_text[max(0, match.start() - 45):match.start()].casefold()
            context = body_text[max(0, match.start() - 45):match.end() + 45].casefold()
            if pattern == patterns[1] and not re.search(r"\brent\b", context):
                continue
            if re.search(r"pet|cats?|dogs?|deposit|fee|trash|storage|parking|utility|commission|application|income|one.time|security|service", context_before):
                continue
            amount = float(normalized_amount)
            if amount > 0 and (amount > price * 10 or amount * 10 < price):
                return True
    return False


def mark_rejections(frame):
    rejected_reason = pd.Series(pd.NA, index=frame.index, dtype="string")

    def mark(mask, reason):
        active = rejected_reason.isna()
        rejected_reason.loc[active & mask.fillna(False)] = reason

    mark(frame.duplicated(keep="first"), "exact_duplicate")
    price_type = frame["price_type"].astype("string").str.casefold()
    mark(price_type.str.contains(r"week|weekly", na=False), "weekly_price")
    prices = pd.to_numeric(frame["price"], errors="coerce")
    mark(prices.isna(), "missing_price")
    contradicted = pd.Series(
        [body_price_contradicts(price, body) for price, body in zip(prices, frame["body"])],
        index=frame.index,
    )
    mark(contradicted, "body_price_contradiction")
    return rejected_reason


def clean_data(frame):
    frame = frame.copy()
    if "pets_allowed" not in frame.columns:
        raise ValueError("Wrong dataset schema; pets_allowed column is missing")
    frame["pets_allowed"] = frame["pets_allowed"].replace("None", "none_allowed")
    frame = frame.replace(to_replace=r"^\s*null\s*$", value=pd.NA, regex=True)

    rejection_reasons = mark_rejections(frame)
    rejected = frame.loc[rejection_reasons.notna()].copy()
    rejected["reject_reason"] = rejection_reasons.loc[rejection_reasons.notna()]
    kept = frame.loc[rejection_reasons.isna()].copy()

    for column in ("price", "bedrooms", "bathrooms", "square_feet", "latitude", "longitude", "id"):
        numeric_column(kept, column)

    if "time" in kept.columns:
        times = pd.to_numeric(kept["time"], errors="coerce")
        if times.notna().any():
            kept["time"] = pd.to_datetime(times, unit="s", errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            kept["time"] = pd.to_datetime(kept["time"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "address" in kept.columns:
        kept["address"] = kept["address"].map(
            lambda value: html.unescape(str(value)) if pd.notna(value) else value
        )
        kept["address"] = (
            kept["address"]
            .astype("string")
            .str.replace(r"<[^>]*>", " ", regex=True)
            .str.replace(r"^[\"']+", "", regex=True)
            .str.strip()
        )

    placeholder = np.isclose(kept["latitude"], US_CENTER[0], atol=0.05) & np.isclose(
        kept["longitude"], US_CENTER[1], atol=0.05
    )
    placeholder = pd.Series(placeholder, index=kept.index).fillna(False)
    kept["flag_placeholder_coords"] = placeholder.astype(bool)
    kept.loc[placeholder, ["latitude", "longitude"]] = np.nan
    kept["flag_coord_mismatch"] = flag_coordinate_mismatches(kept)

    title = kept["title"].fillna("").astype("string")
    kept["flag_bedroom_conflict"] = (
        title.str.contains(r"\bstudio\b", case=False, regex=True, na=False)
        & kept["bedrooms"].ge(1).fillna(False)
    ).astype(bool)

    bedroom_group = kept["bedrooms"].round().astype("Int64")
    price_low = bedroom_group.map({key: value[0] for key, value in PRICE_BOUNDS.items()})
    price_high = bedroom_group.map({key: value[1] for key, value in PRICE_BOUNDS.items()})
    size_low = bedroom_group.map({key: value[0] for key, value in SIZE_BOUNDS.items()})
    size_high = bedroom_group.map({key: value[1] for key, value in SIZE_BOUNDS.items()})
    kept["flag_suspect_price"] = (
        kept["price"].lt(price_low) | kept["price"].gt(price_high)
    ).fillna(False).astype(bool)
    kept["flag_suspect_size"] = (
        kept["square_feet"].lt(size_low) | kept["square_feet"].gt(size_high)
    ).fillna(False).astype(bool)
    return kept, rejected


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_raw_data()
    cleaned, rejected = clean_data(raw)
    if len(cleaned) + len(rejected) != len(raw):
        raise RuntimeError("Row accounting failed during cleaning")
    cleaned.to_csv(CLEAN_DIR / "apartments_clean.csv", index=False)
    cleaned[SLIM_COLUMNS].to_csv(CLEAN_DIR / "apartments_slim.csv", index=False)
    rejected.to_csv(CLEAN_DIR / "rejected_rows.csv", index=False)
    print(f"Raw rows: {len(raw):,}")
    print(f"Kept rows: {len(cleaned):,}")
    print(f"Rejected rows: {len(rejected):,}")
    print(rejected["reject_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()