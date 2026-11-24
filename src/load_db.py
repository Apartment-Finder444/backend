import sqlite3
from pathlib import Path

import pandas as pd

from src.geography import MIDWEST_STATES, campus_distances


ROOT = Path(__file__).resolve().parents[1]
SLIM_FILE = ROOT / "data" / "clean" / "apartments_slim.csv"
ESTIMATES_FILE = ROOT / "data" / "clean" / "fair_rent_estimates.csv"
DATABASE_FILE = ROOT / "data" / "clean" / "apartments.db"
ESTIMATE_COLUMNS = ["fair_rent", "fair_rent_low", "fair_rent_high", "value_label"]


def attach_estimates(frame, estimates_file):
    estimates_file = Path(estimates_file)
    if not estimates_file.exists():
        for column in ESTIMATE_COLUMNS:
            frame[column] = None
        return frame, 0
    estimates = pd.read_csv(estimates_file)
    estimates = estimates.drop_duplicates("id")
    merged = frame.merge(estimates[["id", *ESTIMATE_COLUMNS]], on="id", how="left")
    return merged, int(merged["fair_rent"].notna().sum())


def load_database(input_file=SLIM_FILE, database_file=DATABASE_FILE, estimates_file=ESTIMATES_FILE):
    frame = pd.read_csv(input_file, low_memory=False)
    frame["state"] = frame["state"].fillna("").astype(str).str.upper()
    frame = frame.loc[frame["state"].isin(MIDWEST_STATES)].copy()
    frame["distance_from_campus_miles"] = campus_distances(frame)
    frame, estimated = attach_estimates(frame, estimates_file)
    database_file = Path(database_file)
    database_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_file) as connection:
        connection.execute("DROP TABLE IF EXISTS apartments")
        frame.to_sql("apartments", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX idx_apartments_price ON apartments (price)")
        connection.execute("CREATE INDEX idx_apartments_bedrooms ON apartments (bedrooms)")
        connection.execute("CREATE UNIQUE INDEX idx_apartments_id ON apartments (id)")
        connection.commit()
    return len(frame), estimated


def main():
    count, estimated = load_database()
    print(f"Loaded {count:,} Midwest listings into {DATABASE_FILE}")
    if estimated:
        print(f"Fair rent estimates attached to {estimated:,} listings")
    else:
        print("No fair rent estimates found; run python -m src.model first to add them")


if __name__ == "__main__":
    main()
