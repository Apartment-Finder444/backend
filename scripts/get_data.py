import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import py7zr


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
KAGGLE_DATASET = "adithyaawati/apartments-for-rent-classified"
UCI_URL = "https://archive.ics.uci.edu/static/public/555/apartment+for+rent+classified.zip"
TARGET_NAME = "apartments_for_rent_classified_100K.csv"
REQUIRED_COLUMNS = {"price_type", "square_feet", "cityname", "pets_allowed"}


def header_columns(path):
    with open(path, encoding="cp1252", errors="replace") as handle:
        return {column.strip() for column in handle.readline().strip().split(";")}


def find_full_file(folder):
    matches = [path for path in Path(folder).rglob(TARGET_NAME) if path.is_file()]
    return matches[0] if matches else None


def place(source, destination):
    target_folder = destination / "apartments_for_rent_classified_100K"
    target_folder.mkdir(parents=True, exist_ok=True)
    target = target_folder / TARGET_NAME
    if Path(source).resolve() != target.resolve():
        shutil.copyfile(source, target)
    missing = REQUIRED_COLUMNS - header_columns(target)
    if missing:
        target.unlink()
        raise RuntimeError(f"Downloaded file is the wrong dataset; missing columns {sorted(missing)}")
    return target


def try_kaggle(destination):
    if shutil.which("kaggle") is None:
        print("Kaggle command line tool not installed; falling back to UCI")
        return None
    folder = destination / "kaggle_download"
    folder.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(folder), "--unzip"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"Kaggle download failed ({completed.stderr.strip() or completed.stdout.strip()}); falling back to UCI")
        return None
    found = find_full_file(folder)
    if found is None or not REQUIRED_COLUMNS.issubset(header_columns(found)):
        print("Kaggle mirror does not contain the expected 100K file with the expected columns; falling back to UCI")
        return None
    return found


def try_uci(destination):
    archive = destination / "uci_555_apartment_for_rent_classified.zip"
    if not archive.exists():
        print(f"Downloading {UCI_URL}")
        urllib.request.urlretrieve(UCI_URL, archive)
    extracted = destination / "uci_555_archive"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    found = find_full_file(extracted)
    if found is None:
        inner = next(extracted.rglob("*100K*.7z"), None)
        if inner is None:
            raise RuntimeError("UCI archive does not contain the 100K file")
        with py7zr.SevenZipFile(inner) as seven_zip:
            seven_zip.extractall(extracted / "100K")
        found = find_full_file(extracted)
    if found is None:
        raise RuntimeError("Could not find the 100K CSV after extracting the UCI archive")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(RAW_DIR))
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    destination = Path(arguments.dest)
    destination.mkdir(parents=True, exist_ok=True)
    existing = destination / "apartments_for_rent_classified_100K" / TARGET_NAME
    if existing.exists() and not arguments.force:
        print(f"Already present: {existing}")
        return
    source = try_kaggle(destination) or try_uci(destination)
    target = place(source, destination)
    print(f"Raw data ready at {target}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Data download failed: {error}", file=sys.stderr)
        sys.exit(1)
