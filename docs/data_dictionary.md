# Data dictionary and cleaning record

**Source.** Apartment for Rent Classified, UCI Machine Learning Repository dataset 555, mirrored on Kaggle as `adithyaawati/apartments-for-rent-classified`.

**Raw file.** We use the 100,000-row file, which has 22 attributes and is separated by semicolons. The listings are from 2018 and 2019.

## Cleaning steps

The steps run in `scripts/clean_data.py`, and `scripts/verify_clean.py` checks the results.

1. **Read the raw file safely.** It is read with semicolon separators, cp1252 encoding and quoting turned off. A standard parser silently drops several hundred rows because of stray quote characters, so the script fails unless all 100,000 rows arrive.
2. **Convert missing-value text.** The text `null` is turned into a real missing value in every column.
3. **Keep "no pets" distinct from missing.** In `pets_allowed`, `None` means no pets allowed, so it is renamed `none_allowed` and never read as missing.
4. **Convert dates.** `time` (a Unix timestamp) becomes a date.
5. **Clean addresses.** Stray leading quotes and leftover HTML are stripped from `address`.
6. **Blank placeholder coordinates.** Coordinates sitting at the geographic centre of the US are placeholders. They are blanked and flagged.
7. **Convert weekly rents.** Weekly rents become monthly at 52/12 weeks per month, and the row is marked `price_converted_from_weekly`.
8. **Remove rows**, for the reasons below only. Every removed row goes to `data/clean/rejected_rows.csv` with a `reject_reason`.

| Reason | Rows | Rule |
|---|---|---|
| `body_price_contradiction` | 1,597 | The rent quoted in the listing text differs from the listed price by more than a factor of ten |
| `repost_duplicate` | 1,340 | The same unit posted more than once: identical price, bedrooms, bathrooms, square feet and coordinates, plus an identical title or identical text |
| `exact_duplicate` | 84 | Every column identical |
| `implausible_price` | 19 | Monthly rent below $200 or above $20,000 after conversion, following the proposal's data-entry-error rule |
| `missing_price` | 1 | No price |
| `ambiguous_price_period` | 1 | Price period given as "Monthly\|Weekly", so it cannot be converted reliably |

**Result:** 100,000 raw rows = 96,958 kept + 3,042 rejected.

**Square footage.** The proposal planned to check whether missing square footage is missing at random before choosing between imputation and deletion. Square footage turned out to have no missing values, so neither was needed.

**Suspicious rows are kept and flagged rather than removed:**

| Flag | Rows | Meaning |
|---|---|---|
| `flag_bedroom_conflict` | 2,648 | The title says studio but the bedroom count is one or more |
| `flag_placeholder_coords` | 219 | The coordinates were a placeholder and have been blanked |
| `flag_suspect_size` | 55 | Square footage is implausible for the bedroom count |
| `flag_suspect_price` | 47 | Rent is unusually low or high for the bedroom count |
| `flag_coord_mismatch` | 7 | The coordinates fall far outside the stated city or state |

## Files

| File | Contents |
|---|---|
| `data/clean/apartments_clean.csv` | All kept rows and columns, including listing text (about 100 MB) |
| `data/clean/apartments_slim.csv` | Kept rows, analysis columns only (about 15 MB) |
| `data/clean/rejected_rows.csv` | Removed rows plus `reject_reason` |
| `data/clean/fair_rent_estimates.csv` | Out-of-fold fair rent estimate and 80% range for every usable Midwest listing |
| `data/clean/apartments.db` | SQLite database of Midwest listings used by the app |

## Columns in the cleaned data

| Column | Type | Description | Missing (Midwest) |
|---|---|---|---|
| `id` | integer | Listing identifier, unique after cleaning | 0% |
| `category` | text | Listing category, e.g. `housing/rent/apartment` | 0% |
| `title` | text | Listing title | 0% |
| `body` | text | Listing text (full file only) | 0% |
| `amenities` | text | Comma-separated amenity list | 18.9% |
| `bathrooms` | number | Bathrooms; halves allowed | 0.1% |
| `bedrooms` | number | Bedrooms; 0 means studio | 0.1% |
| `currency` | text | Always USD | 0% |
| `fee` | text | Whether a fee is mentioned | 0% |
| `has_photo` | text | `Yes`, `Thumbnail` or `No` | 0% |
| `pets_allowed` | text | `Cats`, `Dogs`, `Cats,Dogs` or `none_allowed`. Blank means not stated, which is not the same as no. | 50.6% |
| `price` | number | Monthly rent in dollars | 0% |
| `price_display` | text | Price as shown in the original listing | 0% |
| `price_type` | text | `Monthly` for every kept row | 0% |
| `price_converted_from_weekly` | boolean | True if `price` was converted from a weekly figure | 0% |
| `square_feet` | number | Floor area | 0% |
| `address` | text | Street address, cleaned (full file only) | mostly missing |
| `cityname` | text | City | under 1% |
| `state` | text | Two-letter state | 0% |
| `latitude`, `longitude` | number | Area-level location, roughly 2 miles between distinct points | under 1% |
| `source` | text | Listing website | 0% |
| `time` | date | Date posted, 2018 to 2019 | 0% |
| `flag_*` | boolean | Data-quality flags listed above | 0% |

## Engineered columns

These are created in `src/model.py`, and `src/load_db.py` adds them to the database.

| Column | Description |
|---|---|
| `distance_from_campus_miles` | Haversine distance from campus at 41.7959, −88.0855 (Lisle, IL) |
| `price_per_sqft` | Rent divided by square feet, used for comparison and not as a predictor |
| `cost_per_bedroom` | Rent divided by bedrooms, with a studio counted as one; used in the match score and not as a predictor |
| `amenity_count` | Number of listed amenities; blank when there is no amenity list |
| `amenity_*` | One indicator per amenity listed in at least 5% of listings |
| `pets_cats`, `pets_dogs`, `pets_none_allowed`, `pets_unknown` | Pet policy indicators |
| `is_chicago_metro` | Illinois listing in one of the Chicago metro cities named in `src/geography.py` |
| `fair_rent`, `fair_rent_low`, `fair_rent_high` | Random forest estimate and its 80% range |
| `value_label` | `below`, `within` or `above` the range, or `unknown` if there is no estimate |
