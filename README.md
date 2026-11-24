# Campus Rent Finder

A rental search tool for students looking near campus in Lisle, Illinois. It covers about 12,000 Midwest listings and is built around one idea: show how uncertain each number is instead of hiding it.

- Each listing gets a **fair rent range**, not a single price. The asking rent is marked as below, within or above that range.
- Each listing gets a **match score** against your preferences, plus a **confidence** figure that says how much of the score rests on facts the listing actually states.
- **Unknown fields are labelled as unknown.** About half of the Midwest listings (51%) do not mention pets. That means "not stated", not "no", and the app treats it that way.

The listings are from **2018 and 2019**. All rents are historical asking prices.

## What is in the repository

| Path | Purpose |
|---|---|
| `scripts/get_data.py` | Downloads the raw dataset (Kaggle if configured, otherwise the UCI archive) |
| `scripts/clean_data.py` | Cleans the 100,000 raw rows, writes kept rows, rejected rows and a slim file |
| `scripts/verify_clean.py` | Checks the cleaned files against the raw file and exits non-zero on any failure |
| `scripts/explore.py` | Scope counts, missing data and coordinate resolution |
| `src/model.py` | Rent models: linear regression and random forest, prediction intervals, feature importance |
| `notebooks/rent_model.ipynb` | Walk-through of the model results with plots |
| `src/scoring.py` | Pure match-score functions, with no database access |
| `src/load_db.py` | Builds the SQLite database of Midwest listings with distances and fair rent estimates |
| `src/app.py` | Flask API and web page |
| `src/templates/`, `src/static/` | Frontend in plain HTML, CSS and JavaScript |
| `tests/` | pytest suite |
| `reports/` | Verification, exploration and model outputs, plus `results_summary.md` |
| `docs/user_guide.md` | How the match score and fair rent range work, for people using the app |

## Setup

Requires Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (macOS or Linux)
pip install -r requirements.txt
```

## Getting the data

The dataset is "Apartment for Rent Classified": UCI Machine Learning Repository dataset 555, mirrored on Kaggle as `adithyaawati/apartments-for-rent-classified`.

```
python -m scripts.get_data
```

The script tries the Kaggle command line tool first. If that tool is missing, has no credentials, or the mirror no longer has the expected file, it downloads the archive from UCI: https://archive.ics.uci.edu/static/public/555/apartment+for+rent+classified.zip

It uses the 100,000-row file, not the 10,000-row subset. It also checks that the columns include `price_type`, `square_feet`, `cityname` and `pets_allowed`, and refuses anything else so a similarly named dataset never slips in. Raw files go to `data/raw/`, which git ignores.

## Running the pipeline

Run every command from the project root, in this order:

```
python -m scripts.clean_data      writes data/clean/
python -m scripts.verify_clean    writes reports/verification.txt, exits 1 on failure
python -m scripts.explore         writes reports/exploration.txt
python -m src.model               trains models (about a minute), writes models/ and reports/model_results.txt
python -m src.load_db             builds data/clean/apartments.db
```

`src.model` must run before `src.load_db`, because the loader attaches the fair rent estimates the model writes. Both steps are safe to rerun: each one replaces its outputs.

## Running the app

```
python -m src.app
```

Then open http://127.0.0.1:5000.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/search` | Matching listings, best match first |
| `GET /api/listing/<id>` | One listing in the same shape |
| `GET /api/feature-importance` | What drives rent across the dataset |
| `GET /api/options` | States, amenities and priority names for building the form |

Filters for `/api/search`:

- `max_rent`
- `min_bedrooms`, `max_bedrooms`
- `min_bathrooms`
- `city`, `state`
- `max_distance` (miles from campus)
- `pets` (`cats`, `dogs` or `cats,dogs`)
- `confirmed_only`
- `hide_flagged`
- `limit` (default 50, maximum 500)

Preferences that feed the match score:

- `budget` (defaults to `max_rent`)
- `bedrooms` (defaults to `min_bedrooms`)
- `bathrooms` (defaults to `min_bathrooms`)
- `min_square_feet`
- `per_person_budget`
- `amenities` (comma separated)
- `near_campus` (scores distance even without `max_distance`)
- `priorities`: a comma-separated ranking drawn from `rent`, `bedrooms`, `bathrooms`, `square_feet`, `distance`, `amenities`, `cost_per_bedroom`

Each listing in the response includes:

- `match_score` and `match_confidence` (both 0 to 1)
- `score_components`
- `fair_rent`, `fair_rent_low` and `fair_rent_high` (an 80% range)
- `value_label` (`below`, `within`, `above` or `unknown`)
- `unknown_fields`
- `pet_policy` and `pets_unconfirmed`
- `data_flags`

**Pets rule.** When you filter for pets:

- Listings with no stated policy stay in the results, marked `pets_unconfirmed`, and sort below confirmed matches.
- A listing is excluded only when it explicitly says no pets.
- `confirmed_only=true` drops the unconfirmed listings.

## Tests

```
python -m pytest
```

## Data handling notes

- Git ignores `data/raw/`, `data/clean/` and `models/*.joblib`. Anyone cloning the repository runs the pipeline to rebuild them.
- `models/model_metadata.json` is small and is committed, so the model results can be reviewed without retraining.
