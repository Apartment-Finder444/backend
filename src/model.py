import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from src.geography import MIDWEST_STATES, campus_distances, chicago_metro_mask


ROOT = Path(__file__).resolve().parents[1]
SLIM_FILE = ROOT / "data" / "clean" / "apartments_slim.csv"
MODEL_DIR = ROOT / "models"
LINEAR_MODEL_FILE = MODEL_DIR / "linear_model.joblib"
FOREST_MODEL_FILE = MODEL_DIR / "random_forest.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
ESTIMATES_FILE = ROOT / "data" / "clean" / "fair_rent_estimates.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
REPORT_FILE = REPORT_DIR / "model_results.txt"
RANDOM_STATE = 555
TEST_SHARE = 0.2
FOLDS = 5
INTERVAL_COVERAGE = 0.8
AMENITY_MIN_SHARE = 0.05
NEAR_CAMPUS_MILES = 25
PERMUTATION_REPEATS = 5
INTERVAL_BANDS = 5
FOREST_PARAMETERS = {
    "n_estimators": 200,
    "min_samples_leaf": 3,
    "max_features": 0.5,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}
FLAG_COLUMNS = [
    "flag_bedroom_conflict",
    "flag_placeholder_coords",
    "flag_coord_mismatch",
    "flag_suspect_price",
    "flag_suspect_size",
]
REQUIRED_COLUMNS = ["price", "bedrooms", "bathrooms", "square_feet"]
MODEL_KINDS = ("linear", "forest")
MODEL_NAMES = {"linear": "Multiple linear regression", "forest": "Random forest"}
SERIES_COLORS = {"linear": "#2a78d6", "forest": "#eb6834"}
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def parse_amenities(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "null", "none"}:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def amenity_column(name):
    return "amenity_" + "".join(character if character.isalnum() else "_" for character in name.casefold()).strip("_")


def prepare_frame(frame):
    frame = frame.copy()
    for column in ("id", "price", "bedrooms", "bathrooms", "square_feet", "latitude", "longitude"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["state"] = frame["state"].fillna("").astype(str).str.strip().str.upper()
    frame["is_chicago_metro"] = chicago_metro_mask(frame).astype(int)
    if "distance_from_campus_miles" not in frame.columns:
        frame["distance_from_campus_miles"] = campus_distances(frame)
    frame["amenity_list"] = frame["amenities"].map(parse_amenities) if "amenities" in frame.columns else None
    frame["amenity_count"] = frame["amenity_list"].map(lambda items: np.nan if items is None else float(len(items)))
    frame["price_per_sqft"] = frame["price"] / frame["square_feet"].where(frame["square_feet"] > 0)
    frame["cost_per_bedroom"] = frame["price"] / frame["bedrooms"].clip(lower=1)
    flags = [column for column in FLAG_COLUMNS if column in frame.columns]
    if flags:
        frame["flagged"] = frame[flags].apply(lambda column: column.astype(str).str.casefold().isin({"true", "1"})).any(axis=1)
    else:
        frame["flagged"] = False
    return frame


def usable_mask(frame):
    complete = frame[REQUIRED_COLUMNS].notna().all(axis=1)
    return complete & frame["price"].gt(0) & frame["square_feet"].gt(0)


def load_midwest(path=SLIM_FILE):
    frame = pd.read_csv(path, low_memory=False)
    frame["state"] = frame["state"].fillna("").astype(str).str.strip().str.upper()
    frame = frame.loc[frame["state"].isin(MIDWEST_STATES)]
    frame = prepare_frame(frame)
    return frame.loc[usable_mask(frame)].reset_index(drop=True)


def skew_report(frame):
    report = {}
    for column, transform in (("price", np.log), ("square_feet", np.log), ("distance_from_campus_miles", np.log1p)):
        values = frame[column].dropna()
        raw_skew = float(values.skew())
        log_skew = float(transform(values).skew())
        report[column] = {
            "raw_skew": raw_skew,
            "log_skew": log_skew,
            "use_log": abs(log_skew) < abs(raw_skew),
            "transform": "log" if transform is np.log else "log1p",
        }
    return report


def build_spec(train, skew):
    state_counts = train["state"].value_counts()
    states = sorted(state_counts.index)
    reference_state = str(state_counts.index[0])
    with_amenities = train["amenity_list"].dropna()
    amenity_share = with_amenities.explode().value_counts() / max(len(with_amenities), 1)
    amenities = sorted(amenity_share[amenity_share >= AMENITY_MIN_SHARE].index)
    distance_fill = train.groupby("state")["distance_from_campus_miles"].median().dropna().to_dict()
    spec = {
        "log_target": bool(skew["price"]["use_log"]),
        "log_square_feet": bool(skew["square_feet"]["use_log"]),
        "log_distance": bool(skew["distance_from_campus_miles"]["use_log"]),
        "states": states,
        "reference_state": reference_state,
        "amenities": amenities,
        "distance_fill": {key: float(value) for key, value in distance_fill.items()},
        "distance_fill_default": float(train["distance_from_campus_miles"].median()),
        "median_rent": float(train["price"].median()),
        "median_distance": float(train["distance_from_campus_miles"].median()),
    }
    spec["feature_names"] = list(feature_matrix(train.head(1), spec).columns)
    return spec


def square_feet_feature(spec):
    return "log_square_feet" if spec["log_square_feet"] else "square_feet"


def distance_feature(spec):
    return "log_distance_from_campus" if spec["log_distance"] else "distance_from_campus_miles"


def feature_matrix(frame, spec):
    features = pd.DataFrame(index=frame.index)
    features["bedrooms"] = frame["bedrooms"].astype(float)
    features["bathrooms"] = frame["bathrooms"].astype(float)
    square_feet = frame["square_feet"].astype(float)
    features[square_feet_feature(spec)] = np.log(square_feet) if spec["log_square_feet"] else square_feet
    fill = frame["state"].map(spec["distance_fill"]).fillna(spec["distance_fill_default"])
    distance = frame["distance_from_campus_miles"].astype(float).fillna(fill)
    features[distance_feature(spec)] = np.log1p(distance) if spec["log_distance"] else distance
    amenity_lists = frame["amenity_list"]
    features["amenities_unknown"] = amenity_lists.isna().astype(float)
    features["amenity_count"] = frame["amenity_count"].fillna(0.0).astype(float)
    for amenity in spec["amenities"]:
        features[amenity_column(amenity)] = amenity_lists.map(
            lambda items: 0.0 if items is None else float(amenity in items)
        )
    pets = frame["pets_allowed"].fillna("").astype(str).str.casefold() if "pets_allowed" in frame.columns else pd.Series("", index=frame.index)
    features["pets_unknown"] = pets.eq("").astype(float)
    features["pets_none_allowed"] = pets.eq("none_allowed").astype(float)
    features["pets_cats"] = pets.str.contains("cats").astype(float)
    features["pets_dogs"] = pets.str.contains("dogs").astype(float)
    features["is_chicago_metro"] = frame["is_chicago_metro"].astype(float)
    for state in spec["states"]:
        if state != spec["reference_state"]:
            features[f"state_{state}"] = frame["state"].eq(state).astype(float)
    return features


def feature_groups(spec):
    groups = {
        "Bedrooms": ["bedrooms"],
        "Bathrooms": ["bathrooms"],
        "Square footage": [square_feet_feature(spec)],
        "Distance from campus": [distance_feature(spec)],
        "Amenity count": ["amenity_count"],
        "Amenities not listed": ["amenities_unknown"],
        "Pet policy": ["pets_unknown", "pets_none_allowed", "pets_cats", "pets_dogs"],
        "Chicago metro": ["is_chicago_metro"],
        "State": [name for name in spec["feature_names"] if name.startswith("state_")],
    }
    for amenity in spec["amenities"]:
        groups[f"Amenity: {amenity}"] = [amenity_column(amenity)]
    return {name: columns for name, columns in groups.items() if columns}


def target_values(frame, spec):
    prices = frame["price"].astype(float)
    return np.log(prices) if spec["log_target"] else prices


def to_dollars(values, spec):
    values = np.asarray(values, dtype=float)
    return np.exp(values) if spec["log_target"] else values


def new_model(kind):
    if kind == "linear":
        return LinearRegression()
    if kind == "forest":
        return RandomForestRegressor(**FOREST_PARAMETERS)
    raise ValueError(f"Unknown model kind: {kind}")


def fit_model(kind, frame, spec):
    model = new_model(kind)
    model.fit(feature_matrix(frame, spec).to_numpy(), target_values(frame, spec).to_numpy())
    return model


def predict_target(model, frame, spec):
    return model.predict(feature_matrix(frame, spec).to_numpy())


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return {"rmse": None, "mae": None, "r2": None, "rows": 0}
    return {
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "rows": int(len(actual)),
    }


def cross_validate(kind, train, spec):
    folds = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    out_of_fold = np.full(len(train), np.nan)
    fold_metrics = []
    for fit_positions, holdout_positions in folds.split(train):
        fit_rows = train.iloc[fit_positions]
        holdout_rows = train.iloc[holdout_positions]
        model = fit_model(kind, fit_rows, spec)
        predictions = predict_target(model, holdout_rows, spec)
        out_of_fold[holdout_positions] = predictions
        fold_metrics.append(regression_metrics(holdout_rows["price"], to_dollars(predictions, spec)))
    summary = {
        key: {
            "mean": float(np.mean([metric[key] for metric in fold_metrics])),
            "std": float(np.std([metric[key] for metric in fold_metrics])),
        }
        for key in ("rmse", "mae", "r2")
    }
    return out_of_fold, summary


def interval_quantiles(residuals, predicted_target, coverage=INTERVAL_COVERAGE, bands=INTERVAL_BANDS):
    tail = (1 - coverage) / 2
    residuals = np.asarray(residuals, dtype=float)
    predicted_target = np.asarray(predicted_target, dtype=float)
    edges = [float(value) for value in np.quantile(predicted_target, np.linspace(0, 1, bands + 1)[1:-1])]
    positions = np.searchsorted(edges, predicted_target, side="right")
    limits = []
    for band in range(bands):
        band_residuals = residuals[positions == band]
        if len(band_residuals) < 30:
            band_residuals = residuals
        limits.append([float(np.quantile(band_residuals, tail)), float(np.quantile(band_residuals, 1 - tail))])
    return {"coverage": coverage, "edges": edges, "limits": limits}


def interval_bounds(predicted_target, interval, spec):
    predicted_target = np.asarray(predicted_target, dtype=float)
    positions = np.searchsorted(interval["edges"], predicted_target, side="right")
    limits = np.asarray(interval["limits"], dtype=float)[positions]
    return (
        to_dollars(predicted_target + limits[:, 0], spec),
        to_dollars(predicted_target + limits[:, 1], spec),
    )


def coverage_by_predicted_band(actual, predicted_target, interval, spec):
    low, high = interval_bounds(predicted_target, interval, spec)
    positions = np.searchsorted(interval["edges"], np.asarray(predicted_target, dtype=float), side="right")
    actual = np.asarray(actual, dtype=float)
    covered = (actual >= low) & (actual <= high)
    bounds = [-np.inf, *interval["edges"], np.inf]
    rows = []
    for band in range(len(interval["limits"])):
        members = positions == band
        lower = "under" if not np.isfinite(bounds[band]) else f"${float(to_dollars(bounds[band], spec)):,.0f}"
        upper = "and up" if not np.isfinite(bounds[band + 1]) else f"${float(to_dollars(bounds[band + 1], spec)):,.0f}"
        label = f"{lower} {upper}" if lower == "under" or upper == "and up" else f"{lower} to {upper}"
        rows.append({"band": label, "rows": int(members.sum()), "coverage": float(covered[members].mean()) if members.any() else None})
    return rows


def value_label(price, low, high):
    if price is None or low is None or high is None:
        return "unknown"
    if any(isinstance(value, float) and math.isnan(value) for value in (price, low, high)):
        return "unknown"
    if price < low:
        return "below"
    if price > high:
        return "above"
    return "within"


def grouped_permutation_importance(model, frame, spec, groups):
    features = feature_matrix(frame, spec)
    actual = frame["price"].to_numpy(dtype=float)
    base_rmse = regression_metrics(actual, to_dollars(model.predict(features.to_numpy()), spec))["rmse"]
    generator = np.random.default_rng(RANDOM_STATE)
    importance = {}
    for name, columns in groups.items():
        increases = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = features.copy()
            order = generator.permutation(len(shuffled))
            shuffled[columns] = features[columns].to_numpy()[order]
            rmse = regression_metrics(actual, to_dollars(model.predict(shuffled.to_numpy()), spec))["rmse"]
            increases.append(rmse - base_rmse)
        importance[name] = float(np.mean(increases))
    return importance


def linear_dollar_effects(model, spec):
    coefficients = dict(zip(spec["feature_names"], model.coef_))
    median_rent = spec["median_rent"]
    median_distance = spec["median_distance"]
    effects = []
    for name, coefficient in coefficients.items():
        if name == "log_square_feet":
            delta, change = math.log(1.1), "10% more square footage"
        elif name == "square_feet":
            delta, change = 100.0, "100 more square feet"
        elif name == "log_distance_from_campus":
            delta = math.log1p(median_distance + 10) - math.log1p(median_distance)
            change = f"10 more miles from campus, starting at the median {median_distance:.0f} miles"
        elif name == "distance_from_campus_miles":
            delta, change = 10.0, "10 more miles from campus"
        elif name == "bedrooms":
            delta, change = 1.0, "one more bedroom, same square footage"
        elif name == "bathrooms":
            delta, change = 1.0, "one more bathroom"
        elif name == "amenity_count":
            delta, change = 1.0, "one more listed amenity of any kind"
        elif name == "amenities_unknown":
            delta, change = 1.0, "no amenity list in the listing"
        elif name == "is_chicago_metro":
            delta, change = 1.0, "inside the Chicago metro rather than elsewhere in Illinois"
        elif name.startswith("state_"):
            delta, change = 1.0, f"in {name[6:]} rather than {spec['reference_state']}"
        elif name.startswith("pets_"):
            delta, change = 1.0, {
                "pets_unknown": "pet policy not stated",
                "pets_none_allowed": "explicitly no pets",
                "pets_cats": "cats allowed",
                "pets_dogs": "dogs allowed",
            }[name]
        elif name.startswith("amenity_"):
            amenity = next((item for item in spec["amenities"] if amenity_column(item) == name), name)
            delta, change = 1.0, f"{amenity} listed, which also adds one to the amenity count"
        else:
            delta, change = 1.0, f"one unit of {name}"
        total = coefficient * delta
        if name.startswith("amenity_") and name != "amenity_count":
            total += coefficients["amenity_count"]
        if spec["log_target"]:
            dollars = median_rent * (math.exp(total) - 1)
        else:
            dollars = total
        effects.append({"feature": name, "change": change, "coefficient": float(coefficient), "dollars": float(dollars)})
    return effects


def error_by_price(actual, predicted, low, high, bins=5):
    table = pd.DataFrame({"actual": actual, "predicted": predicted, "low": low, "high": high})
    table["band"] = pd.qcut(table["actual"], bins, duplicates="drop")
    table["absolute_error"] = (table["actual"] - table["predicted"]).abs()
    table["percent_error"] = table["absolute_error"] / table["actual"]
    table["covered"] = table["actual"].between(table["low"], table["high"])
    grouped = table.groupby("band", observed=True)
    summary = grouped.agg(
        rows=("actual", "size"),
        mae=("absolute_error", "mean"),
        median_percent_error=("percent_error", "median"),
        coverage=("covered", "mean"),
    ).reset_index()
    summary["band"] = summary["band"].map(lambda interval: f"${interval.left:,.0f} to ${interval.right:,.0f}")
    return summary.to_dict(orient="records")


def spearman(first, second):
    first = pd.Series(first).rank()
    second = pd.Series(second).rank()
    return float(np.corrcoef(first, second)[0, 1])


def evaluate_variant(frame, train_index, test_index, exclude_flagged, skew):
    train = frame.loc[train_index]
    test = frame.loc[test_index]
    if exclude_flagged:
        train = train.loc[~train["flagged"]]
        test = test.loc[~test["flagged"]]
    common_test = frame.loc[test_index]
    common_test = common_test.loc[~common_test["flagged"]]
    spec = build_spec(train, skew)
    groups = feature_groups(spec)
    baseline_value = float(train["price"].mean())
    subsets = {
        "held_out_all": test,
        "held_out_unflagged": common_test,
        "chicago_metro": test.loc[test["is_chicago_metro"].eq(1)],
        "outside_chicago_metro": test.loc[test["is_chicago_metro"].eq(0)],
        "near_campus": test.loc[test["distance_from_campus_miles"].le(NEAR_CAMPUS_MILES)],
    }
    result = {
        "exclude_flagged": exclude_flagged,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "spec": spec,
        "baseline_value": baseline_value,
        "baseline": {name: regression_metrics(rows["price"], np.full(len(rows), baseline_value)) for name, rows in subsets.items()},
        "models": {},
    }
    for kind in MODEL_KINDS:
        out_of_fold, cv_summary = cross_validate(kind, train, spec)
        residuals = target_values(train, spec).to_numpy() - out_of_fold
        quantiles = interval_quantiles(residuals, out_of_fold)
        model = fit_model(kind, train, spec)
        evaluation = {}
        for name, rows in subsets.items():
            evaluation[name] = regression_metrics(rows["price"], to_dollars(predict_target(model, rows, spec), spec)) if len(rows) else regression_metrics([], [])
        predicted_target = predict_target(model, test, spec)
        predicted = to_dollars(predicted_target, spec)
        low, high = interval_bounds(predicted_target, quantiles, spec)
        actual = test["price"].to_numpy(dtype=float)
        coverage = float(np.mean((actual >= low) & (actual <= high)))
        residual_dollars = actual - predicted
        result["models"][kind] = {
            "model": model,
            "cross_validation": cv_summary,
            "evaluation": evaluation,
            "interval_quantiles": quantiles,
            "interval_coverage_held_out": coverage,
            "interval_median_width": float(np.median(high - low)),
            "interval_median_relative_width": float(np.median((high - low) / predicted)),
            "error_by_price": error_by_price(actual, predicted, low, high),
            "coverage_by_predicted_band": coverage_by_predicted_band(actual, predicted_target, quantiles, spec),
            "abs_error_vs_prediction_spearman": spearman(np.abs(residual_dollars), predicted),
            "permutation_importance": grouped_permutation_importance(model, test, spec, groups),
            "plot": {"actual": actual, "predicted": predicted},
        }
        if kind == "linear":
            result["models"][kind]["dollar_effects"] = linear_dollar_effects(model, spec)
        else:
            impurity = dict(zip(spec["feature_names"], model.feature_importances_))
            result["models"][kind]["impurity_importance"] = {
                name: float(sum(impurity[column] for column in columns)) for name, columns in groups.items()
            }
    return result


def chicago_transfer(frame, train_index, exclude_flagged, skew):
    train = frame.loc[train_index]
    train = train.loc[train["is_chicago_metro"].eq(0)]
    chicago = frame.loc[frame["is_chicago_metro"].eq(1)]
    if exclude_flagged:
        train = train.loc[~train["flagged"]]
        chicago = chicago.loc[~chicago["flagged"]]
    spec = build_spec(train, skew)
    baseline_value = float(train["price"].mean())
    illinois_mean = float(train.loc[train["state"].eq("IL"), "price"].mean())
    result = {
        "train_rows": int(len(train)),
        "chicago_rows": int(len(chicago)),
        "baseline": regression_metrics(chicago["price"], np.full(len(chicago), baseline_value)),
        "illinois_baseline": regression_metrics(chicago["price"], np.full(len(chicago), illinois_mean)),
        "models": {},
    }
    for kind in MODEL_KINDS:
        model = fit_model(kind, train, spec)
        result["models"][kind] = regression_metrics(chicago["price"], to_dollars(predict_target(model, chicago, spec), spec))
    return result


def split_indexes(frame):
    return train_test_split(
        frame.index,
        test_size=TEST_SHARE,
        random_state=RANDOM_STATE,
        stratify=frame["is_chicago_metro"],
    )


def choose_model(variant):
    return min(MODEL_KINDS, key=lambda kind: variant["models"][kind]["cross_validation"]["rmse"]["mean"])


def out_of_fold_estimates(frame, kind, exclude_flagged, skew, quantiles):
    folds = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    predicted_target = np.full(len(frame), np.nan)
    for fit_positions, holdout_positions in folds.split(frame):
        fit_rows = frame.iloc[fit_positions]
        if exclude_flagged:
            fit_rows = fit_rows.loc[~fit_rows["flagged"]]
        spec = build_spec(fit_rows, skew)
        model = fit_model(kind, fit_rows, spec)
        predicted_target[holdout_positions] = predict_target(model, frame.iloc[holdout_positions], spec)
    low, high = interval_bounds(predicted_target, quantiles, {"log_target": skew["price"]["use_log"]})
    estimates = pd.DataFrame({
        "id": frame["id"].astype("int64").to_numpy(),
        "fair_rent": to_dollars(predicted_target, {"log_target": skew["price"]["use_log"]}),
        "fair_rent_low": low,
        "fair_rent_high": high,
    })
    for column in ("fair_rent", "fair_rent_low", "fair_rent_high"):
        estimates[column] = estimates[column].round(0)
    estimates["value_label"] = [
        value_label(price, low_value, high_value)
        for price, low_value, high_value in zip(frame["price"], estimates["fair_rent_low"], estimates["fair_rent_high"])
    ]
    return estimates


def model_bundle(kind, variant):
    details = variant["models"][kind]
    return {
        "kind": kind,
        "name": MODEL_NAMES[kind],
        "model": details["model"],
        "spec": variant["spec"],
        "interval": details["interval_quantiles"],
        "exclude_flagged": variant["exclude_flagged"],
    }


def load_bundle(path):
    return joblib.load(path)


def estimate_fair_rent(bundle, listings):
    frame = listings if isinstance(listings, pd.DataFrame) else pd.DataFrame(listings)
    frame = prepare_frame(frame)
    spec = bundle["spec"]
    predicted_target = predict_target(bundle["model"], frame, spec)
    low, high = interval_bounds(predicted_target, bundle["interval"], spec)
    estimates = pd.DataFrame(
        {"fair_rent": to_dollars(predicted_target, spec), "fair_rent_low": low, "fair_rent_high": high},
        index=frame.index,
    )
    estimates["value_label"] = [
        value_label(price, low_value, high_value)
        for price, low_value, high_value in zip(frame["price"], estimates["fair_rent_low"], estimates["fair_rent_high"])
    ]
    return estimates


def importance_table(variant):
    linear = variant["models"]["linear"]
    forest = variant["models"]["forest"]
    effects = {effect["feature"]: effect for effect in linear["dollar_effects"]}
    groups = feature_groups(variant["spec"])
    total_forest = sum(max(value, 0.0) for value in forest["permutation_importance"].values()) or 1.0
    rows = []
    for name, columns in groups.items():
        if len(columns) == 1:
            effect = effects[columns[0]]
            linear_dollars, linear_change = effect["dollars"], effect["change"]
        else:
            values = [effects[column]["dollars"] for column in columns]
            linear_dollars = None
            linear_change = f"ranges from {min(values):+,.0f} to {max(values):+,.0f} dollars across categories"
        rows.append({
            "feature": name,
            "linear_dollars": linear_dollars,
            "linear_change": linear_change,
            "linear_permutation_dollars": linear["permutation_importance"][name],
            "forest_permutation_dollars": forest["permutation_importance"][name],
            "forest_share": max(forest["permutation_importance"][name], 0.0) / total_forest,
            "forest_impurity_share": forest["impurity_importance"][name],
        })
    rows.sort(key=lambda row: row["forest_permutation_dollars"], reverse=True)
    return rows


def money(value):
    return "n/a" if value is None else f"${value:,.0f}"


def ratio(value):
    return "n/a" if value is None else f"{value:.3f}"


def comparison_rows(variant, subset):
    rows = [("Baseline (mean rent)", variant["baseline"][subset])]
    rows.extend((MODEL_NAMES[kind], variant["models"][kind]["evaluation"][subset]) for kind in MODEL_KINDS)
    return rows


def format_table(headers, rows):
    widths = [max(len(str(item)) for item in column) for column in zip(headers, *rows)]
    lines = ["  ".join(str(item).ljust(width) for item, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(str(item).ljust(width) for item, width in zip(row, widths)) for row in rows)
    return lines


def build_report(frame, skew, variants, transfer, chosen_variant_key, chosen_kind, estimates):
    lines = [
        "Rent model results",
        "==================",
        f"Midwest listings with price, bedrooms, bathrooms and square footage: {len(frame):,}",
        f"Flagged rows among them: {int(frame['flagged'].sum()):,}",
        f"Chicago metro rows: {int(frame['is_chicago_metro'].sum()):,}",
        f"Hold-out split: {int(TEST_SHARE * 100)}% stratified on the Chicago metro indicator, random_state={RANDOM_STATE}",
        "",
        "Engineered columns",
        "------------------",
        f"Median price per square foot: ${frame['price_per_sqft'].median():.2f}",
        f"Median cost per bedroom (studios counted as one): ${frame['cost_per_bedroom'].median():,.0f}",
        f"Median distance from campus: {frame['distance_from_campus_miles'].median():,.0f} miles",
        f"Median amenity count where amenities are listed: {frame['amenity_count'].median():.0f}",
        "Price per square foot and cost per bedroom are computed from rent itself, so they are reported and used in",
        "scoring but are not model predictors; using them to predict rent would leak the answer into the inputs.",
        "",
        "Skew check",
        "----------",
    ]
    for column, details in skew.items():
        lines.append(
            f"{column}: skew {details['raw_skew']:.2f} raw, {details['log_skew']:.2f} after {details['transform']}; "
            f"{'log applied' if details['use_log'] else 'left as is'}"
        )
    for key, variant in variants.items():
        label = "excluding flagged rows" if variant["exclude_flagged"] else "including flagged rows"
        lines.extend(["", f"Models {label}", "-" * len(f"Models {label}"), f"Training rows {variant['train_rows']:,}, held-out rows {variant['test_rows']:,}", ""])
        for subset, title in (
            ("held_out_all", "Held-out set"),
            ("held_out_unflagged", "Held-out unflagged rows (same rows for both variants)"),
            ("chicago_metro", "Held-out Chicago metro rows"),
            ("outside_chicago_metro", "Held-out rows outside the Chicago metro"),
            ("near_campus", f"Held-out rows within {NEAR_CAMPUS_MILES} miles of campus"),
        ):
            lines.append(title)
            rows = [(name, money(metric["rmse"]), money(metric["mae"]), ratio(metric["r2"]), metric["rows"]) for name, metric in comparison_rows(variant, subset)]
            lines.extend(format_table(("Model", "RMSE", "MAE", "R2", "Rows"), rows))
            lines.append("")
        lines.append(f"{FOLDS}-fold cross validation on the training set (mean +/- std across folds)")
        rows = []
        for kind in MODEL_KINDS:
            cv = variant["models"][kind]["cross_validation"]
            rows.append((
                MODEL_NAMES[kind],
                f"${cv['rmse']['mean']:,.0f} +/- {cv['rmse']['std']:,.0f}",
                f"${cv['mae']['mean']:,.0f} +/- {cv['mae']['std']:,.0f}",
                f"{cv['r2']['mean']:.3f} +/- {cv['r2']['std']:.3f}",
            ))
        lines.extend(format_table(("Model", "RMSE", "MAE", "R2"), rows))
        lines.append("")
        lines.append(f"{int(INTERVAL_COVERAGE * 100)}% prediction interval from out-of-fold residuals")
        for kind in MODEL_KINDS:
            details = variant["models"][kind]
            lines.append(
                f"{MODEL_NAMES[kind]}: held-out coverage {details['interval_coverage_held_out']:.1%}, "
                f"median width {money(details['interval_median_width'])} "
                f"({details['interval_median_relative_width']:.0%} of the estimate)"
            )
        lines.append("")
        lines.append("Error by actual rent band (held-out)")
        for kind in MODEL_KINDS:
            details = variant["models"][kind]
            lines.append(f"{MODEL_NAMES[kind]} (rank correlation of absolute error with predicted rent: {details['abs_error_vs_prediction_spearman']:.2f})")
            rows = [
                (row["band"], row["rows"], money(row["mae"]), f"{row['median_percent_error']:.1%}", f"{row['coverage']:.1%}")
                for row in details["error_by_price"]
            ]
            lines.extend(format_table(("Actual rent band", "Rows", "MAE", "Median % error", "Interval coverage"), rows))
            rows = [(row["band"], row["rows"], "n/a" if row["coverage"] is None else f"{row['coverage']:.1%}") for row in details["coverage_by_predicted_band"]]
            lines.extend(format_table(("Predicted rent band", "Rows", "Interval coverage"), rows))
            lines.append("")
    lines.extend([
        "Chicago metro fully held out",
        "----------------------------",
        f"Trained on {transfer['train_rows']:,} non-Chicago training rows, tested on all {transfer['chicago_rows']:,} Chicago metro rows",
    ])
    rows = [
        ("Baseline (Midwest mean rent)", money(transfer["baseline"]["rmse"]), money(transfer["baseline"]["mae"]), ratio(transfer["baseline"]["r2"])),
        ("Baseline (Illinois mean rent)", money(transfer["illinois_baseline"]["rmse"]), money(transfer["illinois_baseline"]["mae"]), ratio(transfer["illinois_baseline"]["r2"])),
    ]
    rows.extend((MODEL_NAMES[kind], money(metric["rmse"]), money(metric["mae"]), ratio(metric["r2"])) for kind, metric in transfer["models"].items())
    lines.append("")
    lines.extend(format_table(("Model", "RMSE", "MAE", "R2"), rows))
    chosen = variants[chosen_variant_key]
    lines.extend(["", "What drives rent (models excluding flagged rows)", "------------------------------------------------"])
    lines.append("Linear effect: dollar change at the median rent for the stated change, other features held fixed.")
    lines.append("Permutation importance: increase in held-out RMSE, in dollars, when the feature is shuffled.")
    rows = [
        (
            row["feature"],
            money(row["linear_dollars"]) if row["linear_dollars"] is not None else "see note",
            f"${row['linear_permutation_dollars']:,.0f}",
            f"${row['forest_permutation_dollars']:,.0f}",
            f"{row['forest_impurity_share']:.1%}",
        )
        for row in importance_table(chosen)
    ]
    lines.extend(format_table(("Feature", "Linear $ effect", "Linear perm.", "Forest perm.", "Forest impurity"), rows))
    lines.append("")
    lines.append("Linear effects in full")
    for effect in sorted(chosen["models"]["linear"]["dollar_effects"], key=lambda item: abs(item["dollars"]), reverse=True):
        lines.append(f"  {effect['dollars']:+8,.0f}  {effect['change']}")
    flag_effect = []
    for kind in MODEL_KINDS:
        included = variants["including_flagged"]["models"][kind]["evaluation"]["held_out_unflagged"]
        excluded = variants["excluding_flagged"]["models"][kind]["evaluation"]["held_out_unflagged"]
        flag_effect.append(
            f"{MODEL_NAMES[kind]} on the shared unflagged held-out rows: RMSE {money(excluded['rmse'])} trained without flagged rows "
            f"vs {money(included['rmse'])} trained with them"
        )
    ranking_excluded = choose_model(variants["excluding_flagged"])
    ranking_included = choose_model(variants["including_flagged"])
    lines.extend(["", "Effect of flagged rows", "----------------------", *flag_effect])
    lines.append(
        f"Best model by cross validation: {MODEL_NAMES[ranking_excluded]} without flagged rows, {MODEL_NAMES[ranking_included]} with them; "
        + ("the conclusion does not change." if ranking_excluded == ranking_included else "the conclusion changes.")
    )
    lines.extend([
        "",
        "Fair rent estimate deployed to the app",
        "--------------------------------------",
        f"Model: {MODEL_NAMES[chosen_kind]}, trained excluding flagged rows.",
        f"Interval: {int(INTERVAL_COVERAGE * 100)}% prediction interval from out-of-fold residuals, calibrated separately in {INTERVAL_BANDS} bands of predicted rent.",
        f"Every Midwest listing gets an out-of-fold estimate, so no listing is scored by a model that saw its own rent.",
        f"Estimates written: {len(estimates):,}",
        "Value labels: " + ", ".join(f"{label} {count:,}" for label, count in estimates["value_label"].value_counts().items()),
    ])
    return "\n".join(lines) + "\n"


def save_figures(variant):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["text.parse_math"] = False

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, kind in zip(axes, MODEL_KINDS):
        plot = variant["models"][kind]["plot"]
        residuals = plot["actual"] - plot["predicted"]
        axis.scatter(plot["predicted"], residuals, s=10, alpha=0.35, color=SERIES_COLORS[kind], edgecolors="none")
        axis.axhline(0, color=TEXT_COLOR, linewidth=1)
        axis.set_title(f"{MODEL_NAMES[kind]}: residuals vs predicted", color=TEXT_COLOR, fontsize=11, loc="left")
        axis.set_xlabel("Predicted rent ($)", color=TEXT_COLOR)
        axis.grid(color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(colors=TEXT_COLOR)
    axes[0].set_ylabel("Actual minus predicted ($)", color=TEXT_COLOR)
    figure.tight_layout()
    residual_path = FIGURE_DIR / "residuals.png"
    figure.savefig(residual_path, dpi=130)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    bands = [row["band"] for row in variant["models"]["linear"]["error_by_price"]]
    positions = np.arange(len(bands))
    width = 0.38
    for offset, kind in zip((-width / 2, width / 2), MODEL_KINDS):
        values = [row["mae"] for row in variant["models"][kind]["error_by_price"]]
        axis.bar(positions + offset, values, width=width - 0.02, color=SERIES_COLORS[kind], label=MODEL_NAMES[kind])
    axis.set_xticks(positions, bands, rotation=15, color=TEXT_COLOR)
    axis.set_ylabel("Mean absolute error ($)", color=TEXT_COLOR)
    axis.set_title("Held-out error by actual rent band", color=TEXT_COLOR, fontsize=11, loc="left")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(colors=TEXT_COLOR)
    axis.legend(frameon=False)
    figure.tight_layout()
    band_path = FIGURE_DIR / "error_by_rent_band.png"
    figure.savefig(band_path, dpi=130)
    plt.close(figure)
    return [residual_path, band_path]


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items() if key not in {"model", "plot"}}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def train_all(path=SLIM_FILE, write=True):
    frame = load_midwest(path)
    skew = skew_report(frame)
    train_index, test_index = split_indexes(frame)
    variants = {
        "excluding_flagged": evaluate_variant(frame, train_index, test_index, True, skew),
        "including_flagged": evaluate_variant(frame, train_index, test_index, False, skew),
    }
    transfer = chicago_transfer(frame, train_index, True, skew)
    chosen_variant_key = "excluding_flagged"
    chosen = variants[chosen_variant_key]
    chosen_kind = choose_model(chosen)
    estimates = out_of_fold_estimates(frame, chosen_kind, True, skew, chosen["models"][chosen_kind]["interval_quantiles"])
    report = build_report(frame, skew, variants, transfer, chosen_variant_key, chosen_kind, estimates)
    metadata = {
        "fair_rent_model": chosen_kind,
        "fair_rent_model_name": MODEL_NAMES[chosen_kind],
        "interval_coverage": INTERVAL_COVERAGE,
        "interval": chosen["models"][chosen_kind]["interval_quantiles"],
        "listing_years": "2018-2019",
        "rows": int(len(frame)),
        "skew": skew,
        "importance": importance_table(chosen),
        "linear_effects": chosen["models"]["linear"]["dollar_effects"],
        "variants": {key: json_ready({k: v for k, v in variant.items() if k != "spec"}) for key, variant in variants.items()},
        "chicago_transfer": json_ready(transfer),
        "amenities": chosen["spec"]["amenities"],
    }
    if write:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bundle("linear", chosen), LINEAR_MODEL_FILE, compress=3)
        joblib.dump(model_bundle("forest", chosen), FOREST_MODEL_FILE, compress=3)
        METADATA_FILE.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
        estimates.to_csv(ESTIMATES_FILE, index=False)
        REPORT_FILE.write_text(report, encoding="utf-8")
        save_figures(chosen)
    return {"frame": frame, "skew": skew, "variants": variants, "transfer": transfer, "chosen_kind": chosen_kind, "estimates": estimates, "report": report, "metadata": metadata}


def main():
    results = train_all()
    print(results["report"], end="")


if __name__ == "__main__":
    main()
