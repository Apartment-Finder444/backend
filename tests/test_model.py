import numpy as np
import pandas as pd
import pytest

from src import model


def synthetic_listings(count, seed):
    generator = np.random.default_rng(seed)
    bedrooms = generator.integers(0, 4, count)
    square_feet = 450 + bedrooms * 300 + generator.normal(0, 80, count).clip(-200, 200)
    states = generator.choice(["IL", "OH", "MN"], count)
    state_factor = pd.Series(states).map({"IL": 1.15, "OH": 0.9, "MN": 1.05}).to_numpy()
    price = np.exp(np.log(square_feet) * 0.8 + 1.4 + generator.normal(0, 0.15, count)) * state_factor
    latitude = 41.8 + generator.normal(0, 0.5, count)
    longitude = -88.1 + generator.normal(0, 0.5, count)
    amenities = generator.choice(["Pool,Gym", "Parking", None, "Pool,Parking,Gym"], count)
    pets = generator.choice(["Cats,Dogs", "none_allowed", None], count)
    return pd.DataFrame({
        "id": np.arange(count) + seed * 10_000,
        "price": price.round(0),
        "bedrooms": bedrooms.astype(float),
        "bathrooms": np.where(bedrooms > 1, 2.0, 1.0),
        "square_feet": square_feet.round(0),
        "cityname": np.where(states == "IL", "Chicago", "Other"),
        "state": states,
        "latitude": latitude,
        "longitude": longitude,
        "amenities": amenities,
        "pets_allowed": pets,
    })


@pytest.fixture(scope="module")
def fitted():
    train = model.prepare_frame(synthetic_listings(900, 1))
    skew = model.skew_report(train)
    spec = model.build_spec(train, skew)
    out_of_fold, _ = model.cross_validate("forest", train, spec)
    residuals = model.target_values(train, spec).to_numpy() - out_of_fold
    interval = model.interval_quantiles(residuals, out_of_fold)
    forest = model.fit_model("forest", train, spec)
    bundle = {"kind": "forest", "model": forest, "spec": spec, "interval": interval}
    return bundle


def test_interval_brackets_the_estimate(fitted):
    estimates = model.estimate_fair_rent(fitted, synthetic_listings(200, 2))
    assert (estimates["fair_rent_low"] < estimates["fair_rent"]).all()
    assert (estimates["fair_rent"] < estimates["fair_rent_high"]).all()
    assert (estimates["fair_rent_low"] > 0).all()


def test_interval_coverage_close_to_nominal_on_new_data(fitted):
    fresh = synthetic_listings(600, 3)
    estimates = model.estimate_fair_rent(fitted, fresh)
    covered = fresh["price"].between(estimates["fair_rent_low"], estimates["fair_rent_high"])
    assert 0.68 <= covered.mean() <= 0.92


def test_interval_width_is_sensible(fitted):
    estimates = model.estimate_fair_rent(fitted, synthetic_listings(200, 4))
    relative_width = (estimates["fair_rent_high"] - estimates["fair_rent_low"]) / estimates["fair_rent"]
    assert relative_width.median() < 1.0
    assert relative_width.min() > 0.05


def test_value_label_uses_interval_not_midpoint():
    assert model.value_label(1100, 900, 1300) == "within"
    assert model.value_label(1250, 900, 1300) == "within"
    assert model.value_label(1350, 900, 1300) == "above"
    assert model.value_label(850, 900, 1300) == "below"
    assert model.value_label(1000, None, 1300) == "unknown"


def test_interval_quantiles_cover_requested_share():
    generator = np.random.default_rng(7)
    residuals = generator.normal(0, 0.2, 5000)
    predicted = generator.normal(7, 0.3, 5000)
    interval = model.interval_quantiles(residuals, predicted, coverage=0.8)
    low, high = model.interval_bounds(predicted, interval, {"log_target": True})
    actual = np.exp(predicted + residuals)
    assert np.mean((actual >= low) & (actual <= high)) == pytest.approx(0.8, abs=0.02)
    assert len(interval["limits"]) == model.INTERVAL_BANDS


def test_price_derived_columns_are_not_predictors(fitted):
    names = fitted["spec"]["feature_names"]
    assert "price_per_sqft" not in names
    assert "cost_per_bedroom" not in names
    assert "price" not in names


@pytest.mark.skipif(not model.FOREST_MODEL_FILE.exists(), reason="trained model not built")
def test_saved_model_gives_plausible_range_near_campus():
    bundle = model.load_bundle(model.FOREST_MODEL_FILE)
    listing = pd.DataFrame([{
        "id": 1, "price": 1200, "bedrooms": 1, "bathrooms": 1, "square_feet": 700,
        "cityname": "Lisle", "state": "IL", "latitude": 41.7959, "longitude": -88.0855,
        "amenities": "Parking,Pool", "pets_allowed": "Cats,Dogs",
    }])
    estimate = model.estimate_fair_rent(bundle, listing).iloc[0]
    assert 500 < estimate["fair_rent_low"] < estimate["fair_rent"] < estimate["fair_rent_high"] < 4000
