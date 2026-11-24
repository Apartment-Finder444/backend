import math

import pytest

from src import scoring
from src.scoring import (
    DEFAULT_WEIGHTS,
    NEUTRAL_SCORE,
    amenities_score,
    bathroom_score,
    bedroom_score,
    cost_per_bedroom,
    cost_per_bedroom_score,
    distance_score,
    rent_score,
    score_listing,
    square_feet_score,
    unknown_fields,
    weights_from_ranking,
)


def test_rent_full_score_when_comfortably_under_budget():
    assert rent_score(1000, 1500) == 1.0
    assert rent_score(1275, 1500) == pytest.approx(1.0)


def test_rent_declines_toward_limit_and_drops_sharply_above():
    at_limit = rent_score(1500, 1500)
    just_over = rent_score(1530, 1500)
    assert 0 < at_limit < 1
    assert at_limit == pytest.approx(scoring.RENT_AT_LIMIT_SCORE)
    assert just_over < at_limit / 1.5
    assert rent_score(1650, 1500) == 0.0
    values = [rent_score(price, 1500) for price in range(800, 1800, 10)]
    assert all(first >= second for first, second in zip(values, values[1:]))


def test_rent_unknown_or_no_budget_returns_none():
    assert rent_score(None, 1500) is None
    assert rent_score(1200, None) is None
    assert rent_score(float("nan"), 1500) is None


def test_bedroom_and_bathroom_shape():
    assert bedroom_score(2, 2) == 1.0
    assert bedroom_score(1, 2) == 0.5
    assert bedroom_score(3, 2) == 0.5
    assert bedroom_score(4, 2) == 0.0
    assert bathroom_score(1.5, 1) == 0.5
    assert bathroom_score(1, 1) == 1.0
    assert bathroom_score(3, 1) == 0.0
    assert bedroom_score(None, 2) is None


def test_square_feet_against_minimum():
    assert square_feet_score(900, 800) == 1.0
    assert square_feet_score(800, 800) == 1.0
    assert square_feet_score(680, 800) == pytest.approx(0.5)
    assert square_feet_score(500, 800) == 0.0
    assert square_feet_score(None, 800) is None


def test_distance_is_banded_not_continuous():
    assert distance_score(0.5) == distance_score(2.9) == 1.0
    assert distance_score(5) == distance_score(6.9)
    assert distance_score(5) < 1.0
    assert distance_score(200) == 0.0
    assert distance_score(None) is None
    band_values = [score for _, score in scoring.DISTANCE_BANDS]
    assert band_values == sorted(band_values, reverse=True)


def test_amenities_fraction_of_requested():
    assert amenities_score("Pool,Gym,Parking", ["Pool", "Gym"]) == 1.0
    assert amenities_score("Pool", ["pool", "Gym"]) == 0.5
    assert amenities_score("Parking", ["Pool", "Gym"]) == 0.0
    assert amenities_score(None, ["Pool"]) is None
    assert amenities_score("Pool", []) is None


def test_cost_per_bedroom_counts_studio_as_one():
    assert cost_per_bedroom(1000, 0) == 1000
    assert cost_per_bedroom(1800, 3) == 600
    assert cost_per_bedroom_score(1800, 3, 800) == 1.0
    assert cost_per_bedroom_score(1800, None, 800) is None


def test_default_weights_live_in_a_dictionary_and_sum_to_one():
    assert isinstance(DEFAULT_WEIGHTS, dict)
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
    assert weights_from_ranking(None) == DEFAULT_WEIGHTS


def test_ranking_turns_into_decreasing_weights():
    weights = weights_from_ranking(["distance", "rent"])
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["distance"] > weights["rent"] > max(value for key, value in weights.items() if key not in {"distance", "rent"})
    with pytest.raises(ValueError):
        weights_from_ranking(["sunlight"])


def test_no_preferences_gives_no_score():
    result = score_listing({"price": 1000}, {})
    assert result["match_score"] is None
    assert result["match_confidence"] is None


def test_unknown_amenities_neutral_with_lower_confidence():
    preferences = {"budget": 1500, "amenities": ["Pool", "Gym"]}
    unknown = score_listing({"price": 1200, "amenities": None}, preferences)
    known_bad = score_listing({"price": 1200, "amenities": "Parking"}, preferences)
    known_good = score_listing({"price": 1200, "amenities": "Pool,Gym"}, preferences)
    assert unknown["match_score"] > known_bad["match_score"]
    assert unknown["match_score"] < known_good["match_score"]
    assert unknown["match_confidence"] < known_bad["match_confidence"] == 1.0
    assert unknown["components"]["amenities"]["known"] is False
    assert unknown["unknown_components"] == ["amenities"]


FULL_PREFERENCES = {
    "budget": 1500,
    "bedrooms": 2,
    "bathrooms": 1,
    "min_square_feet": 900,
    "distance": True,
    "amenities": ["Pool", "Gym"],
    "per_person_budget": 600,
}
BAD_LISTING = {
    "price": 2400,
    "bedrooms": 5,
    "bathrooms": 4,
    "square_feet": 300,
    "distance_from_campus_miles": 500,
    "amenities": "Parking",
}


@pytest.mark.parametrize(
    "field",
    ["bedrooms", "bathrooms", "square_feet", "distance_from_campus_miles", "amenities"],
)
def test_unknown_value_never_scores_lower_than_known_bad_value(field):
    unknown_listing = dict(BAD_LISTING)
    unknown_listing[field] = None
    unknown = score_listing(unknown_listing, FULL_PREFERENCES)
    known_bad = score_listing(BAD_LISTING, FULL_PREFERENCES)
    assert unknown["match_score"] >= known_bad["match_score"]
    assert unknown["match_confidence"] < known_bad["match_confidence"]


@pytest.mark.parametrize(
    "component, listing_value",
    [
        ("bedrooms", {"bedrooms": 9}),
        ("bathrooms", {"bathrooms": 6}),
        ("square_feet", {"square_feet": 100}),
        ("distance", {"distance_from_campus_miles": 900}),
        ("amenities", {"amenities": "Tennis"}),
    ],
)
def test_every_component_unknown_is_at_least_as_good_as_worst_known(component, listing_value):
    preferences = dict(FULL_PREFERENCES)
    worst = scoring.component_score(component, {"price": 1000, **listing_value}, preferences)
    assert worst is not None
    assert NEUTRAL_SCORE >= worst


def test_unknown_listing_is_not_pushed_below_known_worse_listing_overall():
    preferences = {"budget": 1500, "bedrooms": 2, "distance": True}
    weights = weights_from_ranking(["distance", "rent", "bedrooms"])
    unknown_location = score_listing({"price": 1300, "bedrooms": 2, "distance_from_campus_miles": None}, preferences, weights)
    far_away = score_listing({"price": 1300, "bedrooms": 2, "distance_from_campus_miles": 400}, preferences, weights)
    assert unknown_location["match_score"] > far_away["match_score"]


def test_score_bounds_and_confidence_bounds():
    result = score_listing({**BAD_LISTING, "price": 1000, "bedrooms": 2}, FULL_PREFERENCES)
    assert 0 <= result["match_score"] <= 1
    assert 0 <= result["match_confidence"] <= 1
    assert not math.isnan(result["match_score"])


def test_unknown_fields_lists_missing_listing_fields():
    listing = {
        "pets_allowed": None,
        "amenities": "Pool",
        "bedrooms": 1,
        "bathrooms": None,
        "square_feet": 700,
        "latitude": 41.8,
        "longitude": None,
        "fair_rent": 1200,
    }
    assert unknown_fields(listing) == ["pet_policy", "bathrooms", "location"]
