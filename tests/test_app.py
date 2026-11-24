import json
import sqlite3

import pytest

from src.app import create_app, filter_pet_results, pet_policy_label


LISTING_KEYS = {
    "id", "price", "bedrooms", "bathrooms", "square_feet", "cityname", "state", "latitude", "longitude",
    "distance_from_campus_miles", "pets_allowed", "pet_policy", "pets_unconfirmed", "amenities", "has_photo",
    "source", "data_flags", "match_score", "match_confidence", "score_components", "fair_rent", "fair_rent_low",
    "fair_rent_high", "value_label", "unknown_fields",
}
ROWS = [
    (1, 1500, 1, 1, 700, "Lisle", "IL", 41.7959, -88.0855, "Cats,Dogs", "Pool,Gym", "Yes", "source", 0, 0, 0, 0, 0, 0.0, 1400, 1150, 1700, "within"),
    (2, 1200, 1, 1, 650, "Lisle", "IL", 41.8, -88.08, None, None, "Yes", "source", 0, 0, 0, 0, 0, 1.0, 1300, 1100, 1600, "within"),
    (3, 1100, 1, 1, 600, "Lisle", "IL", 41.81, -88.07, "none_allowed", "Pool", "Yes", "source", 0, 0, 0, 0, 0, 2.0, 1000, 850, 1050, "above"),
    (4, 900, 2, 1, 900, "Naperville", "IL", 41.75, -88.15, "Dogs", "Parking", "No", "source", 0, 0, 0, 1, 0, 5.0, 1300, 1100, 1550, "below"),
    (5, 2500, 3, 2, 1400, "Chicago", "IL", None, None, "Cats", None, "Yes", "source", 0, 1, 0, 0, 0, None, None, None, None, None),
]


@pytest.fixture
def client(tmp_path):
    database = tmp_path / "apartments.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE apartments ("
            "id INTEGER PRIMARY KEY, price REAL, bedrooms REAL, bathrooms REAL, square_feet REAL, "
            "cityname TEXT, state TEXT, latitude REAL, longitude REAL, pets_allowed TEXT, amenities TEXT, "
            "has_photo TEXT, source TEXT, flag_bedroom_conflict INTEGER, flag_placeholder_coords INTEGER, "
            "flag_coord_mismatch INTEGER, flag_suspect_price INTEGER, flag_suspect_size INTEGER, "
            "distance_from_campus_miles REAL, fair_rent REAL, fair_rent_low REAL, fair_rent_high REAL, value_label TEXT)"
        )
        connection.executemany(f"INSERT INTO apartments VALUES ({', '.join('?' * 23)})", ROWS)
    metadata = tmp_path / "model_metadata.json"
    metadata.write_text(json.dumps({
        "fair_rent_model_name": "Random forest",
        "interval_coverage": 0.8,
        "rows": 5,
        "amenities": ["Gym", "Parking", "Pool"],
        "importance": [{"feature": "Square footage", "forest_permutation_dollars": 180.0, "linear_dollars": 54.0, "linear_change": "10% more square footage"}],
        "linear_effects": [{"feature": "bathrooms", "change": "one more bathroom", "dollars": 195.0, "coefficient": 0.2}],
    }))
    return create_app(database, metadata).test_client()


def test_pets_filter_keeps_unknown_below_confirmed():
    listings = [
        {"id": 1, "price": 1500, "pets_allowed": "Cats,Dogs"},
        {"id": 2, "price": 1200, "pets_allowed": None},
        {"id": 3, "price": 1100, "pets_allowed": "none_allowed"},
    ]
    results = filter_pet_results(listings, pets="dogs")
    assert [listing["id"] for listing in results] == [1, 2]
    assert results[0]["pets_unconfirmed"] is False
    assert results[1]["pets_unconfirmed"] is True
    assert [listing["id"] for listing in filter_pet_results(listings, pets="dogs", confirmed_only=True)] == [1]


def test_pet_labels_separate_unknown_from_no():
    assert pet_policy_label(None) == "Pet policy not listed"
    assert pet_policy_label("none_allowed") == "No pets allowed"
    assert pet_policy_label("Cats,Dogs") == "Cats and dogs allowed"


def test_search_returns_documented_shape(client):
    response = client.get("/api/search?budget=1500&bedrooms=1&amenities=Pool&max_distance=10")
    assert response.status_code == 200
    payload = response.get_json()
    assert {"count", "total_matches", "results", "meta"}.issubset(payload)
    assert payload["meta"]["data_years"] == "2018-2019"
    assert payload["meta"]["fair_rent_interval_coverage"] == 0.8
    for listing in payload["results"]:
        assert LISTING_KEYS.issubset(listing)
        assert 0 <= listing["match_score"] <= 1
        assert 0 <= listing["match_confidence"] <= 1
        assert listing["value_label"] in {"below", "within", "above", "unknown"}
        assert isinstance(listing["unknown_fields"], list)
        if listing["fair_rent"] is not None:
            assert listing["fair_rent_low"] <= listing["fair_rent"] <= listing["fair_rent_high"]
    scores = [listing["match_score"] for listing in payload["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_marks_unknowns_and_missing_estimate(client):
    payload = client.get("/api/search?state=IL&limit=10").get_json()
    by_id = {listing["id"]: listing for listing in payload["results"]}
    assert by_id[2]["pets_unconfirmed"] is True
    assert by_id[2]["pet_policy"] == "Pet policy not listed"
    assert set(by_id[2]["unknown_fields"]) == {"pet_policy", "amenities"}
    assert by_id[5]["value_label"] == "unknown"
    assert by_id[5]["fair_rent"] is None
    assert "location" in by_id[5]["unknown_fields"]
    assert by_id[4]["data_flags"]


def test_pets_filter_on_endpoint(client):
    payload = client.get("/api/search?pets=dogs&budget=2000").get_json()
    ids = [listing["id"] for listing in payload["results"]]
    assert 3 not in ids
    assert 5 not in ids
    assert ids[-1] == 2
    assert all(not listing["pets_unconfirmed"] for listing in payload["results"][:-1])
    confirmed = client.get("/api/search?pets=dogs&confirmed_only=true").get_json()
    assert sorted(listing["id"] for listing in confirmed["results"]) == [1, 4]


def test_unknown_amenities_not_ranked_below_known_miss(client):
    payload = client.get("/api/search?amenities=Gym&min_bedrooms=1&max_bedrooms=1").get_json()
    by_id = {listing["id"]: listing for listing in payload["results"]}
    assert by_id[2]["match_score"] > by_id[3]["match_score"]
    assert by_id[2]["match_confidence"] < by_id[3]["match_confidence"]


def test_hide_flagged(client):
    payload = client.get("/api/search?hide_flagged=true").get_json()
    ids = {listing["id"] for listing in payload["results"]}
    assert 4 not in ids and 5 not in ids


def test_bad_parameters_return_400(client):
    assert client.get("/api/search?max_rent=abc").status_code == 400
    assert client.get("/api/search?priorities=sunlight").status_code == 400
    assert client.get("/api/search?limit=ten").status_code == 400


def test_listing_endpoint_returns_single_listing(client):
    response = client.get("/api/listing/1?budget=1600")
    assert response.status_code == 200
    payload = response.get_json()
    assert LISTING_KEYS.issubset(payload)
    assert payload["fair_rent"] == 1400
    assert payload["match_score"] is not None
    assert client.get("/api/listing/999").status_code == 404


def test_feature_importance_endpoint(client):
    payload = client.get("/api/feature-importance").get_json()
    assert payload["importance"][0]["feature"] == "Square footage"
    assert payload["linear_effects"]


def test_options_and_index(client):
    options = client.get("/api/options").get_json()
    assert options["states"] == ["IL"]
    assert {item["key"] for item in options["priorities"]} >= {"rent", "distance", "amenities"}
    page = client.get("/")
    assert page.status_code == 200
    assert b"2018 and 2019" in page.data
