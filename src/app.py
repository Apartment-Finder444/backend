import json
import math
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.model import estimate_fair_rent

from src.scoring import (
    COMPONENT_LABELS,
    DEFAULT_WEIGHTS,
    DISTANCE_BANDS,
    is_missing,
    parse_amenity_list,
    score_listing,
    unknown_fields,
    weights_from_ranking,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data" / "clean" / "apartments.db"
DEFAULT_METADATA = ROOT / "models" / "model_metadata.json"
DEFAULT_MODEL = ROOT / "models" / "random_forest.joblib"
APP_NAME = "Campus Apartment Finder"
DATA_YEARS = "2018-2019"
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
TRUE_VALUES = {"1", "true", "yes", "on"}
FLAG_LABELS = {
    "flag_bedroom_conflict": "Title says studio but bedroom count is one or more",
    "flag_placeholder_coords": "Location was a placeholder and has been removed",
    "flag_coord_mismatch": "Location falls far from the stated city or state",
    "flag_suspect_price": "Rent is unusually low or high for the bedroom count",
    "flag_suspect_size": "Square footage is implausible for the bedroom count",
}
FALLBACK_AMENITIES = [
    "AC", "Cable or Satellite", "Clubhouse", "Dishwasher", "Fireplace", "Garbage Disposal", "Gym",
    "Internet Access", "Parking", "Patio/Deck", "Playground", "Pool", "Refrigerator", "Storage",
    "Tennis", "Washer Dryer", "Wood Floors",
]
NUMERIC_FILTERS = {
    "max_rent": ("price", "<="),
    "min_bedrooms": ("bedrooms", ">="),
    "max_bedrooms": ("bedrooms", "<="),
    "min_bathrooms": ("bathrooms", ">="),
    "max_distance": ("distance_from_campus_miles", "<="),
}
NUMERIC_PREFERENCES = ("budget", "bedrooms", "bathrooms", "min_square_feet", "per_person_budget")


class BadRequest(ValueError):
    pass


def is_unknown_pet_policy(value):
    return is_missing(value)


def pet_policy_matches(value, requested_pets):
    if is_unknown_pet_policy(value):
        return True
    policy = str(value).strip().casefold()
    if policy == "none_allowed":
        return False
    requested = str(requested_pets).strip().casefold()
    if requested in {"true", "yes", "allowed", "any"}:
        return True
    available = {part.strip() for part in policy.split(",") if part.strip() != "none"}
    wanted = {part.strip() for part in requested.replace("+", ",").split(",") if part.strip()}
    return bool(wanted) and wanted.issubset(available)


def pet_policy_label(value):
    if is_unknown_pet_policy(value):
        return "Pet policy not listed"
    policy = str(value).strip().casefold()
    if policy == "none_allowed":
        return "No pets allowed"
    animals = [part.strip() for part in policy.split(",") if part.strip() in {"cats", "dogs"}]
    if animals == ["cats", "dogs"] or set(animals) == {"cats", "dogs"}:
        return "Cats and dogs allowed"
    if animals == ["cats"]:
        return "Cats allowed"
    if animals == ["dogs"]:
        return "Dogs allowed"
    return "Pet policy not listed"


def filter_pet_results(listings, pets=None, confirmed_only=False):
    results = []
    for listing in listings:
        value = listing.get("pets_allowed")
        unknown = is_unknown_pet_policy(value)
        if confirmed_only and unknown:
            continue
        if pets and not pet_policy_matches(value, pets):
            continue
        result = dict(listing)
        result["pets_unconfirmed"] = unknown
        results.append(result)
    if pets:
        results.sort(key=lambda listing: (listing["pets_unconfirmed"], listing.get("price") or math.inf))
    return results


def truthy(value):
    return value is not None and str(value).strip().casefold() in TRUE_VALUES


def parse_number(arguments, name):
    raw_value = arguments.get(name)
    if raw_value is None or not str(raw_value).strip():
        return None
    try:
        value = float(raw_value)
    except ValueError as error:
        raise BadRequest(f"{name} must be numeric") from error
    if not math.isfinite(value):
        raise BadRequest(f"{name} must be finite")
    return value


def parse_preferences(arguments):
    preferences = {name: parse_number(arguments, name) for name in NUMERIC_PREFERENCES}
    if preferences["budget"] is None:
        preferences["budget"] = parse_number(arguments, "max_rent")
    if preferences["bedrooms"] is None:
        preferences["bedrooms"] = parse_number(arguments, "min_bedrooms")
    if preferences["bathrooms"] is None:
        preferences["bathrooms"] = parse_number(arguments, "min_bathrooms")
    preferences["amenities"] = parse_amenity_list(arguments.get("amenities")) or []
    preferences["distance"] = parse_number(arguments, "max_distance") is not None or truthy(arguments.get("near_campus"))
    ranking = [part.strip() for part in arguments.get("priorities", "").split(",") if part.strip()]
    try:
        preferences["weights"] = weights_from_ranking(ranking)
    except ValueError as error:
        raise BadRequest(str(error)) from error
    preferences["ranking"] = ranking
    return preferences


def clean_value(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def rounded(value, digits=0):
    value = clean_value(value)
    if value is None:
        return None
    try:
        number = round(float(value), digits)
    except (TypeError, ValueError):
        return None
    return int(number) if digits == 0 else number


def as_float(value):
    value = clean_value(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def is_flag_set(value):
    return not is_missing(value) and str(value).strip().casefold() in {"1", "true", "1.0"}


def present_listing(row, preferences):
    listing = {key: clean_value(value) for key, value in row.items()}
    scored = score_listing(listing, preferences, preferences["weights"])
    amenities = parse_amenity_list(listing.get("amenities"))
    label = listing.get("value_label")
    if is_missing(label) or rounded(listing.get("fair_rent")) is None:
        label = "unknown"
    price = as_float(listing.get("price"))
    square_feet = as_float(listing.get("square_feet"))
    return {
        "id": rounded(listing.get("id")),
        "price": rounded(listing.get("price")),
        "bedrooms": clean_value(listing.get("bedrooms")),
        "bathrooms": clean_value(listing.get("bathrooms")),
        "square_feet": rounded(listing.get("square_feet")),
        "price_per_sqft": round(price / square_feet, 2) if price and square_feet else None,
        "cityname": listing.get("cityname"),
        "state": listing.get("state"),
        "latitude": clean_value(listing.get("latitude")),
        "longitude": clean_value(listing.get("longitude")),
        "distance_from_campus_miles": rounded(listing.get("distance_from_campus_miles"), 1),
        "pets_allowed": None if is_unknown_pet_policy(listing.get("pets_allowed")) else listing.get("pets_allowed"),
        "pet_policy": pet_policy_label(listing.get("pets_allowed")),
        "pets_unconfirmed": bool(listing.get("pets_unconfirmed", is_unknown_pet_policy(listing.get("pets_allowed")))),
        "amenities": amenities,
        "has_photo": listing.get("has_photo"),
        "source": listing.get("source"),
        "data_flags": [description for column, description in FLAG_LABELS.items() if is_flag_set(listing.get(column))],
        "match_score": scored["match_score"],
        "match_confidence": scored["match_confidence"],
        "score_components": scored["components"],
        "fair_rent": rounded(listing.get("fair_rent")),
        "fair_rent_low": rounded(listing.get("fair_rent_low")),
        "fair_rent_high": rounded(listing.get("fair_rent_high")),
        "value_label": label,
        "unknown_fields": unknown_fields(listing),
    }


def sort_key(listing, pets_requested):
    score = listing["match_score"]
    return (
        listing["pets_unconfirmed"] if pets_requested else False,
        -(score if score is not None else -1.0),
        -(listing["match_confidence"] or 0.0),
        listing["price"] if listing["price"] is not None else math.inf,
    )


def read_metadata(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_request_frame(arguments):
    required = {name: parse_number(arguments, name) for name in ("bedrooms", "bathrooms", "square_feet")}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise BadRequest(f"Missing required parameters: {', '.join(missing)}")
    if required["square_feet"] <= 0:
        raise BadRequest("square_feet must be positive")
    distance = parse_number(arguments, "distance")
    pets = arguments.get("pets", "").strip().casefold()
    pets_value = {"cats": "Cats", "dogs": "Dogs", "cats,dogs": "Cats,Dogs", "none": "none_allowed"}.get(pets)
    return pd.DataFrame([{
        "id": 0,
        "price": parse_number(arguments, "price"),
        "bedrooms": required["bedrooms"],
        "bathrooms": required["bathrooms"],
        "square_feet": required["square_feet"],
        "cityname": arguments.get("city", "").strip() or None,
        "state": (arguments.get("state", "IL").strip() or "IL").upper(),
        "latitude": None,
        "longitude": None,
        "distance_from_campus_miles": distance,
        "amenities": ",".join(parse_amenity_list(arguments.get("amenities")) or []) or None,
        "pets_allowed": pets_value,
    }])


def create_app(database_path=DEFAULT_DATABASE, metadata_path=DEFAULT_METADATA, model_path=DEFAULT_MODEL):
    application = Flask(__name__)
    application.config["DATABASE_PATH"] = str(database_path)
    application.config["METADATA_PATH"] = str(metadata_path)
    application.config["MODEL_PATH"] = str(model_path)
    loaded_models = {}

    def fair_rent_model():
        path = application.config["MODEL_PATH"]
        if path not in loaded_models:
            loaded_models[path] = joblib.load(path) if Path(path).exists() else None
        return loaded_models[path]

    def database_rows(query, values=()):
        with sqlite3.connect(application.config["DATABASE_PATH"]) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query, values).fetchall()]

    def metadata():
        return read_metadata(application.config["METADATA_PATH"])

    def response_meta(preferences):
        model_metadata = metadata() or {}
        return {
            "data_years": DATA_YEARS,
            "prices_are_historical": True,
            "fair_rent_interval_coverage": model_metadata.get("interval_coverage"),
            "fair_rent_model": model_metadata.get("fair_rent_model_name"),
            "weights": {name: round(value, 3) for name, value in preferences["weights"].items()},
            "neutral_score_for_unknown": 0.5,
        }

    @application.errorhandler(BadRequest)
    def bad_request(error):
        return jsonify({"error": str(error)}), 400

    @application.get("/")
    def index():
        return render_template("index.html", data_years=DATA_YEARS, app_name=APP_NAME)

    @application.get("/api/search")
    def search():
        arguments = request.args
        clauses = ["1 = 1"]
        values = []
        for parameter, (column, operator) in NUMERIC_FILTERS.items():
            value = parse_number(arguments, parameter)
            if value is None:
                continue
            clauses.append(f"{column} {operator} ?")
            values.append(value)
        for parameter, column in (("city", "cityname"), ("state", "state")):
            requested_value = arguments.get(parameter)
            if requested_value and requested_value.strip():
                clauses.append(f"lower({column}) = lower(?)")
                values.append(requested_value.strip())
        if truthy(arguments.get("hide_flagged")):
            for column in FLAG_LABELS:
                clauses.append(f"coalesce({column}, 0) IN (0, '0', 'False', 'false')")
        try:
            limit = int(arguments.get("limit", str(DEFAULT_LIMIT)))
        except ValueError as error:
            raise BadRequest("limit must be an integer") from error
        limit = min(MAX_LIMIT, max(1, limit))
        preferences = parse_preferences(arguments)
        rows = database_rows(f"SELECT * FROM apartments WHERE {' AND '.join(clauses)}", values)
        pets = arguments.get("pets")
        filtered = filter_pet_results(rows, pets=pets, confirmed_only=truthy(arguments.get("confirmed_only")))
        listings = [present_listing(row, preferences) for row in filtered]
        listings.sort(key=lambda listing: sort_key(listing, bool(pets)))
        results = listings[:limit]
        return jsonify({
            "count": len(results),
            "total_matches": len(listings),
            "results": results,
            "meta": response_meta(preferences),
        })

    @application.get("/api/listing/<int:listing_id>")
    def listing(listing_id):
        rows = database_rows("SELECT * FROM apartments WHERE id = ?", (listing_id,))
        if not rows:
            return jsonify({"error": "Listing not found"}), 404
        preferences = parse_preferences(request.args)
        result = present_listing(filter_pet_results(rows)[0], preferences)
        result["meta"] = response_meta(preferences)
        return jsonify(result)

    @application.get("/api/estimate")
    def estimate():
        bundle = fair_rent_model()
        if bundle is None:
            return jsonify({"error": "Trained model not found; run python -m src.model"}), 503
        frame = estimate_request_frame(request.args)
        if frame.loc[0, "state"] not in bundle["spec"]["states"]:
            raise BadRequest(f"state must be one of {', '.join(bundle['spec']['states'])}")
        result = estimate_fair_rent(bundle, frame).iloc[0]
        price = parse_number(request.args, "price")
        return jsonify({
            "fair_rent": rounded(result["fair_rent"]),
            "fair_rent_low": rounded(result["fair_rent_low"]),
            "fair_rent_high": rounded(result["fair_rent_high"]),
            "price": rounded(price),
            "value_label": result["value_label"] if price is not None else "unknown",
            "interval_coverage": bundle["interval"]["coverage"],
            "model": bundle.get("name"),
            "data_years": DATA_YEARS,
            "note": "Estimated from 2018-2019 Midwest listings; a historical range, not a current price.",
        })

    @application.get("/api/feature-importance")
    def feature_importance():
        model_metadata = metadata()
        if not model_metadata:
            return jsonify({"error": "Model results not found; run python -m src.model"}), 503
        return jsonify({
            "data_years": DATA_YEARS,
            "rows": model_metadata.get("rows"),
            "model": model_metadata.get("fair_rent_model_name"),
            "importance": model_metadata.get("importance", []),
            "linear_effects": model_metadata.get("linear_effects", []),
            "notes": [
                "Forest importance is the rise in held-out error, in dollars, when a feature is shuffled.",
                "Linear effects are dollar changes at the median rent with other features held fixed.",
                "Associations in 2018-2019 Midwest listings, not causes.",
            ],
        })

    @application.get("/api/options")
    def options():
        model_metadata = metadata() or {}
        states = [row["state"] for row in database_rows("SELECT DISTINCT state FROM apartments ORDER BY state")]
        return jsonify({
            "data_years": DATA_YEARS,
            "states": states,
            "amenities": model_metadata.get("amenities") or FALLBACK_AMENITIES,
            "priorities": [{"key": key, "label": COMPONENT_LABELS[key]} for key in DEFAULT_WEIGHTS],
            "default_weights": DEFAULT_WEIGHTS,
            "distance_bands": [{"up_to_miles": limit, "score": score} for limit, score in DISTANCE_BANDS],
        })

    return application


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
