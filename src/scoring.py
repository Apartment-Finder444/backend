import math


DEFAULT_WEIGHTS = {
    "rent": 0.25,
    "bedrooms": 0.15,
    "distance": 0.15,
    "cost_per_bedroom": 0.15,
    "bathrooms": 0.10,
    "square_feet": 0.10,
    "amenities": 0.10,
}
COMPONENT_LABELS = {
    "rent": "Rent within budget",
    "bedrooms": "Bedroom count",
    "bathrooms": "Bathroom count",
    "square_feet": "Square footage",
    "distance": "Distance from campus",
    "amenities": "Requested amenities",
    "cost_per_bedroom": "Cost per bedroom",
}
NEUTRAL_SCORE = 0.5
RENT_COMFORT_RATIO = 0.85
RENT_AT_LIMIT_SCORE = 0.6
RENT_JUST_OVER_SCORE = 0.3
RENT_ZERO_RATIO = 1.10
COUNT_PARTIAL_SCORE = 0.5
SIZE_FLOOR_RATIO = 0.7
DISTANCE_BANDS = ((3.0, 1.0), (7.0, 0.85), (15.0, 0.65), (30.0, 0.4), (60.0, 0.2))
DISTANCE_BEYOND_SCORE = 0.0
UNKNOWN_FIELD_CHECKS = {
    "pet_policy": ("pets_allowed",),
    "amenities": ("amenities",),
    "bedrooms": ("bedrooms",),
    "bathrooms": ("bathrooms",),
    "square_feet": ("square_feet",),
    "location": ("latitude", "longitude"),
    "fair_rent": ("fair_rent",),
}


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().casefold() in {"", "nan", "null", "none"}:
        return True
    return False


def as_number(value):
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clamp(value):
    return max(0.0, min(1.0, value))


def budget_score(cost, limit):
    cost = as_number(cost)
    limit = as_number(limit)
    if cost is None or limit is None or limit <= 0:
        return None
    ratio = cost / limit
    if ratio <= RENT_COMFORT_RATIO:
        return 1.0
    if ratio <= 1.0:
        progress = (ratio - RENT_COMFORT_RATIO) / (1.0 - RENT_COMFORT_RATIO)
        return 1.0 - progress * (1.0 - RENT_AT_LIMIT_SCORE)
    if ratio >= RENT_ZERO_RATIO:
        return 0.0
    progress = (ratio - 1.0) / (RENT_ZERO_RATIO - 1.0)
    return clamp(RENT_JUST_OVER_SCORE * (1.0 - progress))


def rent_score(price, budget):
    return budget_score(price, budget)


def count_match_score(value, target):
    value = as_number(value)
    target = as_number(target)
    if value is None or target is None:
        return None
    difference = abs(value - target)
    if difference < 0.25:
        return 1.0
    if difference <= 1.0:
        return COUNT_PARTIAL_SCORE
    return 0.0


def bedroom_score(bedrooms, target):
    return count_match_score(bedrooms, target)


def bathroom_score(bathrooms, target):
    return count_match_score(bathrooms, target)


def square_feet_score(square_feet, minimum):
    square_feet = as_number(square_feet)
    minimum = as_number(minimum)
    if square_feet is None or minimum is None or minimum <= 0 or square_feet <= 0:
        return None
    ratio = square_feet / minimum
    if ratio >= 1.0:
        return 1.0
    if ratio <= SIZE_FLOOR_RATIO:
        return 0.0
    return clamp((ratio - SIZE_FLOOR_RATIO) / (1.0 - SIZE_FLOOR_RATIO))


def distance_score(miles):
    miles = as_number(miles)
    if miles is None or miles < 0:
        return None
    for limit, score in DISTANCE_BANDS:
        if miles <= limit:
            return score
    return DISTANCE_BEYOND_SCORE


def parse_amenity_list(value):
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items
    if is_missing(value):
        return None
    return [part.strip() for part in str(value).split(",") if part.strip()]


def amenities_score(listing_amenities, requested):
    wanted = {item.casefold() for item in (parse_amenity_list(requested) or [])}
    if not wanted:
        return None
    available = parse_amenity_list(listing_amenities)
    if available is None:
        return None
    confirmed = wanted & {item.casefold() for item in available}
    return len(confirmed) / len(wanted)


def cost_per_bedroom(price, bedrooms):
    price = as_number(price)
    bedrooms = as_number(bedrooms)
    if price is None or bedrooms is None:
        return None
    return price / max(bedrooms, 1.0)


def cost_per_bedroom_score(price, bedrooms, per_person_budget):
    return budget_score(cost_per_bedroom(price, bedrooms), per_person_budget)


def weights_from_ranking(ranking=None, defaults=DEFAULT_WEIGHTS):
    if not ranking:
        return dict(defaults)
    ordered = []
    for name in ranking:
        if name not in defaults:
            raise ValueError(f"Unknown priority: {name}")
        if name not in ordered:
            ordered.append(name)
    remaining = sorted((name for name in defaults if name not in ordered), key=lambda name: -defaults[name])
    ordered.extend(remaining)
    count = len(ordered)
    raw = {name: float(count - position) for position, name in enumerate(ordered)}
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def active_components(preferences):
    active = []
    if as_number(preferences.get("budget")) is not None:
        active.append("rent")
    if as_number(preferences.get("bedrooms")) is not None:
        active.append("bedrooms")
    if as_number(preferences.get("bathrooms")) is not None:
        active.append("bathrooms")
    if as_number(preferences.get("min_square_feet")) is not None:
        active.append("square_feet")
    if preferences.get("distance"):
        active.append("distance")
    if parse_amenity_list(preferences.get("amenities")):
        active.append("amenities")
    if as_number(preferences.get("per_person_budget")) is not None:
        active.append("cost_per_bedroom")
    return active


def component_score(name, listing, preferences):
    if name == "rent":
        return rent_score(listing.get("price"), preferences.get("budget"))
    if name == "bedrooms":
        return bedroom_score(listing.get("bedrooms"), preferences.get("bedrooms"))
    if name == "bathrooms":
        return bathroom_score(listing.get("bathrooms"), preferences.get("bathrooms"))
    if name == "square_feet":
        return square_feet_score(listing.get("square_feet"), preferences.get("min_square_feet"))
    if name == "distance":
        return distance_score(listing.get("distance_from_campus_miles"))
    if name == "amenities":
        return amenities_score(listing.get("amenities"), preferences.get("amenities"))
    if name == "cost_per_bedroom":
        return cost_per_bedroom_score(listing.get("price"), listing.get("bedrooms"), preferences.get("per_person_budget"))
    raise ValueError(f"Unknown component: {name}")


def combine(component_values, weights):
    total_weight = sum(weights[name] for name in component_values)
    if total_weight <= 0:
        return None, None
    score = 0.0
    known_weight = 0.0
    for name, value in component_values.items():
        weight = weights[name]
        if value is None:
            score += weight * NEUTRAL_SCORE
        else:
            score += weight * value
            known_weight += weight
    return score / total_weight, known_weight / total_weight


def score_listing(listing, preferences, weights=None):
    if weights is None:
        weights = weights_from_ranking(preferences.get("ranking"))
    active = active_components(preferences)
    values = {name: component_score(name, listing, preferences) for name in active}
    if not values:
        return {"match_score": None, "match_confidence": None, "components": {}, "unknown_components": []}
    score, confidence = combine(values, weights)
    total_weight = sum(weights[name] for name in values)
    components = {
        name: {
            "label": COMPONENT_LABELS[name],
            "score": None if value is None else round(value, 3),
            "known": value is not None,
            "weight": round(weights[name] / total_weight, 3),
        }
        for name, value in values.items()
    }
    return {
        "match_score": round(score, 3),
        "match_confidence": round(confidence, 3),
        "components": components,
        "unknown_components": [name for name, value in values.items() if value is None],
    }


def unknown_fields(listing):
    return [
        name
        for name, columns in UNKNOWN_FIELD_CHECKS.items()
        if any(is_missing(listing.get(column)) for column in columns)
    ]
