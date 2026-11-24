import pandas as pd
import pytest

from scripts.clean_data import PRICE_CEILING, PRICE_FLOOR, mark_rejections, monthly_prices, repost_duplicates


def listing(**overrides):
    row = {
        "id": "1",
        "title": "One BR in Lisle",
        "body": "Nice place.",
        "price": "1000",
        "price_type": "Monthly",
        "bedrooms": "1",
        "bathrooms": "1",
        "square_feet": "700",
        "latitude": "41.79",
        "longitude": "-88.08",
        "source": "RentLingo",
    }
    row.update(overrides)
    return row


def test_weekly_rent_converted_to_monthly():
    frame = pd.DataFrame([listing(price="300", price_type="Weekly"), listing(id="2")])
    prices, weekly = monthly_prices(frame)
    assert prices.iloc[0] == pytest.approx(300 * 52 / 12)
    assert prices.iloc[1] == 1000
    assert weekly.tolist() == [True, False]


def test_repost_of_same_unit_detected_by_title_or_body():
    frame = pd.DataFrame([
        listing(id="1"),
        listing(id="2", source="RentDigs.com"),
        listing(id="3", title="Different title", body="Nice place."),
        listing(id="4", title="Another title", body="Different body"),
        listing(id="5", price="1100"),
    ])
    assert repost_duplicates(frame).tolist() == [False, True, True, False, False]


def test_rejection_reasons():
    frame = pd.DataFrame([
        listing(id="1"),
        listing(id="1"),
        listing(id="3", price="150", square_feet="701"),
        listing(id="4", price=str(PRICE_CEILING + 1), square_feet="702"),
        listing(id="5", price=None, square_feet="703"),
        listing(id="6", price_type="Monthly|Weekly", square_feet="704"),
        listing(id="7", price="5000", price_type="Weekly", square_feet="705"),
        listing(id="8", price=str(PRICE_FLOOR), square_feet="706"),
    ])
    reasons = mark_rejections(frame).tolist()
    assert reasons[0] is pd.NA
    assert reasons[1] == "exact_duplicate"
    assert reasons[2] == "implausible_price"
    assert reasons[3] == "implausible_price"
    assert reasons[4] == "missing_price"
    assert reasons[5] == "ambiguous_price_period"
    assert reasons[6] == "implausible_price"
    assert reasons[7] is pd.NA
