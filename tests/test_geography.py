import pandas as pd
import pytest

from src.geography import CAMPUS_LATITUDE, CAMPUS_LONGITUDE, campus_distances, haversine_miles


def test_same_point_is_zero():
    assert haversine_miles(41.7959, -88.0855, 41.7959, -88.0855) == pytest.approx(0)


def test_one_degree_of_latitude_is_about_69_miles():
    assert haversine_miles(41.7959, -88.0855, 42.7959, -88.0855) == pytest.approx(69.09, abs=0.1)


def test_one_degree_of_longitude_near_campus_is_about_51_miles():
    assert haversine_miles(41.7959, -88.0855, 41.7959, -87.0855) == pytest.approx(51.5, abs=0.3)


def test_chicago_to_new_york():
    assert haversine_miles(41.8781, -87.6298, 40.7128, -74.0060) == pytest.approx(712, abs=3)


def test_campus_to_chicago_loop():
    assert haversine_miles(CAMPUS_LATITUDE, CAMPUS_LONGITUDE, 41.8781, -87.6298) == pytest.approx(24.2, abs=0.5)


def test_distance_is_symmetric():
    forward = haversine_miles(41.0, -88.0, 44.9778, -93.2650)
    backward = haversine_miles(44.9778, -93.2650, 41.0, -88.0)
    assert forward == pytest.approx(backward)


def test_campus_distances_leaves_missing_coordinates_blank():
    frame = pd.DataFrame({"latitude": [CAMPUS_LATITUDE, None], "longitude": [CAMPUS_LONGITUDE, -88.0]})
    distances = campus_distances(frame)
    assert distances.iloc[0] == pytest.approx(0)
    assert pd.isna(distances.iloc[1])
