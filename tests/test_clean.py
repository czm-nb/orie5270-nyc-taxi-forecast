import pandas as pd

from nyc_taxi_forecast.clean import clean_trips


def test_clean_trips_drops_bad_zone_and_null_pickup() -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2025-01-01 00:10:00", pd.NA, "2025-01-01 01:00:00"],
            "tpep_dropoff_datetime": ["2025-01-01 00:20:00", "2025-01-01 00:05:00", "2025-01-01 01:10:00"],
            "PULocationID": [0, 10, 10],
        }
    )
    out = clean_trips(df)
    assert len(out) == 1
    assert int(out["PULocationID"].iloc[0]) == 10


def test_clean_trips_drops_outside_study_window() -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2007-12-05 18:00:00", "2025-01-01 00:10:00"],
            "tpep_dropoff_datetime": ["2007-12-05 19:00:00", "2025-01-01 00:20:00"],
            "PULocationID": [10, 10],
        }
    )
    out = clean_trips(df)
    assert len(out) == 1


def test_clean_trips_drops_dropoff_before_pickup() -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2025-01-01 00:10:00"],
            "tpep_dropoff_datetime": ["2025-01-01 00:05:00"],
            "PULocationID": [10],
        }
    )
    out = clean_trips(df)
    assert out.empty
