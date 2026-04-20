import pandas as pd
import pytest

from nyc_taxi_forecast.aggregate import filter_hour_range, merge_hourly_pickups, trips_to_hourly_pickups
from nyc_taxi_forecast.clean import clean_trips


def test_trips_to_hourly_pickups_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        trips_to_hourly_pickups(pd.DataFrame({"tpep_pickup_datetime": [1]}))


def test_trips_to_hourly_pickups_tz_aware_pickup_converts() -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-02 15:00:00"]).tz_localize("UTC"),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-01-02 15:20:00"]).tz_localize("UTC"),
            "PULocationID": [12],
        }
    )
    cleaned = clean_trips(df)
    hourly = trips_to_hourly_pickups(cleaned)
    assert hourly["hour_start"].dt.tz is not None


def test_merge_hourly_pickups_empty_parts() -> None:
    out = merge_hourly_pickups([])
    assert list(out.columns) == ["zone_id", "hour_start", "pickup_count"]
    assert len(out) == 0


def test_filter_hour_range_empty_df() -> None:
    df = pd.DataFrame(columns=["zone_id", "hour_start", "pickup_count"])
    start = pd.Timestamp("2025-01-01", tz="America/New_York")
    end = pd.Timestamp("2025-01-02", tz="America/New_York")
    out = filter_hour_range(df, start, end)
    assert len(out) == 0


def test_filter_hour_range_naive_bounds_raises() -> None:
    df = pd.DataFrame(
        {
            "zone_id": [1],
            "hour_start": [pd.Timestamp("2025-01-01 00:00:00", tz="America/New_York")],
            "pickup_count": [1],
        }
    )
    with pytest.raises(ValueError):
        filter_hour_range(df, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02", tz="America/New_York"))
