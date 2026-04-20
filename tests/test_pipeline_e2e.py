from pathlib import Path

import pandas as pd

from nyc_taxi_forecast.pipeline import build_hourly_pickups, week1_january_2025_sample


def _write_minimal_yellow_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-02 10:05:00", "2025-01-02 10:35:00", "2025-01-03 18:10:00"]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                ["2025-01-02 10:20:00", "2025-01-02 10:55:00", "2025-01-03 18:40:00"]
            ),
            "PULocationID": [7, 7, 9],
        }
    )
    df.to_parquet(path, index=False)


def test_build_hourly_pickups_merges_multiple_parquet_files(tmp_path: Path) -> None:
    _write_minimal_yellow_parquet(tmp_path / "yellow_tripdata_2025-01.parquet")
    _write_minimal_yellow_parquet(tmp_path / "yellow_tripdata_2025-02.parquet")
    hourly = build_hourly_pickups(tmp_path)
    assert {"zone_id", "hour_start", "pickup_count"} == set(hourly.columns)
    assert hourly["pickup_count"].sum() == 6


def test_week1_january_2025_sample_filters_hour_range() -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=240, freq="h", tz=tz)
    hourly = pd.DataFrame({"zone_id": 1, "hour_start": hours, "pickup_count": 1})
    sample = week1_january_2025_sample(hourly)
    assert sample["hour_start"].min() >= pd.Timestamp("2025-01-01", tz=tz)
    assert sample["hour_start"].max() < pd.Timestamp("2025-01-08", tz=tz)
