from __future__ import annotations

from pathlib import Path

import pandas as pd

from nyc_taxi_forecast.aggregate import filter_hour_range, merge_hourly_pickups, trips_to_hourly_pickups
from nyc_taxi_forecast.clean import clean_trips
from nyc_taxi_forecast.io import discover_yellow_parquet_files, read_yellow_trips
from nyc_taxi_forecast.panel import add_lag_features, add_next_hour_target, add_time_features, complete_hourly_panel


def build_hourly_pickups(
    input_dir: Path,
    *,
    glob_pattern: str = "yellow_tripdata_*.parquet",
) -> pd.DataFrame:
    paths = discover_yellow_parquet_files(input_dir, glob_pattern)
    parts: list[pd.DataFrame] = []
    for path in paths:
        raw = read_yellow_trips(path)
        cleaned = clean_trips(raw)
        parts.append(trips_to_hourly_pickups(cleaned))
    return merge_hourly_pickups(parts)


def week1_january_2025_sample(hourly: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2025-01-01", tz="America/New_York")
    end = pd.Timestamp("2025-01-08", tz="America/New_York")
    return filter_hour_range(hourly, start, end)


def build_forecasting_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    """Dense hourly grid per zone + next-hour target + calendar + lag features."""
    dense = complete_hourly_panel(hourly)
    labeled = add_next_hour_target(dense)
    timed = add_time_features(labeled)
    return add_lag_features(timed)
