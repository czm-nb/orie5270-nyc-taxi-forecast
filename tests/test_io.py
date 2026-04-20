from pathlib import Path

import pandas as pd
import pytest

from nyc_taxi_forecast.io import discover_yellow_parquet_files, read_yellow_trips


def test_discover_yellow_parquet_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "yellow_tripdata_2025-02.parquet").write_bytes(b"")
    (tmp_path / "yellow_tripdata_2025-01.parquet").write_bytes(b"")
    paths = discover_yellow_parquet_files(tmp_path)
    assert [p.name for p in paths] == ["yellow_tripdata_2025-01.parquet", "yellow_tripdata_2025-02.parquet"]


def test_discover_yellow_parquet_files_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_yellow_parquet_files(tmp_path)


def test_read_yellow_trips_default_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-02 10:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-01-02 10:15:00"]),
            "PULocationID": [10],
        }
    )
    path = tmp_path / "trips.parquet"
    df.to_parquet(path, index=False)
    out = read_yellow_trips(path)
    assert list(out.columns) == ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID"]


def test_read_yellow_trips_custom_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-02 10:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-01-02 10:15:00"]),
            "PULocationID": [10],
            "extra": [1],
        }
    )
    path = tmp_path / "trips.parquet"
    df.to_parquet(path, index=False)
    out = read_yellow_trips(path, columns=("tpep_pickup_datetime", "PULocationID"))
    assert list(out.columns) == ["tpep_pickup_datetime", "PULocationID"]
