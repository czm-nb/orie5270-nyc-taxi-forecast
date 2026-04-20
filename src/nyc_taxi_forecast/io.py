from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_COLUMNS = ("tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID")


def discover_yellow_parquet_files(input_dir: Path, pattern: str = "yellow_tripdata_*.parquet") -> list[Path]:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        msg = f"No parquet files matching {pattern!r} under {input_dir}"
        raise FileNotFoundError(msg)
    return paths


def read_yellow_trips(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    cols = list(columns or DEFAULT_COLUMNS)
    return pd.read_parquet(path, columns=cols, engine="pyarrow")
