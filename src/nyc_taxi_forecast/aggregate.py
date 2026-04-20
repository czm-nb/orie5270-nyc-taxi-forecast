from __future__ import annotations

import pandas as pd


def trips_to_hourly_pickups(df: pd.DataFrame) -> pd.DataFrame:
    if "tpep_pickup_datetime" not in df.columns or "PULocationID" not in df.columns:
        msg = "Expected columns tpep_pickup_datetime and PULocationID"
        raise ValueError(msg)

    pickup = df["tpep_pickup_datetime"]
    if getattr(pickup.dt, "tz", None) is None:
        pickup = pickup.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    else:
        pickup = pickup.dt.tz_convert("America/New_York")

    tmp = pd.DataFrame({"zone_id": df["PULocationID"].astype("int64"), "hour_start": pickup.dt.floor("h")})
    tmp = tmp[tmp["hour_start"].notna()]

    grouped = tmp.groupby(["zone_id", "hour_start"], sort=True).size().rename("pickup_count").reset_index()
    return grouped


def merge_hourly_pickups(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["zone_id", "hour_start", "pickup_count"])
    stacked = pd.concat(parts, ignore_index=True)
    return (
        stacked.groupby(["zone_id", "hour_start"], sort=True)["pickup_count"]
        .sum()
        .reset_index()
        .sort_values(["hour_start", "zone_id"])
        .reset_index(drop=True)
    )


def filter_hour_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    if start.tzinfo is None or end.tzinfo is None:
        msg = "start and end must be timezone-aware timestamps"
        raise ValueError(msg)
    mask = (df["hour_start"] >= start) & (df["hour_start"] < end)
    return df.loc[mask].reset_index(drop=True)
