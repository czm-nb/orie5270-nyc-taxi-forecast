from __future__ import annotations

import pandas as pd


def complete_hourly_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    """Expand sparse (zone_id, hour_start) counts to a dense regular hourly grid per zone."""
    if hourly.empty:
        return hourly

    required = {"zone_id", "hour_start", "pickup_count"}
    if not required.issubset(hourly.columns):
        msg = f"hourly must contain columns {sorted(required)}"
        raise ValueError(msg)

    min_h = hourly["hour_start"].min()
    max_h = hourly["hour_start"].max()
    zones = hourly["zone_id"].drop_duplicates().sort_values()

    hours = pd.date_range(min_h, max_h, freq="h")
    idx = pd.MultiIndex.from_product([zones.to_numpy(), hours], names=["zone_id", "hour_start"])

    dense = (
        hourly.set_index(["zone_id", "hour_start"])
        .sort_index()
        .reindex(idx)
        .fillna({"pickup_count": 0})
        .reset_index()
    )
    dense["pickup_count"] = dense["pickup_count"].astype("int64")
    return dense


def add_next_hour_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Append pickup_count_next_hour = demand in the following hour for the same zone."""
    if panel.empty:
        return panel.assign(pickup_count_next_hour=pd.Series(dtype="float64"))

    df = panel.sort_values(["zone_id", "hour_start"], kind="mergesort").reset_index(drop=True)
    df["pickup_count_next_hour"] = df.groupby("zone_id", sort=False)["pickup_count"].shift(-1)
    return df[df["pickup_count_next_hour"].notna()].reset_index(drop=True)


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    local = out["hour_start"].dt.tz_convert("America/New_York")
    out["hour"] = local.dt.hour.astype("int16")
    out["dayofweek"] = local.dt.dayofweek.astype("int16")
    return out


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add same-zone demand lags on a dense, time-sorted panel (168h = 7d)."""
    if panel.empty:
        out = panel.copy()
        out["pickup_count_lag1"] = pd.Series(dtype="int64")
        out["pickup_count_lag168"] = pd.Series(dtype="int64")
        return out

    df = panel.sort_values(["zone_id", "hour_start"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("zone_id", sort=False)["pickup_count"]
    df["pickup_count_lag1"] = g.shift(1).fillna(0).astype("int64")
    df["pickup_count_lag168"] = g.shift(168).fillna(0).astype("int64")
    return df
