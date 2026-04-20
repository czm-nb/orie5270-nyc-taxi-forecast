from __future__ import annotations

import pandas as pd

# TLC location IDs are small integers; IDs outside this band are almost always bad rows.
_MIN_ZONE_ID = 1
_MAX_ZONE_ID = 265

# Keep TLC rows aligned to the project dataset window (drops rare bogus timestamps).
_STUDY_START = pd.Timestamp("2025-01-01", tz="America/New_York")
_STUDY_END = pd.Timestamp("2025-04-01", tz="America/New_York")


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    if "tpep_pickup_datetime" not in df.columns or "PULocationID" not in df.columns:
        msg = "Expected columns tpep_pickup_datetime and PULocationID"
        raise ValueError(msg)

    out = df.copy()
    out["tpep_pickup_datetime"] = pd.to_datetime(out["tpep_pickup_datetime"], errors="coerce")
    out = out[out["tpep_pickup_datetime"].notna()]

    out["PULocationID"] = pd.to_numeric(out["PULocationID"], errors="coerce")
    out = out[out["PULocationID"].notna()]
    out["PULocationID"] = out["PULocationID"].astype("int64")

    out = out[(out["PULocationID"] >= _MIN_ZONE_ID) & (out["PULocationID"] <= _MAX_ZONE_ID)]

    pickup = out["tpep_pickup_datetime"]
    if pickup.dt.tz is None:
        pickup_local = pickup.dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        pickup_local = pickup.dt.tz_convert("America/New_York")
    in_study = (pickup_local >= _STUDY_START) & (pickup_local < _STUDY_END)
    out = out.loc[in_study.fillna(False)].copy()

    if "tpep_dropoff_datetime" in out.columns:
        out["tpep_dropoff_datetime"] = pd.to_datetime(out["tpep_dropoff_datetime"], errors="coerce")
        mask = out["tpep_dropoff_datetime"].notna()
        out = out.loc[
            ~mask
            | (out.loc[mask, "tpep_dropoff_datetime"] >= out.loc[mask, "tpep_pickup_datetime"])
        ]

    return out.reset_index(drop=True)
