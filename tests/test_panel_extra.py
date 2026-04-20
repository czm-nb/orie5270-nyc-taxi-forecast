import pandas as pd
import pytest

from nyc_taxi_forecast.panel import (
    add_lag_features,
    add_next_hour_target,
    complete_hourly_panel,
)


def test_complete_hourly_panel_empty() -> None:
    df = pd.DataFrame(columns=["zone_id", "hour_start", "pickup_count"])
    out = complete_hourly_panel(df)
    assert len(out) == 0


def test_complete_hourly_panel_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        complete_hourly_panel(pd.DataFrame({"zone_id": [1]}))


def test_add_next_hour_target_empty() -> None:
    df = pd.DataFrame(columns=["zone_id", "hour_start", "pickup_count"])
    out = add_next_hour_target(df)
    assert "pickup_count_next_hour" in out.columns
    assert len(out) == 0


def test_add_lag_features_empty() -> None:
    df = pd.DataFrame(columns=["zone_id", "hour_start", "pickup_count"])
    out = add_lag_features(df)
    assert set(out.columns) == {
        "zone_id",
        "hour_start",
        "pickup_count",
        "pickup_count_lag1",
        "pickup_count_lag168",
    }
    assert len(out) == 0
