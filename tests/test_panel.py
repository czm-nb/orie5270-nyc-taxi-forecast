import pandas as pd

from nyc_taxi_forecast.panel import add_lag_features, add_next_hour_target, add_time_features, complete_hourly_panel


def test_complete_hourly_panel_fills_missing_hours_with_zero() -> None:
    tz = "America/New_York"
    t0 = pd.Timestamp("2025-01-01 00:00:00", tz=tz)
    t1 = pd.Timestamp("2025-01-01 02:00:00", tz=tz)
    hourly = pd.DataFrame({"zone_id": [7, 7], "hour_start": [t0, t1], "pickup_count": [3, 4]})
    dense = complete_hourly_panel(hourly)
    assert len(dense) == 3  # hours 0,1,2 for zone 7
    t_mid = t0 + pd.Timedelta(hours=1)
    mid = dense.loc[(dense["zone_id"] == 7) & (dense["hour_start"] == t_mid), "pickup_count"].iloc[0]
    assert int(mid) == 0


def test_add_next_hour_target_shifts_within_zone() -> None:
    tz = "America/New_York"
    t0 = pd.Timestamp("2025-01-01 00:00:00", tz=tz)
    t1 = pd.Timestamp("2025-01-01 01:00:00", tz=tz)
    panel = pd.DataFrame(
        {
            "zone_id": [1, 1, 2],
            "hour_start": [t0, t1, t0],
            "pickup_count": [10, 20, 5],
        }
    )
    out = add_next_hour_target(panel)
    row0 = out.loc[(out["zone_id"] == 1) & (out["hour_start"] == t0)].iloc[0]
    assert int(row0["pickup_count_next_hour"]) == 20


def test_add_lag_features_shift_within_zone() -> None:
    tz = "America/New_York"
    hours = [pd.Timestamp("2025-01-01", tz=tz) + pd.Timedelta(hours=i) for i in range(5)]
    panel = pd.DataFrame({"zone_id": [1] * 5, "hour_start": hours, "pickup_count": [1, 2, 3, 4, 5]})
    out = add_lag_features(panel)
    row2 = out.loc[out["hour_start"] == hours[2]].iloc[0]
    assert int(row2["pickup_count_lag1"]) == 2


def test_add_time_features_extracts_hour_and_dow() -> None:
    tz = "America/New_York"
    t0 = pd.Timestamp("2025-01-01 15:00:00", tz=tz)  # Wednesday in 2025? Jan 1 2025 is Wednesday
    panel = pd.DataFrame({"zone_id": [1], "hour_start": [t0], "pickup_count": [1], "pickup_count_next_hour": [2]})
    out = add_time_features(panel)
    assert int(out["hour"].iloc[0]) == 15
    assert int(out["dayofweek"].iloc[0]) == 2  # Wednesday
