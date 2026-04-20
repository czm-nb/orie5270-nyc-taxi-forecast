import pandas as pd
import pytest

from nyc_taxi_forecast.baseline import _prepare_features, predict_next_hour, train_baseline_model
from nyc_taxi_forecast.pipeline import build_forecasting_panel


def test_train_baseline_model_runs_on_toy_panel() -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=320, freq="h", tz=tz)
    rows = []
    for z in (1, 2):
        for h in hours:
            rows.append({"zone_id": z, "hour_start": h, "pickup_count": int((h.hour + z) % 20)})
    hourly = pd.DataFrame(rows)
    panel = build_forecasting_panel(hourly)

    split = pd.Timestamp("2025-01-06", tz=tz)
    out = train_baseline_model(panel, split_time=split)
    assert out.n_train > 0 and out.n_test > 0
    assert "mae" in out.test_metrics


def test_predict_next_hour_matches_model_predict() -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=200, freq="h", tz=tz)
    rows = []
    for z in (3,):
        for h in hours:
            rows.append({"zone_id": z, "hour_start": h, "pickup_count": int(h.hour % 9)})
    panel = build_forecasting_panel(pd.DataFrame(rows))
    split = pd.Timestamp("2025-01-05", tz=tz)
    result = train_baseline_model(panel, split_time=split)
    model = result.model

    p = predict_next_hour(
        model,
        zone_id=3,
        hour=14,
        dayofweek=2,
        pickup_count=5.0,
        pickup_count_lag1=4.0,
        pickup_count_lag168=3.0,
    )
    row = pd.DataFrame(
        [
            {
                "zone_id": 3,
                "hour": 14,
                "dayofweek": 2,
                "pickup_count": 5.0,
                "pickup_count_lag1": 4.0,
                "pickup_count_lag168": 3.0,
            }
        ]
    )
    expected = float(model.predict(_prepare_features(row))[0])
    assert p == pytest.approx(expected)
