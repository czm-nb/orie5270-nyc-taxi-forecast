import pandas as pd
import pytest

from nyc_taxi_forecast.baseline import attach_predictions, train_baseline_model
from nyc_taxi_forecast.pipeline import build_forecasting_panel


def test_train_baseline_model_requires_tz_aware_split() -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=120, freq="h", tz=tz)
    rows = [{"zone_id": 1, "hour_start": h, "pickup_count": int(h.hour % 5)} for h in hours]
    panel = build_forecasting_panel(pd.DataFrame(rows))
    with pytest.raises(ValueError):
        train_baseline_model(panel, split_time=pd.Timestamp("2025-01-05"))


def test_train_baseline_model_requires_feature_columns() -> None:
    panel = pd.DataFrame({"hour_start": [pd.Timestamp("2025-01-01", tz="America/New_York")], "pickup_count_next_hour": [1.0]})
    with pytest.raises(ValueError):
        train_baseline_model(panel, split_time=pd.Timestamp("2025-01-02", tz="America/New_York"))


def test_attach_predictions_smoke() -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=160, freq="h", tz=tz)
    rows = [{"zone_id": 2, "hour_start": h, "pickup_count": int(h.hour % 4)} for h in hours]
    panel = build_forecasting_panel(pd.DataFrame(rows))
    split = pd.Timestamp("2025-01-05", tz=tz)
    model = train_baseline_model(panel, split_time=split).model
    scored = attach_predictions(panel.head(50), model)
    assert "pred_pickup_count_next_hour" in scored.columns
