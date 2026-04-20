from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from nyc_taxi_forecast.evaluate import regression_metrics

FEATURE_COLUMNS = (
    "zone_id",
    "hour",
    "dayofweek",
    "pickup_count",
    "pickup_count_lag1",
    "pickup_count_lag168",
)


@dataclass(frozen=True)
class TrainEvalResult:
    model: HistGradientBoostingRegressor
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    n_train: int
    n_test: int


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    # Numeric-only features keep the baseline robust to unseen categorical levels at test time.
    return frame[list(FEATURE_COLUMNS)].astype("float64")


def _build_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=10,
        learning_rate=0.06,
        max_iter=250,
        random_state=0,
    )


def train_baseline_model(
    panel: pd.DataFrame,
    *,
    split_time: pd.Timestamp,
) -> TrainEvalResult:
    """Train a tree baseline on rows with hour_start < split_time; evaluate on holdout."""
    if split_time.tzinfo is None:
        msg = "split_time must be timezone-aware"
        raise ValueError(msg)

    missing = [c for c in (*FEATURE_COLUMNS, "pickup_count_next_hour", "hour_start") if c not in panel.columns]
    if missing:
        msg = f"panel missing columns: {missing}"
        raise ValueError(msg)

    train_mask = panel["hour_start"] < split_time
    test_mask = panel["hour_start"] >= split_time

    train = panel.loc[train_mask].reset_index(drop=True)
    test = panel.loc[test_mask].reset_index(drop=True)
    if train.empty or test.empty:
        msg = "Train or test split is empty; adjust split_time."
        raise ValueError(msg)

    x_train = _prepare_features(train)
    y_train = train["pickup_count_next_hour"].to_numpy(dtype=float)
    x_test = _prepare_features(test)
    y_test = test["pickup_count_next_hour"].to_numpy(dtype=float)

    model = _build_model()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    train_metrics = regression_metrics(y_train, pred_train)
    test_metrics = regression_metrics(y_test, pred_test)
    naive = naive_mean_baseline(y_train, y_test)
    test_metrics["naive_mean_mae"] = naive["mae"]
    test_metrics["naive_mean_rmse"] = naive["rmse"]

    return TrainEvalResult(
        model=model,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        n_train=int(len(train)),
        n_test=int(len(test)),
    )


def attach_predictions(panel: pd.DataFrame, model: HistGradientBoostingRegressor) -> pd.DataFrame:
    out = panel.copy()
    out["pred_pickup_count_next_hour"] = model.predict(_prepare_features(out))
    return out


def naive_mean_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(y_train))
    pred = np.full_like(y_test, fill_value=mean, dtype=float)
    return regression_metrics(y_test, pred)


def predict_next_hour(
    model: HistGradientBoostingRegressor,
    *,
    zone_id: int,
    hour: int,
    dayofweek: int,
    pickup_count: float,
    pickup_count_lag1: float,
    pickup_count_lag168: float,
) -> float:
    row = pd.DataFrame(
        [
            {
                "zone_id": zone_id,
                "hour": hour,
                "dayofweek": dayofweek,
                "pickup_count": pickup_count,
                "pickup_count_lag1": pickup_count_lag1,
                "pickup_count_lag168": pickup_count_lag168,
            }
        ]
    )
    return float(model.predict(_prepare_features(row))[0])
