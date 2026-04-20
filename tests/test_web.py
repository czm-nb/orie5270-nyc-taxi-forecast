import importlib
from pathlib import Path

import joblib
import pandas as pd
import pytest
from starlette.testclient import TestClient

from nyc_taxi_forecast.baseline import train_baseline_model
from nyc_taxi_forecast.pipeline import build_forecasting_panel


def test_static_background_and_taxi_icon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=220, freq="h", tz=tz)
    rows = [{"zone_id": 9, "hour_start": h, "pickup_count": int(h.hour % 7)} for h in hours]
    panel = build_forecasting_panel(pd.DataFrame(rows))
    split = pd.Timestamp("2025-01-05", tz=tz)
    model = train_baseline_model(panel, split_time=split).model

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("NYC_TAXI_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NYC_TAXI_HOURLY_PATH", str(tmp_path / "missing_hourly.parquet"))

    import nyc_taxi_forecast.web as web

    importlib.reload(web)

    with TestClient(web.app) as client:
        bg = client.get("/static/nyc-background.jpg")
        icon = client.get("/static/taxi-icon.svg")
        zones = client.get("/static/taxi_zones.geojson")
    assert bg.status_code == 200
    assert bg.headers.get("content-type", "").startswith("image/")
    assert len(bg.content) > 10_000
    assert icon.status_code == 200
    assert "svg" in icon.headers.get("content-type", "")
    assert zones.status_code == 200
    assert len(zones.content) > 100_000
    assert b'"LocationID"' in zones.content


def test_index_page_has_no_jinja_leakage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=220, freq="h", tz=tz)
    rows = [{"zone_id": 9, "hour_start": h, "pickup_count": int(h.hour % 7)} for h in hours]
    panel = build_forecasting_panel(pd.DataFrame(rows))
    split = pd.Timestamp("2025-01-05", tz=tz)
    model = train_baseline_model(panel, split_time=split).model

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("NYC_TAXI_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NYC_TAXI_HOURLY_PATH", str(tmp_path / "missing_hourly.parquet"))

    import nyc_taxi_forecast.web as web

    importlib.reload(web)

    with TestClient(web.app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "{{" not in resp.text
    assert "{%" not in resp.text


def test_api_predict_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=220, freq="h", tz=tz)
    rows = [{"zone_id": 9, "hour_start": h, "pickup_count": int(h.hour % 7)} for h in hours]
    panel = build_forecasting_panel(pd.DataFrame(rows))
    split = pd.Timestamp("2025-01-05", tz=tz)
    model = train_baseline_model(panel, split_time=split).model

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("NYC_TAXI_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NYC_TAXI_HOURLY_PATH", str(tmp_path / "missing_hourly.parquet"))

    import nyc_taxi_forecast.web as web

    importlib.reload(web)

    with TestClient(web.app) as client:
        meta = client.get("/api/meta")
        assert meta.status_code == 200
        mj = meta.json()
        assert "model_path" in mj
        assert mj.get("auto_pickup_counts") is False
        assert mj.get("hourly_pickups_path") in (None, "")
        err = mj.get("hourly_load_error") or ""
        assert isinstance(err, str) and len(err) > 5

        resp = client.post(
            "/api/predict",
            json={
                "zone_id": 9,
                "hour": 14,
                "dayofweek": 2,
                "pickup_count": 5.0,
                "pickup_count_lag1": 4.0,
                "pickup_count_lag168": 3.0,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))


def test_api_predict_error_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=220, freq="h", tz=tz)
    rows = [{"zone_id": 9, "hour_start": h, "pickup_count": int(h.hour % 7)} for h in hours]
    panel = build_forecasting_panel(pd.DataFrame(rows))
    split = pd.Timestamp("2025-01-05", tz=tz)
    model = train_baseline_model(panel, split_time=split).model

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("NYC_TAXI_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NYC_TAXI_HOURLY_PATH", str(tmp_path / "missing_hourly.parquet"))

    import nyc_taxi_forecast.web as web

    importlib.reload(web)

    def _boom(*_args: object, **_kwargs: object) -> float:
        raise RuntimeError("bad predict")

    monkeypatch.setattr(web, "predict_next_hour", _boom)

    with TestClient(web.app) as client:
        resp = client.post(
            "/api/predict",
            json={
                "zone_id": 9,
                "hour": 14,
                "dayofweek": 2,
                "pickup_count": 5.0,
                "pickup_count_lag1": 4.0,
                "pickup_count_lag168": 3.0,
            },
        )
    assert resp.status_code == 400
    assert "bad predict" in resp.json().get("error", "")


def test_api_meta_loads_hourly_next_to_model_when_env_path_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If NYC_TAXI_HOURLY_PATH is wrong but hourly_pickups.parquet sits next to the model, still enable auto counts."""
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=260, freq="h", tz=tz)
    rows = [{"zone_id": 9, "hour_start": h, "pickup_count": int(h.hour % 7) + 1} for h in hours]
    panel = build_forecasting_panel(pd.DataFrame(rows))
    split = pd.Timestamp("2025-01-05", tz=tz)
    model = train_baseline_model(panel, split_time=split).model
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    pd.DataFrame(rows).to_parquet(tmp_path / "hourly_pickups.parquet", index=False)

    monkeypatch.setenv("NYC_TAXI_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NYC_TAXI_HOURLY_PATH", str(tmp_path / "nonexistent.parquet"))

    import nyc_taxi_forecast.web as web

    importlib.reload(web)

    with TestClient(web.app) as client:
        meta = client.get("/api/meta").json()
        assert meta.get("auto_pickup_counts") is True
        assert meta.get("default_scenario_date")
        assert meta.get("default_zone_id") == 9
        assert meta.get("hourly_load_error") in (None, "")
        assert meta.get("hourly_pickups_path")


def test_api_pickup_context_from_hourly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=260, freq="h", tz=tz)
    rows = [{"zone_id": 9, "hour_start": h, "pickup_count": int(h.hour % 7) + 1} for h in hours]
    hourly_path = tmp_path / "hourly.parquet"
    pd.DataFrame(rows).to_parquet(hourly_path, index=False)

    panel = build_forecasting_panel(pd.DataFrame(rows))
    split = pd.Timestamp("2025-01-05", tz=tz)
    model = train_baseline_model(panel, split_time=split).model
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    monkeypatch.setenv("NYC_TAXI_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NYC_TAXI_HOURLY_PATH", str(hourly_path))

    import nyc_taxi_forecast.web as web

    importlib.reload(web)

    with TestClient(web.app) as client:
        meta = client.get("/api/meta").json()
        assert meta.get("auto_pickup_counts") is True
        assert meta.get("default_scenario_date")
        assert meta.get("default_scenario_hour") is not None
        assert meta.get("default_dayofweek") is not None
        assert meta.get("default_zone_id") == 9
        assert meta.get("hourly_pickups_path")
        assert meta.get("hourly_load_error") in (None, "")
        ok0 = client.get(
            "/api/pickup-context",
            params={
                "zone_id": 9,
                "scenario_date": meta["default_scenario_date"],
                "hour": meta["default_scenario_hour"],
            },
        )
        assert ok0.status_code == 200

        ok = client.get(
            "/api/pickup-context",
            params={"zone_id": 9, "scenario_date": "2025-01-10", "hour": 14},
        )
        assert ok.status_code == 200
        body = ok.json()
        assert body.get("ok") is True
        assert "pickup_count" in body
        assert "pickup_count_lag1" in body
        assert "pickup_count_lag168" in body
        assert body["dayofweek"] == 4  # Friday

        missing = client.get(
            "/api/pickup-context",
            params={"zone_id": 9, "scenario_date": "2030-01-01", "hour": 0},
        )
        assert missing.status_code == 404


def test_predict_page_missing_model_errors_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "nope.joblib"
    monkeypatch.setenv("NYC_TAXI_MODEL_PATH", str(missing))

    import nyc_taxi_forecast.web as web

    importlib.reload(web)

    with pytest.raises(FileNotFoundError):
        with TestClient(web.app):
            pass
