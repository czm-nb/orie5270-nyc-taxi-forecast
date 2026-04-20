from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import uvicorn
from typer.testing import CliRunner

from nyc_taxi_forecast.cli import app


def _write_minimal_yellow_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2025-01-02 10:05:00", "2025-01-02 10:35:00", "2025-01-03 18:10:00"]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                ["2025-01-02 10:20:00", "2025-01-02 10:55:00", "2025-01-03 18:40:00"]
            ),
            "PULocationID": [7, 7, 9],
        }
    )
    df.to_parquet(path, index=False)


def _write_synthetic_hourly_panel(path: Path) -> None:
    tz = "America/New_York"
    hours = pd.date_range("2025-01-01", periods=60 * 24, freq="h", tz=tz)
    hourly = pd.DataFrame({"zone_id": 3, "hour_start": hours, "pickup_count": 1})
    hourly.to_parquet(path, index=False)


def test_cli_version_prints_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_cli_build_hourly_writes_outputs(tmp_path: Path) -> None:
    _write_minimal_yellow_parquet(tmp_path / "yellow_tripdata_2025-01.parquet")
    out = tmp_path / "hourly.parquet"
    sample = tmp_path / "sample.parquet"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-hourly",
            "--input-dir",
            str(tmp_path),
            "--output",
            str(out),
            "--sample-output",
            str(sample),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert out.is_file() and sample.is_file()


def test_cli_train_baseline_writes_artifacts(tmp_path: Path) -> None:
    hourly = tmp_path / "hourly.parquet"
    _write_synthetic_hourly_panel(hourly)
    metrics = tmp_path / "metrics.json"
    model = tmp_path / "model.joblib"
    preds = tmp_path / "preds.parquet"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train-baseline",
            "--hourly-path",
            str(hourly),
            "--split-date",
            "2025-02-10",
            "--metrics-out",
            str(metrics),
            "--model-out",
            str(model),
            "--preds-out",
            str(preds),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert metrics.is_file() and model.is_file() and preds.is_file()


def test_cli_serve_invokes_uvicorn_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model.joblib"
    model.write_bytes(b"not a real model")  # serve does not load; uvicorn is patched

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _capture_run(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(uvicorn, "run", _capture_run)

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--model-path", str(model), "--host", "127.0.0.1", "--port", "8765"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert calls
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 8765
