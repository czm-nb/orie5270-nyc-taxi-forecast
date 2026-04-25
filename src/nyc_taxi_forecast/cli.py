from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import typer

from nyc_taxi_forecast import __version__
from nyc_taxi_forecast.baseline import attach_predictions, train_baseline_model
from nyc_taxi_forecast.pipeline import build_forecasting_panel, build_hourly_pickups, week1_january_2025_sample

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("build-hourly")
def build_hourly(
    input_dir: Path = typer.Option(
        Path("data/original_data"),
        help="Directory containing yellow trip Parquet files (default: data/original_data)",
    ),
    glob_pattern: str = typer.Option("yellow_tripdata_*.parquet", help="Glob pattern under input_dir"),
    output: Path = typer.Option(Path("results/hourly_pickups.parquet"), help="Full hourly panel output"),
    sample_output: Path = typer.Option(
        Path("results/hourly_pickups_week1_2025-01.parquet"),
        help="Small sample output (first week of Jan 2025, NYC local time)",
    ),
) -> None:
    """Aggregate cleaned yellow taxi trips to hourly pickup counts by TLC zone."""
    hourly = build_hourly_pickups(input_dir, glob_pattern=glob_pattern)
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output, index=False)

    sample = week1_january_2025_sample(hourly)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_output, index=False)

    typer.echo(f"Wrote {len(hourly):,} rows -> {output}")
    typer.echo(f"Wrote {len(sample):,} rows -> {sample_output}")


@app.command("train-baseline")
def train_baseline_cmd(
    hourly_path: Path = typer.Option(
        Path("results/hourly_pickups.parquet"),
        help="Sparse hourly pickup counts produced by build-hourly",
    ),
    split_date: str = typer.Option(
        "2025-03-01",
        help="Train on rows with hour_start before this America/New_York calendar date (midnight)",
    ),
    metrics_out: Path = typer.Option(Path("results/baseline_metrics.json")),
    model_out: Path = typer.Option(Path("results/baseline_model.joblib")),
    preds_out: Path = typer.Option(Path("results/baseline_test_predictions.parquet")),
) -> None:
    """Complete the hourly panel, add next-hour targets, train a baseline GBDT, and write metrics."""
    split_ts = pd.Timestamp(split_date, tz="America/New_York")
    hourly = pd.read_parquet(hourly_path, engine="pyarrow")
    panel = build_forecasting_panel(hourly)
    result = train_baseline_model(panel, split_time=split_ts)

    metrics = {
        "split_time": split_ts.isoformat(),
        "n_train": result.n_train,
        "n_test": result.n_test,
        "train_metrics": result.train_metrics,
        "test_metrics": result.test_metrics,
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(result.model, model_out)

    test_panel = panel.loc[panel["hour_start"] >= split_ts].reset_index(drop=True)
    preds = attach_predictions(test_panel, result.model)
    preds.to_parquet(preds_out, index=False)

    typer.echo(f"Wrote metrics -> {metrics_out}")
    typer.echo(f"Wrote model -> {model_out}")
    typer.echo(f"Wrote test predictions ({len(preds):,} rows) -> {preds_out}")


@app.command("serve")
def serve(
    model_path: Path = typer.Option(Path("results/baseline_model.joblib"), help="Trained baseline model path"),
    hourly_path: Path = typer.Option(
        Path("results/hourly_pickups.parquet"),
        help="Hourly pickups parquet (same as train-baseline input) so the web UI can auto-fill demand fields",
    ),
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Serve the interactive HTML prediction page (FastAPI + local model)."""
    import os

    import uvicorn

    os.environ["NYC_TAXI_MODEL_PATH"] = str(model_path.expanduser().resolve())
    os.environ["NYC_TAXI_HOURLY_PATH"] = str(hourly_path.expanduser().resolve())
    uvicorn.run(
        "nyc_taxi_forecast.web:app",
        host=host,
        port=port,
        factory=False,
        reload=False,
    )


@app.command("version")
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


def main() -> None:
    app()
