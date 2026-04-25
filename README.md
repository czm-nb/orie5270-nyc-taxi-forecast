# NYC Yellow Taxi — hourly pickup demand by zone

This repository implements a pipeline from TLC Yellow Taxi Parquet files to **hourly pickup counts by TLC taxi zone** (`PULocationID`), using **America/New_York** hour buckets, plus a baseline next-hour demand model and a small local web UI.

## Repository layout

| Path | Purpose |
|------|---------|
| **`data/original_data/`** | TLC Yellow Taxi monthly Parquet inputs (`yellow_tripdata_*.parquet`, Jan–Mar 2025). See `data/README.md`. |
| **`analysis/`** | Optional space for notebooks or exploratory work (see `analysis/README.md`). |
| **`src/nyc_taxi_forecast/`** | Python package: I/O, cleaning, hourly aggregation, panel, baseline model, FastAPI web UI. |
| **`tests/`** | `pytest` suite. |
| **`results/`** | Generated outputs (`build-hourly`, `train-baseline`); created locally and **gitignored**. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

**Python:** 3.11 or newer (`requires-python` in `pyproject.toml`).

**Installed by the command above** (versions are lower bounds; `pip` resolves current releases):

| Role | Packages |
|------|----------|
| **Core pipeline & model** | `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `joblib`, `typer` |
| **Web UI** | `fastapi`, `uvicorn[standard]` |
| **Development / tests** (the `[dev]` extra) | `pytest`, `httpx`, `pytest-cov` |

For a minimal install without test tooling, use `pip install -e .` instead of `pip install -e ".[dev]"`.

**Included in this repo:** the same three Parquet files under **`data/original_data/`** (~190 MB total) so `build-hourly` can run without a separate download. For other months or updates, download from the TLC trip record page ([NYC TLC data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)) into that folder (or pass `--input-dir` to another directory).

## Build the hourly panel

```bash
nyc-taxi-forecast build-hourly
```

(Default `--input-dir` is `data/original_data`. To use another folder: `nyc-taxi-forecast build-hourly --input-dir /path/to/parquets`.)

Outputs:

- `results/hourly_pickups.parquet` — full Jan–Mar 2025 panel (after cleaning)
- `results/hourly_pickups_week1_2025-01.parquet` — small sample slice for quick checks

## Baseline next-hour model

Requires `results/hourly_pickups.parquet` from `build-hourly`.

```bash
nyc-taxi-forecast train-baseline --split-date 2025-03-01
```

(`--hourly-path` defaults to `results/hourly_pickups.parquet`; pass it explicitly if your file lives elsewhere.)

The training table includes **calendar features** (`hour`, `dayofweek`) plus **same-zone lags** (`pickup_count_lag1`, `pickup_count_lag168`).

Writes:

- `results/baseline_metrics.json`
- `results/baseline_model.joblib`
- `results/baseline_test_predictions.parquet` (March 2025 holdout rows with predictions)

## Interactive HTML predictor (local)

The UI is **HTML and JavaScript** served by FastAPI (no Jinja templates). The browser calls **`POST /api/predict`** (JSON) and **`GET /api/meta`** / **`GET /api/pickup-context`** for configuration. **Serve the app with Uvicorn** and open the URL below; opening the static page via a **`file://`** URL will not reach the API.

Run **`serve` from the project root** (so `results/…` paths resolve). The CLI **defaults** `--hourly-path` to `results/hourly_pickups.parquet` (same default as `train-baseline`), so the usual command is:

```bash
nyc-taxi-forecast serve --model-path results/baseline_model.joblib --host 127.0.0.1 --port 8000
```

To point at the small week-one slice instead:

```bash
nyc-taxi-forecast serve --model-path results/baseline_model.joblib --hourly-path results/hourly_pickups_week1_2025-01.parquet
```

If you start **`uvicorn`** yourself, set **`NYC_TAXI_MODEL_PATH`** and **`NYC_TAXI_HOURLY_PATH`** to absolute paths (or run from the repo so `results/hourly_pickups.parquet` exists).

Then open **`http://127.0.0.1:8000/`** in your browser (same host/port you started). If you used a different port, use that instead (for example `http://127.0.0.1:8765/`).

Bundled UI assets live under **`/static/`** (NYC background JPEG, taxi SVG, TLC zone GeoJSON for the map, attributions).

The map uses **Leaflet** + **CARTO dark tiles** (loaded from the public CDN when you have internet). Panning is restricted to NYC; zone polygons cover the **five boroughs** only (see `static/ATTRIBUTIONS.txt`).

## Tests (and coverage)

`pytest` is configured (in `pyproject.toml`) to measure **line coverage on `nyc_taxi_forecast`** and **fail if coverage drops below 80%**.

```bash
pytest
```

## Scope and limitations

**Pipeline and software:** Raw TLC Parquet → hourly aggregates → dense panel → labels → train/test split is **deterministic** and covered by **`pytest`** under `tests/`. Reproducibility assumes the same input files and a consistent Python environment (see `pyproject.toml`).

**Model:** The baseline is a gradient-boosted regressor on tabular zone–time features with same-zone lags; it is **not** a full spatio-temporal citywide model. After training, **`results/baseline_metrics.json`** reports test **MAE/RMSE** relative to a **global-mean** naive baseline on the same holdout. Regression outputs may be **fractional**; treat them as expected demand and round if reporting integer counts.

**Web UI:** When **`hourly_pickups.parquet`** is loaded on the server, demand and lag fields are filled from that file. If it is not loaded, values may be entered manually (non-negative; zeros allowed); forecasts depend on the quality of those inputs.

**Not modeled:** Direct cross-zone coupling beyond what appears in the chosen features, and **no** predictive uncertainty intervals (e.g. quantiles or conformal bands).
