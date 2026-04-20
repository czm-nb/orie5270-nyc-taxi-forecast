# NYC Yellow Taxi — hourly pickup demand (scaffold)

This repo contains a small Python package that turns TLC Yellow Taxi Parquet files into **hourly pickup counts by TLC taxi zone** (`PULocationID`), using **America/New_York** hour buckets.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Download **Yellow Taxi** monthly Parquet from the TLC trip record page (search for “TLC trip record data” / [NYC TLC data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)), then put `yellow_tripdata_2025-*.parquet` in the project root (or point `build-hourly --input-dir` elsewhere). **Those raw files are not stored in Git** (too large for GitHub); each clone must add them locally before `build-hourly`.

## Build the hourly panel

```bash
nyc-taxi-forecast build-hourly --input-dir .
```

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

The UI is **plain HTML + JavaScript** (no Jinja): NYC night-sky styling, taxi-yellow accents, a skyline silhouette, and inline SVG icons on key inputs. The browser calls **`POST /api/predict`** (JSON) and **`GET /api/meta`** / **`GET /api/pickup-context`** for setup. Use the server URL below—**do not open `predict.html` from Finder** (`file://`), or API calls will not reach your app.

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

## Reliability: what you can honestly claim

**Pipeline / code reliability (high for a course project):** the path from raw Parquet → hourly counts → dense panel → labels → train/test split is **deterministic**, **unit-tested**, and **reproducible** if you pin Python dependency versions and keep the same raw inputs.

**Forecasting accuracy / scientific reliability (modest):** the shipped model is a **strong baseline**, not a state-of-the-art spatio-temporal forecaster.

- **Evidence it beats a silly baseline:** `results/baseline_metrics.json` compares test **MAE/RMSE** against a **global-mean** naive predictor on the same holdout.
- **Important limitations for your write-up:**
  - Predictions can be **fractional** (regression output); counts should be interpreted as **expected demand**, then optionally rounded.
  - The web UI **auto-fills** demand and lags when the server loads `hourly_pickups.parquet`; without it you can still predict using **manual** inputs (zeros allowed, but outputs may be **weak or misleading** if the numbers are not grounded in real demand).
  - **No spatial coupling across zones** (each row is still “tabular zone features + lags”), so it will miss true city-wide shock patterns unless they are already encoded in the lags/current count.
  - **No uncertainty intervals** (no conformal / quantile / probabilistic model).

If your rubric asks for “reliability,” separate **software correctness** (tests, leakage-safe split definition, documented filters) from **model performance** (metrics on holdout + limitations).
