from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingRegressor

from nyc_taxi_forecast.baseline import predict_next_hour
from nyc_taxi_forecast.pipeline import build_forecasting_panel

log = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_INDEX = _STATIC_DIR / "predict.html"
_NY_TZ = "America/New_York"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _hourly_path_from_env() -> Path:
    raw = os.environ.get("NYC_TAXI_HOURLY_PATH", "results/hourly_pickups.parquet")
    return Path(raw).expanduser().resolve()


def _hour_start_as_ny(s: pd.Series) -> pd.Series:
    if s.dt.tz is None:
        return s.dt.tz_localize(_NY_TZ, ambiguous="infer", nonexistent="shift_forward")
    return s.dt.tz_convert(_NY_TZ)


def _parse_scenario_hour_start(scenario_date: str, hour: int) -> pd.Timestamp:
    if not _DATE_RE.match(scenario_date):
        msg = "scenario_date must be YYYY-MM-DD"
        raise ValueError(msg)
    d = date.fromisoformat(scenario_date)
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hour, tz=_NY_TZ)


def _try_build_panel_from_parquet(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, f"file not found: {path}"
    try:
        hourly = pd.read_parquet(path, engine="pyarrow")
        return build_forecasting_panel(hourly), None
    except Exception as exc:  # noqa: BLE001 — startup: skip bad file
        return None, f"{path}: {type(exc).__name__}: {exc}"[:500]


def _resolve_hourly_panel(model_path: Path) -> tuple[pd.DataFrame | None, Path | None, str | None]:
    """Try env hourly path, then parquet files next to the model (full panel or week-1 sample)."""
    candidates: list[Path] = [
        _hourly_path_from_env(),
        model_path.parent / "hourly_pickups.parquet",
        model_path.parent / "hourly_pickups_week1_2025-01.parquet",
    ]
    seen: set[str] = set()
    messages: list[str] = []
    for cand in candidates:
        key = str(cand.expanduser().resolve())
        if key in seen:
            continue
        seen.add(key)
        panel, err = _try_build_panel_from_parquet(cand)
        if panel is not None:
            return panel, cand, None
        if err:
            messages.append(err)
    detail = " · ".join(messages) if messages else "no hourly parquet candidates"
    return None, None, detail


def _default_scenario_for_ui(panel: pd.DataFrame | None) -> tuple[str, int, int] | None:
    """Pick date/hour/dayofweek for the form when NY 'now' is outside the loaded hourly history."""
    if panel is None or panel.empty:
        return None
    ny_times = _hour_start_as_ny(panel["hour_start"])
    lo = ny_times.min()
    hi = ny_times.max()
    now_floor = pd.Timestamp.now(tz=_NY_TZ).floor("h")
    clamped = min(max(now_floor, lo), hi)
    return (clamped.strftime("%Y-%m-%d"), int(clamped.hour), int(clamped.dayofweek))


def _lookup_pickup_row(panel: pd.DataFrame, zone_id: int, hour_start: pd.Timestamp) -> pd.Series | None:
    hs = hour_start
    if hs.tz is None:
        hs = hs.tz_localize(_NY_TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        hs = hs.tz_convert(_NY_TZ)
    ny_times = _hour_start_as_ny(panel["hour_start"])
    mask = (panel["zone_id"] == zone_id) & (ny_times == hs)
    hit = panel.loc[mask]
    if hit.empty:
        return None
    return hit.iloc[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    raw = os.environ.get("NYC_TAXI_MODEL_PATH", "results/baseline_model.joblib")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        msg = f"Model file not found: {path}. Train first: nyc-taxi-forecast train-baseline"
        raise FileNotFoundError(msg)
    model = joblib.load(path)
    if not isinstance(model, HistGradientBoostingRegressor):
        msg = f"Expected HistGradientBoostingRegressor in {path}"
        raise TypeError(msg)
    app.state.model = model
    app.state.model_path = str(path)

    panel, hp_used, hourly_err = _resolve_hourly_panel(path)
    app.state.panel = panel
    app.state.hourly_path = str(hp_used.expanduser().resolve()) if hp_used is not None and panel is not None else None
    app.state.hourly_load_error = None if panel is not None else hourly_err
    if panel is None and hourly_err:
        log.warning("Hourly pickups not loaded: %s", hourly_err)
    yield


app = FastAPI(title="NYC yellow taxi next-hour demand (demo)", lifespan=lifespan)


class PredictRequest(BaseModel):
    zone_id: int = Field(ge=1, le=265)
    hour: int = Field(ge=0, le=23)
    dayofweek: int = Field(ge=0, le=6)
    pickup_count: float = Field(ge=0)
    pickup_count_lag1: float = Field(default=0.0, ge=0)
    pickup_count_lag168: float = Field(default=0.0, ge=0)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve a static HTML UI (no Jinja) so the page never shows raw template tags."""
    html = _STATIC_INDEX.read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/api/meta")
async def api_meta(request: Request) -> dict[str, str | bool | int | None]:
    panel: pd.DataFrame | None = getattr(request.app.state, "panel", None)
    auto = panel is not None
    out: dict[str, str | bool | int | None] = {
        "model_path": request.app.state.model_path,
        "auto_pickup_counts": auto,
        "hourly_pickups_path": getattr(request.app.state, "hourly_path", None),
        "hourly_load_error": getattr(request.app.state, "hourly_load_error", None),
        "default_zone_id": None,
        "default_scenario_date": None,
        "default_scenario_hour": None,
        "default_dayofweek": None,
    }
    if auto and panel is not None:
        out["default_zone_id"] = int(panel["zone_id"].min())
        dft = _default_scenario_for_ui(panel)
        if dft is not None:
            out["default_scenario_date"], out["default_scenario_hour"], out["default_dayofweek"] = dft
    return out


@app.get("/api/pickup-context")
async def api_pickup_context(
    request: Request,
    zone_id: int = Query(ge=1, le=265),
    scenario_date: str = Query(min_length=10, max_length=10),
    hour: int = Query(ge=0, le=23),
) -> JSONResponse:
    panel: pd.DataFrame | None = getattr(request.app.state, "panel", None)
    if panel is None or panel.empty:
        return JSONResponse(
            {"ok": False, "error": "No hourly pickup file loaded on the server (set NYC_TAXI_HOURLY_PATH)."},
            status_code=503,
        )
    try:
        hs = _parse_scenario_hour_start(scenario_date, hour)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    row = _lookup_pickup_row(panel, zone_id, hs)
    if row is None:
        return JSONResponse(
            {
                "ok": False,
                "error": "No training row for that zone and time (check date/hour are inside the hourly file, "
                "and not the last hour of history for that zone).",
            },
            status_code=404,
        )

    return JSONResponse(
        {
            "ok": True,
            "pickup_count": int(row["pickup_count"]),
            "pickup_count_lag1": int(row["pickup_count_lag1"]),
            "pickup_count_lag168": int(row["pickup_count_lag168"]),
            "dayofweek": int(row["dayofweek"]),
            "hour": int(row["hour"]),
            "hour_start": row["hour_start"].isoformat() if hasattr(row["hour_start"], "isoformat") else str(row["hour_start"]),
        }
    )


@app.post("/api/predict")
async def api_predict(request: Request, body: PredictRequest) -> JSONResponse:
    model: HistGradientBoostingRegressor = request.app.state.model
    try:
        pred = predict_next_hour(
            model,
            zone_id=body.zone_id,
            hour=body.hour,
            dayofweek=body.dayofweek,
            pickup_count=body.pickup_count,
            pickup_count_lag1=body.pickup_count_lag1,
            pickup_count_lag168=body.pickup_count_lag168,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"prediction": pred})


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
