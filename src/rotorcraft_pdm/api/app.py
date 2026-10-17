# =============================================================
# WHAT THIS FILE DOES:
# The prediction HTTP service. A caller sends recent engine
# cycles or a vibration summary and gets a remaining-life
# interval, an anomaly score, or the fleet table.
#
# HOW IT CONNECTS TO OTHER FILES:
# - models/rul.py produced the bundle this file loads
# - models/decision.py turns the interval into an action
# - dashboard/app.py can call this service, or read the same files
#
# RUN IT:
#   uvicorn rotorcraft_pdm.api.app:app --host 127.0.0.1 --port 8000
# =============================================================

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rotorcraft_pdm.models.decision import recommend_action
from rotorcraft_pdm.monitoring import score_drift
from rotorcraft_pdm.settings import MODEL_DIR, REPORTS_DIR

app = FastAPI(
    title="Rotorcraft predictive maintenance (proof of concept)",
    description=(
        "Public-data proof of concept. Not affiliated with the U.S. Army or Boeing. "
        "Not an airworthiness authority."
    ),
    version="0.1.0",
)


class EngineCycle(BaseModel):
    # One cycle of the sensors the model was trained on, already named.
    values: dict[str, float]


class RulRequest(BaseModel):
    # The last few cycles, oldest first. The service uses the latest row
    # for the classical model. Sequence models need the saved window length.
    cycles: list[EngineCycle] = Field(min_length=1)


class AnomalyRequest(BaseModel):
    # A feature vector from features/vibration.py, in that column order.
    features: list[float] = Field(min_length=1)


class DriftRequest(BaseModel):
    # Rows of the sensors stored in the drift baseline, same order.
    rows: list[list[float]] = Field(min_length=1)


def _load_rul():
    path = MODEL_DIR / "rul_bundle.joblib"
    if not path.exists():
        raise HTTPException(status_code=503, detail="RUL model is not trained yet.")
    return joblib.load(path)


def _vector_from_request(bundle, request: RulRequest):
    latest = request.cycles[-1].values
    missing = [column for column in bundle["feature_columns"] if column not in latest]
    # Rolling columns are optional at the HTTP boundary. If the caller sends
    # only the raw sensors, we fill the rolling columns with the raw value.
    # That is a degraded request, and the response says so.
    row = {}
    degraded = False
    for column in bundle["feature_columns"]:
        if column in latest:
            row[column] = latest[column]
        elif column.rsplit("_", 2)[0] in latest:
            row[column] = latest[column.rsplit("_", 2)[0]]
            degraded = True
        else:
            missing.append(column)
    if any(column not in row for column in bundle["feature_columns"]):
        raise HTTPException(
            status_code=422,
            detail=f"Missing columns: {sorted(set(missing))}",
        )
    frame = pd.DataFrame([row])[bundle["feature_columns"]]
    return frame, degraded


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rul_model_present": (MODEL_DIR / "rul_bundle.joblib").exists(),
        "disclaimer": "Proof of concept. Not an airworthiness release.",
    }


@app.post("/rul")
def predict_rul(request: RulRequest):
    bundle = _load_rul()
    frame, degraded = _vector_from_request(bundle, request)
    point = float(np.clip(bundle["model"].predict(frame)[0], 0, None))
    low = float(bundle["quantiles"]["low"].predict(frame)[0])
    high = float(bundle["quantiles"]["high"].predict(frame)[0])
    if low > high:
        low, high = high, low
    action = recommend_action(low, point)
    return {
        "model": bundle["winner"],
        "rul_cycles": point,
        "lower_cycles": low,
        "upper_cycles": high,
        "degraded_features": degraded,
        **action,
    }


@app.post("/anomaly")
def predict_anomaly(request: AnomalyRequest):
    path = MODEL_DIR / "isolation_forest.joblib"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Anomaly model is not trained yet.")
    from rotorcraft_pdm.models.anomaly import baseline_scores

    bundle = joblib.load(path)
    width = bundle["scaler"].n_features_in_
    if len(request.features) != width:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {width} features, got {len(request.features)}.",
        )
    scores = baseline_scores(bundle, np.asarray([request.features], dtype=float))
    return {
        "anomaly_score": float(scores[0]),
        "disclaimer": "Higher means less like the healthy training flights.",
    }


@app.post("/drift")
def predict_drift(request: DriftRequest):
    path = MODEL_DIR / "drift_baseline.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Drift baseline is not saved yet.")
    import json

    baseline = json.loads(path.read_text(encoding="utf-8"))
    return score_drift(baseline, np.asarray(request.rows, dtype=float))


@app.get("/fleet")
def fleet_status():
    path = REPORTS_DIR / "fleet.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Fleet simulation has not been run.")
    table = pd.read_csv(path)
    return {"aircraft": table.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
