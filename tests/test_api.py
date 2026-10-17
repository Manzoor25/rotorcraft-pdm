# =============================================================
# WHAT THIS FILE DOES:
# Checks the HTTP contract with a tiny in-memory stand-in for
# the trained bundle. It does not load the real model file.
# =============================================================

import numpy as np
from fastapi.testclient import TestClient

from rotorcraft_pdm.api import app as app_module
from rotorcraft_pdm.api.app import app


class _StubModel:
    def predict(self, frame):
        return np.asarray([30.0])


def test_health_does_not_require_a_model():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "airworthiness" in response.json()["disclaimer"]


def test_rul_rejects_a_missing_column(tmp_path, monkeypatch):
    bundle = {
        "model": _StubModel(),
        "quantiles": {"low": _StubModel(), "high": _StubModel()},
        "feature_columns": ["sensor_2", "sensor_2_mean_5"],
        "winner": "stub",
    }
    monkeypatch.setattr(app_module, "_load_rul", lambda: bundle)
    client = TestClient(app)
    response = client.post("/rul", json={"cycles": [{"values": {"setting_1": 1.0}}]})
    assert response.status_code == 422
