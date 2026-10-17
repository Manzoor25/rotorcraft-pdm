# =============================================================
# WHAT THIS FILE DOES:
# The training run. It compares classical models, fits a small
# LSTM and a small 1D CNN, fits vibration detectors when the
# Airbus files are present, and writes the reports.
#
# HOW IT CONNECTS TO OTHER FILES:
# - data/cmapss.py loads the engine files
# - models/* do the fitting
# - reports/metrics.json is what EVALUATION.md and the dashboard read
#
# FLOW:
# 1. For each C-MAPSS subset, fit the four classical models.
# 2. Pick the winner by NASA score on the FD001 last-cycle exam.
# 3. Fit sequence models on FD001 and compare them to that winner.
# 4. If vibration files exist, fit the anomaly models.
# 5. Simulate 12 fictional aircraft.
# =============================================================

import json
import sys
from pathlib import Path

# A script is not inside the package. This line lets `python scripts/train.py`
# find rotorcraft_pdm even before the editable install is active.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import joblib
import mlflow
import numpy as np
import optuna

from rotorcraft_pdm.data.cmapss import load_subset, scale_frame
from rotorcraft_pdm.evaluation.metrics import (
    anomaly_metrics,
    interval_coverage,
    rul_metrics,
)
from rotorcraft_pdm.features.engine import (
    feature_matrix,
    last_cycle_rows,
    rolling_feature_frame,
    sequence_windows,
)
from rotorcraft_pdm.models.anomaly import (
    autoencoder_scores,
    baseline_scores,
    fit_autoencoder,
    fit_baselines,
)
from rotorcraft_pdm.models.decision import recommend_action
from rotorcraft_pdm.models.rul import build_point_models, fit_quantile_models
from rotorcraft_pdm.models.sequence import fit_sequence_model, predict_sequence
from rotorcraft_pdm.monitoring import fit_drift_baseline
from rotorcraft_pdm.settings import (
    CMAPSS_SUBSETS,
    EXPERIMENT_NAME,
    MODEL_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    RUL_CAP,
    SEQUENCE_LENGTH,
)
from rotorcraft_pdm.simulation.fleet import simulate_fleet


# --------------------------------------------------------------
# FUNCTION: _tracking_uri
# WHAT IT DOES: Points MLflow at a local SQLite file, unless Azure
#               already set an address in the environment
# WHY WE NEED IT: MLflow 2 can use a folder, but one database file
#                 is easier to find and to delete.
# INPUTS: None
# OUTPUTS: A tracking URI string
# --------------------------------------------------------------
def _tracking_uri():
    import os

    existing = os.getenv("MLFLOW_TRACKING_URI")
    if existing and existing.startswith("azureml:"):
        return existing
    database = (PROJECT_ROOT / "mlflow.db").resolve()
    return f"sqlite:///{database.as_posix()}"


# --------------------------------------------------------------
# FUNCTION: train_classical
# WHAT IT DOES: Fits four models on one subset and scores the exam
# INPUTS: subset name, such as FD001
# OUTPUTS: A list of result rows, plus the fitted winner pieces for FD001
# --------------------------------------------------------------
def train_classical(subset, small=False):
    loaded = load_subset(subset)
    train = scale_frame(loaded["train"], loaded["scaler"])
    test = scale_frame(loaded["test"], loaded["scaler"])
    train_features = rolling_feature_frame(train, loaded["kept_sensors"])
    test_features = rolling_feature_frame(test, loaded["kept_sensors"])
    x_train, y_train = feature_matrix(train_features)
    x_test, _y_test_all = feature_matrix(test_features)
    exam = last_cycle_rows(test_features)
    x_exam, y_exam = feature_matrix(exam)

    rows = []
    fitted = {}
    for name, model in build_point_models(small=small).items():
        model.fit(x_train, y_train)
        predicted = np.clip(model.predict(x_exam), 0, RUL_CAP)
        scores = rul_metrics(y_exam, predicted)
        rows.append({"subset": subset, "model": name, **scores})
        fitted[name] = model
        print(f"{subset} {name}: NASA {scores['nasa_score']:.1f} RMSE {scores['rmse']:.2f}")
    return {
        "rows": rows,
        "fitted": fitted,
        "x_train": x_train,
        "y_train": y_train,
        "x_exam": x_exam,
        "y_exam": y_exam,
        "exam": exam,
        "loaded": loaded,
        "feature_columns": list(x_train.columns),
    }


# --------------------------------------------------------------
# FUNCTION: tune_lightgbm
# WHAT IT DOES: Runs a fixed number of Optuna trials on FD001
# WHY WE NEED IT: The default leaf count is a guess. A small search
#                 checks whether a different size helps, without an
#                 open-ended hunt.
# INPUTS: the FD001 training bundle, number of trials
# OUTPUTS: The best trial's parameters and NASA score
# --------------------------------------------------------------
def tune_lightgbm(bundle, n_trials=8):
    def objective(trial):
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=200,
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15),
            random_state=42,
            verbose=-1,
        )
        model.fit(bundle["x_train"], bundle["y_train"])
        predicted = np.clip(model.predict(bundle["x_exam"]), 0, RUL_CAP)
        return rul_metrics(bundle["y_exam"], predicted)["nasa_score"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_nasa_score": study.best_value}


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    by_subset = {}
    for subset in CMAPSS_SUBSETS:
        by_subset[subset] = train_classical(subset)
        all_rows.extend(by_subset[subset]["rows"])

    fd001 = by_subset["FD001"]
    classical = [row for row in fd001["rows"]]
    winner_name = min(classical, key=lambda row: row["nasa_score"])["model"]
    print(f"FD001 classical winner by NASA score: {winner_name}")

    quantiles = fit_quantile_models(fd001["x_train"], fd001["y_train"])
    low = quantiles["low"].predict(fd001["x_exam"])
    high = quantiles["high"].predict(fd001["x_exam"])
    point = np.clip(fd001["fitted"][winner_name].predict(fd001["x_exam"]), 0, RUL_CAP)
    coverage = interval_coverage(fd001["y_exam"], low, high)

    tuned = tune_lightgbm(fd001, n_trials=8)

    sequence_rows = []
    scaled_train = scale_frame(fd001["loaded"]["train"], fd001["loaded"]["scaler"])
    scaled_test = scale_frame(fd001["loaded"]["test"], fd001["loaded"]["scaler"])
    train_w, train_y, _units, _cycles = sequence_windows(
        scaled_train, fd001["loaded"]["kept_sensors"], SEQUENCE_LENGTH
    )
    test_w, test_y, test_units, test_cycles = sequence_windows(
        scaled_test, fd001["loaded"]["kept_sensors"], SEQUENCE_LENGTH
    )
    last_cycle = scaled_test.groupby("unit")["cycle"].max().rename("last_cycle").reset_index()
    sequence_bundles = {}
    for kind in ("lstm", "cnn"):
        bundle = fit_sequence_model(kind, train_w, train_y, epochs=6, hidden=32)
        predicted = predict_sequence(bundle, test_w)
        exam_mask = []
        last_map = dict(zip(last_cycle["unit"], last_cycle["last_cycle"]))
        for unit, cycle in zip(test_units, test_cycles):
            exam_mask.append(cycle == last_map[int(unit)])
        exam_mask = np.asarray(exam_mask)
        scores = rul_metrics(test_y[exam_mask], predicted[exam_mask])
        sequence_rows.append({"subset": "FD001", "model": kind, **scores})
        sequence_bundles[kind] = bundle
        print(f"FD001 {kind}: NASA {scores['nasa_score']:.1f} RMSE {scores['rmse']:.2f}")

    anomaly_report = train_vibration()
    fleet = simulate_fleet(fd001["y_exam"], low)
    fleet["table"].to_csv(REPORTS_DIR / "fleet.csv", index=False)

    drift = fit_drift_baseline(fd001["loaded"]["train"][fd001["loaded"]["kept_sensors"]].to_numpy())
    (MODEL_DIR / "drift_baseline.json").write_text(json.dumps(drift), encoding="utf-8")

    joblib.dump(
        {
            "model": fd001["fitted"][winner_name],
            "quantiles": quantiles,
            "scaler": fd001["loaded"]["scaler"],
            "kept_sensors": fd001["loaded"]["kept_sensors"],
            "feature_columns": fd001["feature_columns"],
            "winner": winner_name,
        },
        MODEL_DIR / "rul_bundle.joblib",
    )
    for kind, bundle in sequence_bundles.items():
        import torch

        torch.save(bundle["model"].state_dict(), MODEL_DIR / f"{kind}.pt")

    sample = recommend_action(float(low[0]), float(point[0]))
    report = {
        "classical": all_rows,
        "sequence": sequence_rows,
        "fd001_winner": winner_name,
        "interval_coverage_10_90": coverage,
        "optuna_lightgbm": tuned,
        "anomaly": anomaly_report,
        "fleet": {key: value for key, value in fleet.items() if key != "table"},
        "sample_action": sample,
        "dropped_sensors": {
            subset: by_subset[subset]["loaded"]["dropped_sensors"] for subset in CMAPSS_SUBSETS
        },
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name="fd001-summary"):
        mlflow.log_param("winner", winner_name)
        mlflow.log_metric("interval_coverage", coverage)
        winner_row = next(row for row in classical if row["model"] == winner_name)
        mlflow.log_metric("nasa_score", winner_row["nasa_score"])
        mlflow.log_metric("rmse", winner_row["rmse"])
    print(json.dumps(report["fleet"], indent=2))
    print(f"Wrote {REPORTS_DIR / 'metrics.json'}")


# --------------------------------------------------------------
# FUNCTION: train_vibration
# WHAT IT DOES: Fits anomaly models if the Airbus files are on disk
# WHY WE NEED IT: The engine models must still train on a machine
#                 that has not finished the 1 GB vibration download.
# INPUTS: None
# OUTPUTS: A report dict, or a note that the files were absent
# --------------------------------------------------------------
def train_vibration():
    from rotorcraft_pdm.data.vibration import load_airbus

    try:
        loaded = load_airbus()
    except FileNotFoundError as error:
        print(error)
        return {"status": "skipped", "reason": str(error)}

    baselines = fit_baselines(loaded["train_features"])
    report = {}
    for name, bundle in baselines.items():
        scores = baseline_scores(bundle, loaded["valid_features"])
        # A higher score is more anomalous. The cutoff is the 98th
        # percentile of healthy training scores, same idea as the autoencoder.
        train_scores = baseline_scores(bundle, loaded["train_features"])
        cutoff = float(np.quantile(train_scores, 0.98))
        predicted = (scores >= cutoff).astype(int)
        report[name] = anomaly_metrics(loaded["valid_labels"], scores, predicted)
        report[name]["threshold"] = cutoff
        joblib.dump(bundle, MODEL_DIR / f"{name}.joblib")
        print(name, report[name])

    auto = fit_autoencoder(loaded["train_features"], epochs=25)
    scores, predicted = autoencoder_scores(auto, loaded["valid_features"])
    report["autoencoder"] = anomaly_metrics(loaded["valid_labels"], scores, predicted)
    report["autoencoder"]["threshold"] = auto["threshold"]
    joblib.dump(
        {
            "scaler": auto["scaler"],
            "threshold": auto["threshold"],
            "state": auto["model"].state_dict(),
            "width": loaded["train_features"].shape[1],
        },
        MODEL_DIR / "autoencoder.joblib",
    )
    print("autoencoder", report["autoencoder"])
    return report


if __name__ == "__main__":
    main()
