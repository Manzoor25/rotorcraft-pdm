# =============================================================
# WHAT THIS FILE DOES:
# Scores remaining-life predictions and anomaly detectors.
#
# HOW IT CONNECTS TO OTHER FILES:
# - scripts/train.py calls these after each model
# - EVALUATION.md is written from the numbers these functions return
#
# THE NASA SCORE, IN PLAIN ENGLISH:
# Predicting that an engine will fail sooner than it really will
# (an early pull) wastes a maintenance slot. Predicting that it
# will last longer than it really will (a late call) can leave a
# failing engine in service. The NASA score charges more for the
# late call. A plain average hides that difference.
# =============================================================

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------
# FUNCTION: nasa_score
# WHAT IT DOES: The official asymmetric penalty for RUL error
# WHY WE NEED IT: Being late by 20 cycles is more dangerous than
#                 being early by 20 cycles. RMSE treats them as equal.
# INPUTS: y_true, y_pred, remaining cycles
# OUTPUTS: A single number. Lower is better. Zero is a perfect call.
#
# LEARNING NOTE: d = predicted - true.
#   d < 0  early  -> exp(-d/13) - 1
#   d >= 0 late   -> exp(d/10) - 1
# The late branch uses 10, not 13, so the same-sized mistake grows faster.
# --------------------------------------------------------------
def nasa_score(y_true, y_pred):
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = np.exp(-error / 13.0) - 1.0
    late = np.exp(error / 10.0) - 1.0
    penalty = np.where(error < 0, early, late)
    return float(np.sum(penalty))


# --------------------------------------------------------------
# FUNCTION: rul_metrics
# WHAT IT DOES: RMSE, MAE, and the NASA score, together
# INPUTS: y_true, y_pred
# OUTPUTS: A dictionary of floats
# --------------------------------------------------------------
def rul_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
    }


# --------------------------------------------------------------
# FUNCTION: interval_coverage
# WHAT IT DOES: Fraction of true values that land inside the interval
# WHY WE NEED IT: A wide interval can look humble and still be useless.
#                 Coverage tells you how often the interval was honest.
# INPUTS: y_true, lower, upper
# OUTPUTS: A float between 0 and 1
# --------------------------------------------------------------
def interval_coverage(y_true, lower, upper):
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.mean(inside))


# --------------------------------------------------------------
# FUNCTION: anomaly_metrics
# WHAT IT DOES: Scores a yes/no anomaly detector, and counts misses
# WHY WE NEED IT: A missed anomaly is the dangerous case in aviation
# INPUTS: y_true (1 = anomaly), y_score (higher = more anomalous), y_pred
# OUTPUTS: A dictionary
# --------------------------------------------------------------
# --------------------------------------------------------------
# FUNCTION: should_promote
# WHAT IT DOES: Says yes only when the candidate NASA score is lower
# WHY WE NEED IT: A tie must not replace a model that is already in service
# INPUTS: candidate_score, production_score
# OUTPUTS: True or False
# --------------------------------------------------------------
def should_promote(candidate_score, production_score):
    return float(candidate_score) < float(production_score)


def anomaly_metrics(y_true, y_score, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    missed = int(np.sum((y_true == 1) & (y_pred == 0)))
    more_than_one_class = len(np.unique(y_true)) > 1
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if more_than_one_class else float("nan"),
        "pr_auc": (
            float(average_precision_score(y_true, y_score)) if more_than_one_class else float("nan")
        ),
        "false_negatives": missed,
        "support_positive": int(np.sum(y_true == 1)),
    }
