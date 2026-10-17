# =============================================================
# WHAT THIS FILE DOES:
# A small drift check. Drift means "today's sensor values do not
# look like the values the model was trained on."
#
# HOW IT CONNECTS TO OTHER FILES:
# - scripts/train.py saves the training mean and spread
# - the API can call population_stability() on a new batch
#
# WHY THIS IS BASIC:
# A full drift system pages a human and retrains. This function
# only measures the shift and returns a flag. RUNBOOK.md says
# what a person should do when the flag is true.
# =============================================================

import numpy as np


# --------------------------------------------------------------
# FUNCTION: fit_drift_baseline
# WHAT IT DOES: Stores the training mean and standard deviation
# INPUTS: a 2-D array, one row per cycle, one column per sensor
# OUTPUTS: A dict that score_drift() can reuse
# --------------------------------------------------------------
def fit_drift_baseline(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.clip(np.std(values, axis=0), 1e-6, None).tolist(),
    }


# --------------------------------------------------------------
# FUNCTION: score_drift
# WHAT IT DOES: Average absolute z-score of the new batch versus training
# WHY WE NEED IT: A z-score of 3 means the new average is three training
#                 spreads away from the old average. That is a shift.
# INPUTS: baseline from fit_drift_baseline, new values
# OUTPUTS: A dict with the score and a boolean flag
# --------------------------------------------------------------
def score_drift(baseline, values, warn_at=1.5):
    values = np.asarray(values, dtype=float)
    mean = np.asarray(baseline["mean"], dtype=float)
    std = np.asarray(baseline["std"], dtype=float)
    shift = np.abs(np.mean(values, axis=0) - mean) / std
    score = float(np.mean(shift))
    return {"mean_abs_z": score, "drift_flag": bool(score >= warn_at)}
