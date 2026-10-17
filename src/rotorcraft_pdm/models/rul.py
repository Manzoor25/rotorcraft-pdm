# =============================================================
# WHAT THIS FILE DOES:
# Trains the classical remaining-life models: a straight line,
# a random forest, LightGBM, and XGBoost. LightGBM is also fit
# twice more, at the 10th and 90th percentiles, so we get an
# interval instead of only a point.
#
# HOW IT CONNECTS TO OTHER FILES:
# - features/engine.py builds the table
# - evaluation/metrics.py scores the last-cycle predictions
# - models/decision.py reads the lower end of the interval
#
# WHY FOUR MODELS:
# The line is the baseline a skeptic can explain. The trees can
# bend. If a tree does not beat the line by a meaningful amount,
# the fancy model is not earning its keep.
# =============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from rotorcraft_pdm.settings import RANDOM_STATE


# --------------------------------------------------------------
# FUNCTION: build_point_models
# WHAT IT DOES: Creates the unfitted point-estimate models
# INPUTS: small (True uses tiny forests so tests finish quickly)
# OUTPUTS: A dictionary of name -> model
# --------------------------------------------------------------
def build_point_models(small=False):
    if small:
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=10, max_depth=3, random_state=RANDOM_STATE
            ),
            "lightgbm": _lightgbm(n_estimators=10, num_leaves=7),
            "xgboost": _xgboost(n_estimators=10, max_depth=2),
        }
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "lightgbm": _lightgbm(n_estimators=300, num_leaves=31),
        "xgboost": _xgboost(n_estimators=300, max_depth=6),
    }


# --------------------------------------------------------------
# FUNCTION: _lightgbm
# WHAT IT DOES: Builds one LightGBM regressor with quiet logs
# WHY WE NEED IT: LightGBM prints a progress bar unless we ask it not to
# INPUTS: tree count, leaf count, optional quantile alpha
# OUTPUTS: An unfitted model
# --------------------------------------------------------------
def _lightgbm(n_estimators, num_leaves, alpha=None):
    from lightgbm import LGBMRegressor

    extra = {}
    if alpha is not None:
        extra = {"objective": "quantile", "alpha": alpha}
    return LGBMRegressor(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbose=-1,
        **extra,
    )


# --------------------------------------------------------------
# FUNCTION: _xgboost
# WHAT IT DOES: Builds one XGBoost regressor
# INPUTS: tree count, depth
# OUTPUTS: An unfitted model
# --------------------------------------------------------------
def _xgboost(n_estimators, max_depth):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.9,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


# --------------------------------------------------------------
# FUNCTION: fit_quantile_models
# WHAT IT DOES: Fits the low and high LightGBM quantile models
# WHY WE NEED IT: A point estimate hides how unsure the model is.
#                 The 10th percentile is the cautious remaining life.
#                 The maintenance policy uses that cautious number.
# INPUTS: features, targets, small
# OUTPUTS: A dict with keys "low" and "high"
# --------------------------------------------------------------
def fit_quantile_models(features, targets, small=False):
    trees = 10 if small else 300
    leaves = 7 if small else 31
    low = _lightgbm(trees, leaves, alpha=0.1)
    high = _lightgbm(trees, leaves, alpha=0.9)
    low.fit(features, targets)
    high.fit(features, targets)
    return {"low": low, "high": high}
