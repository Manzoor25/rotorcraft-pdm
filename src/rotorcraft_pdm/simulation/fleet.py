# =============================================================
# WHAT THIS FILE DOES:
# Walks 12 fictional aircraft through time and compares two
# maintenance policies:
#   1. Fixed interval: pull the engine every N cycles, no model.
#   2. Cautious RUL: pull when the lower confidence bound is short.
#
# HOW IT CONNECTS TO OTHER FILES:
# - models/decision.py names the actions
# - scripts/train.py writes the fleet table the dashboard reads
#
# THE AIRCRAFT ARE NOT REAL.
# Tail numbers are POC-01 through POC-12. They are not Army serials.
# =============================================================

import numpy as np
import pandas as pd

from rotorcraft_pdm.settings import GROUND_BELOW_CYCLES, RANDOM_STATE


# --------------------------------------------------------------
# FUNCTION: simulate_fleet
# WHAT IT DOES: Runs both policies on the same hidden failure times
# WHY WE NEED IT: A lower RMSE does not, by itself, mean fewer
#                 unplanned failures. This function counts those.
# INPUTS:
#   true_rul_at_start: remaining cycles at the moment we start watching
#   lower_bounds: the model's cautious remaining life at that same moment
#   fixed_interval: cycles between scheduled pulls if we ignore the model
# OUTPUTS: A dictionary of counts plus a per-aircraft table
#
# LEARNING NOTE: We start watching at a random point in each engine's
# remaining life, not at cycle 0. A policy that only looks good at
# cycle 0 is not a fleet policy.
# --------------------------------------------------------------
def simulate_fleet(true_rul_at_start, lower_bounds, fixed_interval=80, n_aircraft=12):
    rng = np.random.default_rng(RANDOM_STATE)
    true_rul_at_start = np.asarray(true_rul_at_start, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    chosen = rng.choice(len(true_rul_at_start), size=n_aircraft, replace=False)
    rows = []
    for index, engine_index in enumerate(chosen, start=1):
        hidden_life = float(true_rul_at_start[engine_index])
        cautious = float(lower_bounds[engine_index])
        # Fixed policy: the next planned pull is `fixed_interval` cycles away,
        # no matter what the sensors say.
        fixed_failure = hidden_life < fixed_interval
        if cautious < GROUND_BELOW_CYCLES:
            # Ground now. That is a planned action, not an in-service failure.
            model_failure = False
            action = "ground"
            model_early = True
        else:
            planned_at = min(max(cautious, 0.0), float(fixed_interval))
            model_failure = hidden_life < planned_at
            action = "schedule_maintenance" if planned_at < fixed_interval else "inspect"
            model_early = action == "schedule_maintenance" and not model_failure
        rows.append(
            {
                "tail": f"POC-{index:02d}",
                "source_engine": int(engine_index + 1),
                "hidden_rul": hidden_life,
                "cautious_rul": cautious,
                "fixed_interval_failure": bool(fixed_failure),
                "model_unplanned_failure": bool(model_failure),
                "model_pulls_early": bool(model_early),
                "action": action,
            }
        )
    table = pd.DataFrame(rows)
    return {
        "aircraft": n_aircraft,
        "fixed_interval": fixed_interval,
        "fixed_unplanned_failures": int(table["fixed_interval_failure"].sum()),
        "model_unplanned_failures": int(table["model_unplanned_failure"].sum()),
        "model_early_pulls": int(table["model_pulls_early"].sum()),
        "availability_fixed": float(1.0 - table["fixed_interval_failure"].mean()),
        "availability_model": float(1.0 - table["model_unplanned_failure"].mean()),
        "table": table,
    }
