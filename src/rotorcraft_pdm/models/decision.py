# =============================================================
# WHAT THIS FILE DOES:
# Turns a remaining-life interval into an action a maintainer
# can read: inspect, schedule maintenance, or ground the aircraft.
#
# HOW IT CONNECTS TO OTHER FILES:
# - The quantile models in rul.py produce the interval
# - simulation/fleet.py applies this policy across 12 aircraft
# - the API returns the same action string
#
# THIS IS DECISION SUPPORT, NOT AN AIRWORTHINESS AUTHORITY.
# The cutoffs are written in settings.py so they can be argued with.
# They are not a flight-release rule.
# =============================================================

from rotorcraft_pdm.settings import GROUND_BELOW_CYCLES, SCHEDULE_BELOW_CYCLES


# --------------------------------------------------------------
# FUNCTION: recommend_action
# WHAT IT DOES: Picks an action from the cautious end of the interval
# WHY WE NEED IT: Acting on the point estimate ignores the chance that
#                 the truth is worse than the average guess. The lower
#                 bound is the "what if we are optimistic" check.
# INPUTS:
#   lower_bound: predicted 10th percentile of remaining cycles
#   point: the model's central guess, used only in the explanation
# OUTPUTS: A dictionary with the action and a one-sentence reason
#
# LEARNING NOTE: The alternative is a cost matrix (cost of a wasted
# inspection versus cost of an in-service failure). We still use
# thresholds, but they stand in for that cost choice. The simulation
# is where the cost shows up as failures avoided versus extra pulls.
# --------------------------------------------------------------
def recommend_action(lower_bound, point):
    lower_bound = float(lower_bound)
    point = float(point)
    if lower_bound < GROUND_BELOW_CYCLES:
        action = "ground"
        reason = (
            f"Cautious remaining life is {lower_bound:.0f} cycles, "
            f"under the ground threshold of {GROUND_BELOW_CYCLES}."
        )
    elif lower_bound < SCHEDULE_BELOW_CYCLES:
        action = "schedule_maintenance"
        reason = (
            f"Cautious remaining life is {lower_bound:.0f} cycles. "
            "Plan a maintenance window before the next flying period."
        )
    else:
        action = "inspect"
        reason = (
            f"Cautious remaining life is {lower_bound:.0f} cycles "
            f"(central guess {point:.0f}). Continue, and inspect on the normal cycle."
        )
    return {
        "action": action,
        "lower_bound_cycles": lower_bound,
        "point_cycles": point,
        "reason": reason,
        "disclaimer": "Decision support only. Not an airworthiness release.",
    }
