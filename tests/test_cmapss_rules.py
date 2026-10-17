# =============================================================
# WHAT THIS FILE DOES:
# Checks the RUL cap and the constant-sensor drop without the
# full NASA download. CI can run this on a clean computer.
# =============================================================

import pandas as pd

from rotorcraft_pdm.data.cmapss import add_train_rul, constant_sensors
from rotorcraft_pdm.data.schemas import validate_cmapss
from rotorcraft_pdm.models.decision import recommend_action
from rotorcraft_pdm.settings import GROUND_BELOW_CYCLES, RUL_CAP


def _tiny_engine():
    rows = []
    for cycle in range(1, 6):
        row = {"unit": 1, "cycle": cycle, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
        for sensor in range(1, 22):
            row[f"sensor_{sensor}"] = 1.0 if sensor == 1 else float(cycle)
        rows.append(row)
    return pd.DataFrame(rows)


def test_rul_is_capped():
    frame = _tiny_engine()
    labeled = add_train_rul(frame, cap=2)
    assert labeled["rul"].max() == 2
    assert RUL_CAP == 125


def test_constant_sensor_is_named():
    stuck = constant_sensors(_tiny_engine())
    assert "sensor_1" in stuck
    assert "sensor_2" not in stuck


def test_schema_accepts_a_complete_row():
    frame = _tiny_engine()
    checked = validate_cmapss(frame)
    assert len(checked) == 5


def test_ground_action_uses_the_lower_bound():
    action = recommend_action(lower_bound=GROUND_BELOW_CYCLES - 1, point=80)
    assert action["action"] == "ground"
    assert "airworthiness" in action["disclaimer"]
