# =============================================================
# WHAT THIS FILE DOES:
# Loads the NASA C-MAPSS turbofan files and turns them into a
# table with a remaining-useful-life label.
#
# HOW IT CONNECTS TO OTHER FILES:
# - settings.py names the subsets and the RUL cap
# - features/engine.py builds the columns the classical models see
# - scripts/train.py calls load_subset()
#
# WHY A TURBOFAN FILE IS IN A HELICOPTER PROJECT:
# A public turboshaft run-to-failure file does not exist. C-MAPSS
# is the standard public proxy. The differences are written in
# DATA_SOURCES.md. This loader does not pretend the engines are T700s.
# =============================================================

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from rotorcraft_pdm.settings import CMAPSS_ZIP_NAME, RAW_DIR, RUL_CAP

SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
COLUMN_NAMES = ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]


# --------------------------------------------------------------
# FUNCTION: ensure_extracted
# WHAT IT DOES: Unzips CMAPSSData.zip once, next to the zip
# WHY WE NEED IT: The official download is one zip, not loose text files
# INPUTS: raw_dir, the folder that holds the zip
# OUTPUTS: The folder that now contains train_FD001.txt and the rest
# --------------------------------------------------------------
def ensure_extracted(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    marker = raw_dir / "train_FD001.txt"
    if marker.exists():
        return raw_dir
    zip_path = raw_dir / CMAPSS_ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}. Run: python scripts/download_data.py")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)
    return raw_dir


# --------------------------------------------------------------
# FUNCTION: read_trajectory_file
# WHAT IT DOES: Reads one space-separated C-MAPSS text file
# WHY WE NEED IT: The files have no header. pandas would otherwise
#                 invent names, and every later file would disagree.
# INPUTS: path to train_FD00x.txt or test_FD00x.txt
# OUTPUTS: A DataFrame with unit, cycle, 3 settings, 21 sensors
# --------------------------------------------------------------
def read_trajectory_file(path):
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    frame["unit"] = frame["unit"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    return frame


# --------------------------------------------------------------
# FUNCTION: add_train_rul
# WHAT IT DOES: Labels each training row with cycles until failure
# WHY WE NEED IT: The training file ends at failure, so the last
#                 cycle of each engine is RUL 0. Earlier cycles count up.
# INPUTS: frame from a train_*.txt file, cap (the piecewise ceiling)
# OUTPUTS: The same rows, plus a rul column
#
# LEARNING NOTE: Without the cap, the model is asked to predict 300
# cycles of life from a healthy engine that looks identical to one
# with 200 cycles left. The cap says "until degradation is visible,
# call the life 'at least this many'."
# --------------------------------------------------------------
def add_train_rul(frame, cap=RUL_CAP):
    labeled = frame.copy()
    last_cycle = labeled.groupby("unit")["cycle"].transform("max")
    raw_rul = last_cycle - labeled["cycle"]
    labeled["rul"] = raw_rul.clip(upper=cap).astype(float)
    return labeled


# --------------------------------------------------------------
# FUNCTION: add_test_rul
# WHAT IT DOES: Attaches the official remaining-life vector to a test file
# WHY WE NEED IT: Test engines are stopped BEFORE failure. The true
#                 remaining life is in RUL_FD00x.txt, one number per engine.
# INPUTS: test frame, rul_values (one number per unit, in unit order)
# OUTPUTS: The test frame plus a rul column, capped the same way as training
# --------------------------------------------------------------
def add_test_rul(frame, rul_values, cap=RUL_CAP):
    labeled = frame.copy()
    last_cycle = labeled.groupby("unit")["cycle"].transform("max")
    # rul_values[0] belongs to unit 1. Units in the file start at 1.
    rul_at_end = labeled["unit"].map(
        {unit: float(rul_values[unit - 1]) for unit in range(1, len(rul_values) + 1)}
    )
    cycles_before_end = last_cycle - labeled["cycle"]
    labeled["rul"] = (rul_at_end + cycles_before_end).clip(upper=cap)
    return labeled


# --------------------------------------------------------------
# FUNCTION: read_rul_vector
# WHAT IT DOES: Reads RUL_FD00x.txt, one remaining-life number per engine
# INPUTS: path
# OUTPUTS: A 1-D numpy array
# --------------------------------------------------------------
def read_rul_vector(path):
    values = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)
    return values


# --------------------------------------------------------------
# FUNCTION: constant_sensors
# WHAT IT DOES: Finds sensors that never change in the training file
# WHY WE NEED IT: A constant column cannot predict anything, and it
#                 breaks scaling (standard deviation is zero).
# INPUTS: training frame
# OUTPUTS: A list of sensor column names
# --------------------------------------------------------------
def constant_sensors(frame, tolerance=1e-8):
    stuck = []
    for column in SENSOR_COLUMNS:
        if float(frame[column].std(ddof=0)) <= tolerance:
            stuck.append(column)
    return stuck


# --------------------------------------------------------------
# FUNCTION: regime_id
# WHAT IT DOES: Turns the three throttle settings into a regime number
# WHY WE NEED IT: FD002 and FD004 mix six flight conditions. A sensor
#                 value that is "high" in one condition is normal in another.
#                 We scale each condition separately.
# INPUTS: frame with setting_1, setting_2, setting_3
# OUTPUTS: A Series of integer regime ids
#
# LEARNING NOTE: Rounding is enough because the six conditions are
# distinct, not a smooth cloud. A clustering model is the alternative.
# Rounding does not need to be fit, so train and serve cannot disagree.
# --------------------------------------------------------------
def regime_id(frame):
    rounded = np.column_stack(
        [
            np.round(frame["setting_1"].to_numpy(dtype=float), 3),
            np.round(frame["setting_2"].to_numpy(dtype=float), 5),
            np.round(frame["setting_3"].to_numpy(dtype=float), 3),
        ]
    )
    codes, _uniques = pd.factorize(pd.Series(map(tuple, rounded)), sort=True)
    return pd.Series(codes, index=frame.index, name="regime")


# --------------------------------------------------------------
# FUNCTION: fit_scaler
# WHAT IT DOES: Records mean and standard deviation per regime and sensor
# WHY WE NEED IT: Neural nets and linear models need numbers on a
#                 similar scale. The scaler must be fit on training rows
#                 only, then reused on test rows.
# INPUTS: training frame, sensor columns to keep
# OUTPUTS: A dict that scale_frame() understands
# --------------------------------------------------------------
def fit_scaler(frame, sensor_columns):
    work = frame.copy()
    work["regime"] = regime_id(work)
    stats = {}
    for regime, group in work.groupby("regime"):
        means = group[sensor_columns].mean()
        stds = group[sensor_columns].std(ddof=0).replace(0, 1.0)
        stats[int(regime)] = {
            "mean": means.to_dict(),
            "std": stds.to_dict(),
        }
    return {"sensors": list(sensor_columns), "regimes": stats}


# --------------------------------------------------------------
# FUNCTION: scale_frame
# WHAT IT DOES: Applies a fitted scaler. Unseen regimes use the
#               global training mean so a new condition does not crash serving.
# INPUTS: frame, scaler dict from fit_scaler
# OUTPUTS: A copy with sensors scaled and a regime column added
# --------------------------------------------------------------
def scale_frame(frame, scaler):
    scaled = frame.copy()
    scaled["regime"] = regime_id(scaled)
    sensors = scaler["sensors"]
    global_mean = {}
    global_std = {}
    for column in sensors:
        means = [item["mean"][column] for item in scaler["regimes"].values()]
        stds = [item["std"][column] for item in scaler["regimes"].values()]
        global_mean[column] = float(np.mean(means))
        global_std[column] = float(np.mean(stds)) or 1.0

    for column in sensors:
        means = scaled["regime"].map(
            lambda regime, column=column: scaler["regimes"]
            .get(int(regime), {})
            .get("mean", {})
            .get(column, global_mean[column])
        )
        stds = scaled["regime"].map(
            lambda regime, column=column: scaler["regimes"]
            .get(int(regime), {})
            .get("std", {})
            .get(column, global_std[column])
        )
        scaled[column] = (scaled[column] - means) / stds.replace(0, 1.0)
    return scaled


# --------------------------------------------------------------
# FUNCTION: load_subset
# WHAT IT DOES: Loads one FD00x subset, labels RUL, drops dead sensors
# INPUTS: subset name, such as "FD001"
# OUTPUTS: A dict with train, test, scaler, and the dropped sensor names
# --------------------------------------------------------------
def load_subset(subset, raw_dir=RAW_DIR, cap=RUL_CAP):
    folder = ensure_extracted(raw_dir)
    train = add_train_rul(read_trajectory_file(folder / f"train_{subset}.txt"), cap=cap)
    rul_values = read_rul_vector(folder / f"RUL_{subset}.txt")
    test = add_test_rul(
        read_trajectory_file(folder / f"test_{subset}.txt"),
        rul_values,
        cap=cap,
    )
    dropped = constant_sensors(train)
    kept = [column for column in SENSOR_COLUMNS if column not in dropped]
    scaler = fit_scaler(train, kept)
    return {
        "subset": subset,
        "train": train,
        "test": test,
        "dropped_sensors": dropped,
        "kept_sensors": kept,
        "scaler": scaler,
        "rul_cap": cap,
    }


if __name__ == "__main__":
    loaded = load_subset("FD001")
    print(f"FD001 train rows: {len(loaded['train'])}")
    print(f"Dropped constant sensors: {loaded['dropped_sensors']}")
