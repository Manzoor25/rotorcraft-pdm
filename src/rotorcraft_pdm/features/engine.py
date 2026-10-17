# =============================================================
# WHAT THIS FILE DOES:
# Turns a raw engine table into the columns a classical model
# can learn from: the latest sensor values, plus short rolling
# summaries. A rolling summary is "what did this sensor do over
# the last few cycles," not just the latest snapshot.
#
# HOW IT CONNECTS TO OTHER FILES:
# - data/cmapss.py supplies the table
# - models/rul.py trains on the table this file returns
# =============================================================

import numpy as np
import pandas as pd

from rotorcraft_pdm.data.cmapss import SETTING_COLUMNS


# --------------------------------------------------------------
# FUNCTION: rolling_feature_frame
# WHAT IT DOES: Adds rolling mean and rolling standard deviation
# WHY WE NEED IT: Degradation is a trend. One snapshot can look
#                 normal. Five snapshots drifting upward do not.
# INPUTS:
#   frame: one subset, with a unit column so we never roll across engines
#   sensor_columns: sensors that are not constant
#   window: how many past cycles to summarize
# OUTPUTS: A DataFrame. The first (window - 1) cycles of each engine
#          are dropped, because the rolling window is not full yet.
#
# LEARNING NOTE: groupby(unit) is required. Without it, engine 2's
# first cycle would include engine 1's last cycles. That is leakage
# across engines, and it inflates the score.
# --------------------------------------------------------------
def rolling_feature_frame(frame, sensor_columns, window=5):
    pieces = []
    feature_columns = [*SETTING_COLUMNS, *sensor_columns]
    for _unit, group in frame.groupby("unit", sort=False):
        ordered = group.sort_values("cycle").copy()
        rolled = ordered[sensor_columns].rolling(window=window, min_periods=window)
        ordered[[f"{column}_mean_{window}" for column in sensor_columns]] = rolled.mean().to_numpy()
        ordered[[f"{column}_std_{window}" for column in sensor_columns]] = rolled.std().to_numpy()
        pieces.append(ordered.dropna(subset=[f"{sensor_columns[0]}_mean_{window}"]))
    built = pd.concat(pieces, ignore_index=True)
    keep = ["unit", "cycle", "rul", *feature_columns]
    keep += [f"{column}_mean_{window}" for column in sensor_columns]
    keep += [f"{column}_std_{window}" for column in sensor_columns]
    return built[keep]


# --------------------------------------------------------------
# FUNCTION: last_cycle_rows
# WHAT IT DOES: Keeps only the final cycle of each engine
# WHY WE NEED IT: The official C-MAPSS score is on that last cycle,
#                 not on every cycle in the middle of the test.
# INPUTS: frame with unit and cycle
# OUTPUTS: One row per engine
# --------------------------------------------------------------
def last_cycle_rows(frame):
    last = frame.groupby("unit")["cycle"].transform("max")
    return frame.loc[frame["cycle"] == last].copy()


# --------------------------------------------------------------
# FUNCTION: feature_matrix
# WHAT IT DOES: Splits a feature table into X and y
# INPUTS: frame from rolling_feature_frame
# OUTPUTS: (X dataframe, y array)
# --------------------------------------------------------------
def feature_matrix(frame):
    drop = ["unit", "rul"]
    features = frame.drop(columns=[column for column in drop if column in frame.columns])
    target = frame["rul"].to_numpy(dtype=float)
    return features, target


# --------------------------------------------------------------
# FUNCTION: sequence_windows
# WHAT IT DOES: Cuts each engine into overlapping windows for LSTM/CNN
# WHY WE NEED IT: A sequence model wants a short movie, not one photo
# INPUTS: scaled frame, sensor columns, window length
# OUTPUTS: windows (n, length, sensors), targets (n,), units (n,)
# --------------------------------------------------------------
def sequence_windows(frame, sensor_columns, length):
    windows = []
    targets = []
    units = []
    cycles = []
    for unit, group in frame.groupby("unit", sort=False):
        ordered = group.sort_values("cycle")
        values = ordered[sensor_columns].to_numpy(dtype=np.float32)
        rul = ordered["rul"].to_numpy(dtype=np.float32)
        cycle = ordered["cycle"].to_numpy(dtype=np.int32)
        if len(ordered) < length:
            continue
        for end in range(length - 1, len(ordered)):
            start = end - length + 1
            windows.append(values[start : end + 1])
            targets.append(rul[end])
            units.append(int(unit))
            cycles.append(int(cycle[end]))
    return (
        np.stack(windows),
        np.asarray(targets, dtype=np.float32),
        np.asarray(units),
        np.asarray(cycles),
    )
