# =============================================================
# WHAT THIS FILE DOES:
# Loads the Airbus helicopter accelerometer files and turns each
# one-minute trace into the feature vector in features/vibration.py.
#
# HOW IT CONNECTS TO OTHER FILES:
# - The HDF5 files are downloaded by scripts/download_data.py
# - models/anomaly.py trains on the matrices this file returns
#
# LICENSE:
# The Airbus files are CC BY-NC-SA 4.0. This loader does not copy
# them into Git. See DATA_SOURCES.md.
# =============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from rotorcraft_pdm.features.vibration import sequence_feature_vector
from rotorcraft_pdm.settings import RAW_DIR


# --------------------------------------------------------------
# FUNCTION: _read_hdf_frame
# WHAT IT DOES: Opens one Airbus HDF5 table
# WHY WE NEED IT: The file is a pandas table stored as HDF5, not a
#                 plain CSV. pandas.read_hdf is the intended reader.
#                 h5py is the fallback if the pytables extra is missing.
# INPUTS: path, key ("dftrain" or "dfvalid")
# OUTPUTS: A DataFrame, one row per one-minute sequence
# --------------------------------------------------------------
def _read_hdf_frame(path, key):
    path = Path(path)
    try:
        return pd.read_hdf(path, key=key)
    except (ImportError, ValueError, OSError):
        import h5py

        with h5py.File(path, "r") as handle:
            node = handle[key]
            if hasattr(node, "keys"):
                # pandas HDF5 stores the values under a block inside the key.
                values = node["block0_values"][:]
            else:
                values = node[:]
        return pd.DataFrame(values)


# --------------------------------------------------------------
# FUNCTION: features_from_frame
# WHAT IT DOES: Summarizes every row of a vibration table
# WHY WE NEED IT: We do not want to keep 61,440 samples per flight
#                 minute in memory after the summary exists.
# INPUTS: frame, one sequence per row
# OUTPUTS: A 2-D numpy array, one feature vector per row
# --------------------------------------------------------------
def features_from_frame(frame):
    rows = []
    values = frame.to_numpy(dtype=np.float32)
    for row in values:
        rows.append(sequence_feature_vector(row))
    return np.vstack(rows)


# --------------------------------------------------------------
# FUNCTION: load_airbus
# WHAT IT DOES: Loads healthy training traces and labeled validation traces
# WHY WE NEED IT: The anomaly models need "what normal looks like" and
#                 a labeled exam set. Those are two different files.
# INPUTS: raw_dir
# OUTPUTS: A dict with train_features, valid_features, and valid_labels
# --------------------------------------------------------------
def load_airbus(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "airbus_train.h5"
    valid_path = raw_dir / "airbus_validation.h5"
    label_path = raw_dir / "airbus_ground_truth.csv"
    missing = [path.name for path in (train_path, valid_path, label_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Airbus files missing: " f"{missing}. Run python scripts/download_data.py"
        )
    train = features_from_frame(_read_hdf_frame(train_path, "dftrain"))
    valid = features_from_frame(_read_hdf_frame(valid_path, "dfvalid"))
    labels = pd.read_csv(label_path)
    if "anomaly" not in labels.columns:
        raise ValueError("Ground truth must have an anomaly column.")
    return {
        "train_features": train,
        "valid_features": valid,
        "valid_labels": labels["anomaly"].to_numpy(dtype=int),
    }


if __name__ == "__main__":
    loaded = load_airbus()
    print("train", loaded["train_features"].shape)
    print("valid", loaded["valid_features"].shape)
    print("anomalies", int(loaded["valid_labels"].sum()))
