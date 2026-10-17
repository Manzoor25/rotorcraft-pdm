# =============================================================
# WHAT THIS FILE DOES:
# Trains vibration anomaly detectors. Isolation forest and
# one-class SVM learn "what normal looks like" and flag the rest.
# The autoencoder learns to rebuild a normal feature vector.
# A vector it cannot rebuild is treated as abnormal.
#
# HOW IT CONNECTS TO OTHER FILES:
# - features/vibration.py produces the vectors
# - data/vibration.py loads the Airbus files
# - the API calls score_autoencoder() on a new trace
# =============================================================

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from rotorcraft_pdm.settings import RANDOM_STATE


# --------------------------------------------------------------
# FUNCTION: fit_baselines
# WHAT IT DOES: Fits the two classical one-class models on healthy rows
# WHY WE NEED IT: If a neural net cannot beat these, it is not the story
# INPUTS: train_features (healthy only), small
# OUTPUTS: A dict of name -> fitted bundle
# --------------------------------------------------------------
def fit_baselines(train_features, small=False):
    scaler = StandardScaler().fit(train_features)
    scaled = scaler.transform(train_features)
    forest = IsolationForest(
        n_estimators=20 if small else 200,
        contamination=0.02,
        random_state=RANDOM_STATE,
    )
    forest.fit(scaled)
    # One-class SVM is slow on thousands of rows. A fixed sample is enough
    # for a baseline, and the sample size is written into the bundle.
    sample_size = min(len(scaled), 80 if small else 1500)
    rng = np.random.default_rng(RANDOM_STATE)
    chosen = rng.choice(len(scaled), size=sample_size, replace=False)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(scaled[chosen])
    return {
        "isolation_forest": {"model": forest, "scaler": scaler},
        "one_class_svm": {"model": svm, "scaler": scaler, "sample_size": sample_size},
    }


# --------------------------------------------------------------
# FUNCTION: baseline_scores
# WHAT IT DOES: Higher means more anomalous
# WHY WE NEED IT: Isolation forest's decision_function is higher for
#                 normal points. We flip the sign so every detector
#                 in this project uses the same direction.
# INPUTS: bundle, features
# OUTPUTS: A 1-D score array
# --------------------------------------------------------------
def baseline_scores(bundle, features):
    scaled = bundle["scaler"].transform(features)
    decision = bundle["model"].decision_function(scaled)
    return -np.asarray(decision, dtype=float)


# --------------------------------------------------------------
# FUNCTION: fit_autoencoder
# WHAT IT DOES: Trains a small network to rebuild healthy feature vectors
# WHY WE NEED IT: Reconstruction error is a second opinion that does
#                 not assume the anomalies form a neat blob.
# INPUTS: train_features, epochs, hidden
# OUTPUTS: A bundle with the network, the scaler, and the threshold
# --------------------------------------------------------------
def fit_autoencoder(train_features, epochs=20, hidden=16):
    import torch
    from torch import nn

    torch.manual_seed(RANDOM_STATE)
    scaler = StandardScaler().fit(train_features)
    scaled = scaler.transform(train_features).astype(np.float32)
    width = scaled.shape[1]
    model = nn.Sequential(
        nn.Linear(width, hidden),
        nn.ReLU(),
        nn.Linear(hidden, max(4, hidden // 2)),
        nn.ReLU(),
        nn.Linear(max(4, hidden // 2), hidden),
        nn.ReLU(),
        nn.Linear(hidden, width),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tensor = torch.tensor(scaled)
    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        rebuilt = model(tensor)
        loss = loss_fn(rebuilt, tensor)
        loss.backward()
        optimizer.step()
    errors = reconstruction_error(model, tensor)
    # The threshold is the 98th percentile of healthy error.
    # About 2% of healthy flights will be flagged. That is a choice,
    # not a regulation. It is stored with the model so serving uses
    # the same cutoff the evaluation used.
    threshold = float(np.quantile(errors, 0.98))
    model.eval()
    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "train_mse": float(loss.detach()),
    }


# --------------------------------------------------------------
# FUNCTION: reconstruction_error
# WHAT IT DOES: Mean squared error between a vector and its rebuild
# INPUTS: model, tensor
# OUTPUTS: A 1-D numpy array
# --------------------------------------------------------------
def reconstruction_error(model, tensor):
    import torch

    model.eval()
    with torch.no_grad():
        rebuilt = model(tensor)
        error = torch.mean((rebuilt - tensor) ** 2, dim=1)
    return error.cpu().numpy()


# --------------------------------------------------------------
# FUNCTION: autoencoder_scores
# WHAT IT DOES: Scores new rows. Higher means harder to rebuild.
# INPUTS: bundle, features
# OUTPUTS: scores, and a yes/no prediction using the stored threshold
# --------------------------------------------------------------
def autoencoder_scores(bundle, features):
    import torch

    scaled = bundle["scaler"].transform(features).astype(np.float32)
    scores = reconstruction_error(bundle["model"], torch.tensor(scaled))
    predicted = (scores >= bundle["threshold"]).astype(int)
    return scores, predicted
