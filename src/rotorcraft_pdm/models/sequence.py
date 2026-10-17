# =============================================================
# WHAT THIS FILE DOES:
# Trains small sequence models for remaining life: an LSTM and a
# 1D convolutional net. A sequence model sees the last 30 cycles
# as a short movie, not as one flattened row.
#
# HOW IT CONNECTS TO OTHER FILES:
# - features/engine.py sequence_windows() builds the movies
# - scripts/train.py calls fit_sequence_model()
#
# WHY THESE ARE SMALL:
# FD001 is about 20,000 cycles. A huge network would memorize the
# simulator. A small network, trained on CPU, is the honest size.
# =============================================================

import numpy as np

from rotorcraft_pdm.settings import RANDOM_STATE


# --------------------------------------------------------------
# FUNCTION: fit_sequence_model
# WHAT IT DOES: Trains one sequence model and returns it plus predictions
# WHY WE NEED IT: PyTorch training is a loop. Keeping the loop here
#                 means the script does not grow a second copy.
# INPUTS:
#   kind: "lstm" or "cnn"
#   train_x, train_y: windows and targets
#   epochs, batch_size, hidden
# OUTPUTS: A dict with the torch module and the training loss
# --------------------------------------------------------------
def fit_sequence_model(kind, train_x, train_y, epochs=8, batch_size=128, hidden=32):
    import torch
    from torch import nn

    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cpu")
    model = _build(kind, n_features=train_x.shape[-1], hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    features = torch.tensor(train_x, dtype=torch.float32)
    targets = torch.tensor(train_y, dtype=torch.float32).view(-1, 1)
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    dataset = torch.utils.data.TensorDataset(features, targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    model.train()
    last_loss = None
    for _epoch in range(epochs):
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predicted = model(batch_x.to(device))
            loss = loss_fn(predicted, batch_y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        last_loss = float(np.mean(losses))
    model.eval()
    return {"kind": kind, "model": model, "train_mse": last_loss}


# --------------------------------------------------------------
# FUNCTION: predict_sequence
# WHAT IT DOES: Runs a fitted sequence model without tracking gradients
# INPUTS: bundle from fit_sequence_model, windows
# OUTPUTS: A 1-D numpy array of predicted RUL, clipped at zero
# --------------------------------------------------------------
def predict_sequence(bundle, windows):
    import torch

    model = bundle["model"]
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(windows, dtype=torch.float32)
        predicted = model(tensor).cpu().numpy().ravel()
    return np.clip(predicted, 0, None)


# --------------------------------------------------------------
# FUNCTION: _build
# WHAT IT DOES: Constructs the untrained network
# WHY THE CLASSES ARE INSIDE THIS FUNCTION:
# Defining them at import time would force every test file to import
# PyTorch, even tests that only check the NASA score.
# INPUTS: kind, number of sensors, hidden size
# OUTPUTS: A torch.nn.Module
# --------------------------------------------------------------
def _build(kind, n_features, hidden):
    from torch import nn

    class LstmRegressor(nn.Module):
        # The LSTM reads the window from oldest cycle to newest.
        # The last hidden state is the summary we turn into one RUL number.
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, inputs):
            _output, (hidden_state, _cell) = self.lstm(inputs)
            return self.head(hidden_state[-1])

    class CnnRegressor(nn.Module):
        # A 1D convolution looks at a few neighboring cycles at a time,
        # like a mechanic glancing at a short stretch of a trend line.
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, inputs):
            # Conv1d wants (batch, sensors, time). The windows are (batch, time, sensors).
            channels_first = inputs.transpose(1, 2)
            pooled = self.net(channels_first).squeeze(-1)
            return self.head(pooled)

    if kind == "lstm":
        return LstmRegressor()
    if kind == "cnn":
        return CnnRegressor()
    raise ValueError("kind must be 'lstm' or 'cnn'.")
