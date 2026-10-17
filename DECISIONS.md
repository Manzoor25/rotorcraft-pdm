# Decisions

When two reasonable choices existed, this file records the one that was made and the one that was rejected.

## 1. Public proxies, not a real Apache tape

**Chose:** NASA C-MAPSS for engine remaining life, and the Airbus flight-test accelerometer set for vibration.
**Rejected:** Asking for operational HUMS data, or inventing a dataset and calling it Apache data.

A portfolio that needs a clearance to run is not a portfolio. The proxy argument and its limits are in [DATA_SOURCES.md](DATA_SOURCES.md).

## 2. Skip N-CMAPSS

**Chose:** The classic four C-MAPSS text files.
**Rejected:** Blocking the project on a NASA request for N-CMAPSS.

N-CMAPSS is the closer modern simulator, and it is not a stable anonymous download. The classic files are the ones every public paper can be compared against.

## 3. RUL cap of 125 cycles

**Chose:** Piecewise-linear cap at 125.
**Rejected:** Predicting the raw remaining life from the first healthy cycle, and rejected a cap of 130 only because 125 is the value the Saxena line of papers made comparable.

Early in life the sensors have not moved. Asking the model for an exact large number teaches it to memorize engine identity. The cap says "at least 125, until the degradation is visible."

## 4. NASA score picks the winner

**Chose:** Lowest NASA score on the last cycle of each FD001 test engine.
**Rejected:** Lowest RMSE, and rejected picking the winner on all four subsets mixed together.

FD001 is the single-condition exam. Mixing it with FD004's six conditions would let a model win by being lucky on an easier subset. RMSE treats a late call and an early call as the same mistake. In aviation they are not. The score is explained in [EVALUATION.md](EVALUATION.md).

## 5. Uncertainty from quantile regression

**Chose:** LightGBM at the 10th and 90th percentiles.
**Rejected:** MC dropout as the primary interval. Dropout intervals on a small net are noisy and slow to calibrate. Quantile regression is a direct model of the bound the maintenance policy needs.

Coverage of those intervals is reported. If coverage is far from 80 percent, the interval is not honest, and EVALUATION.md says so.

## 6. Actions use the lower bound

**Chose:** Ground below 15 cautious cycles, schedule below 40, otherwise inspect.
**Rejected:** Acting on the point estimate.

The point estimate is the average guess. The lower bound is the "what if we are optimistic" number. These cutoffs are teaching choices, not a flight-release rule. The fleet simulation is where they earn or fail their keep.

## 7. Sequence models on FD001 only

**Chose:** LSTM and 1D CNN on FD001. Classical models on all four subsets.
**Rejected:** A transformer, and rejected training the LSTM on FD004.

FD004 is several times larger and has six regimes. A transformer would add parameters without a new question. The CNN already looks at neighboring cycles. CPU time was spent on the exam that has a published protocol.

## 8. Vibration models see summaries, not 61,440 raw samples

**Chose:** One-second windows, then crest factor, kurtosis, and broad FFT bands.
**Rejected:** A raw-sample autoencoder on the full minute, and rejected a narrow "1x rotor" bin.

We do not know the rotor speed of each Airbus flight. A narrow bin would be a guess. Broad bands are something a HUMS screen already shows. The autoencoder rebuilds the summary vector. Reconstruction error is the anomaly score.

## 9. Azure storage stays private

**Chose:** Bicep creates the storage account with public access disabled, matching the subscription policy that ignores attempts to turn it on.
**Rejected:** Fighting the policy, and rejected reusing the bank-marketing workspace.

A laptop may not be able to upload blobs. The runbook says so. Models are still trained locally and registered when the identity path works. The previous bank endpoint was deleted before this project created new compute, because it billed about $0.085 an hour and did not scale to zero. That model remains in its own registry and can be redeployed from that repository.

## 10. The settings module is not named config.py

**Chose:** `settings.py`.
**Rejected:** `config.py`.

Azure ML's scoring container already imports a module named config. The collision crashed a previous endpoint. This project does not repeat that name.
