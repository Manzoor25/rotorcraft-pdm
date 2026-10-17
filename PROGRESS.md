# Progress

## Done

- Public repository: https://github.com/Manzoor25/rotorcraft-pdm
- NASA C-MAPSS and the Airbus accelerometer files downloaded. Licenses are in [DATA_SOURCES.md](DATA_SOURCES.md). Raw files are gitignored.
- Azure resource group `rg-rotorcraft-pdm`, storage `strotorpdm001`, and workspace `mlw-rotorcraft-pdm` created from [infra/main.bicep](infra/main.bicep).
- Classical models, a small LSTM, a small 1D CNN, and three vibration detectors trained. Random forest won FD001 on NASA score (986.4). One-class SVM missed the fewest vibration anomalies (49 of 297).
- FastAPI, Streamlit, Dockerfiles, pytest (14 passing), and the lint workflow are in the repo.
- Evaluation, decisions, runbook, costs, data-quality report, and the GitHub Pages post are written.
- The previous bank-marketing online endpoint was deleted so it would not keep billing. Its model registry was left in place.

## Next

- Register the trained bundles in Azure ML from a compute identity, because a laptop upload is blocked.
- If registration succeeds, deploy one small endpoint, invoke it once, and delete it the same day.

## Blocker

Laptop upload to `strotorpdm001` fails because the subscription forces public network access off. The error is that the request is blocked by the storage account network rules.

To clear a laptop upload, I would need permission to enable public network access on `strotorpdm001`, or a private path from this machine. The next attempt does not need that permission: a short job on a workspace compute cluster, using that cluster's identity, can register the model that is already in the GitHub repo. That job spends a few cents of CPU. It is not done yet.
