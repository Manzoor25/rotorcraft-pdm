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

- Nothing further can be registered until the blocker below is cleared.
- After that, deploy one `Standard_F2s_v2` endpoint, invoke it once, and delete it the same day.

## Blocker

Azure ML model registration is blocked on this subscription. I tried three times.

1. A laptop upload to `strotorpdm001` is rejected because public network access is forced off.
2. A compute job inside the managed virtual network, using the cluster identity, still failed. The model registry service lists blobs with an account key. The storage account returns `KeyBasedAuthenticationNotPermitted`.
3. Switching the workspace to identity datastore auth did not change that registry call. A retry after the switch failed the same way. Job name: `placid_hamster_m8vywglps0`.

What I need from you: a storage account that allows shared-key access for the Azure ML registry, or a policy exception on `strotorpdm001` that permits key authentication for the workspace service. I cannot grant that from this login. The trained files are in the GitHub repo under `models/`, so registration can be retried without retraining once storage auth allows it.

The cluster `cpu-rotor` has a minimum of 0 nodes, so it should not keep billing after the failed jobs. No online endpoint was created for this project.
