# Runbook

This is the operator note for the proof of concept. It is not a flight-line procedure.

## What "healthy" looks like

- `GET /health` returns `status: ok`.
- `reports/metrics.json` exists and names an FD001 winner.
- The Azure resource group `rg-rotorcraft-pdm` exists.
- No managed online endpoint is left running after a demo. Check with:

```powershell
az ml online-endpoint list --resource-group rg-rotorcraft-pdm --workspace-name mlw-rotorcraft-pdm -o table
```

## Retrain

From the project folder, with the virtual environment active:

```powershell
python scripts/download_data.py
python scripts/train.py
pytest
```

The winner is the FD001 model with the lowest NASA score. A candidate replaces a saved production score only if `python scripts/promote.py reports/metrics.json PRODUCTION_SCORE` exits 0.

## Drift

`models/drift_baseline.json` stores the training mean and spread of the sensors that were kept. `POST /drift` with a batch of new rows returns `drift_flag`. If the flag is true:

1. Do not treat the remaining-life number as current.
2. Re-run `python scripts/train.py` on a refreshed extract.
3. Promote only if the NASA score improved.

This check is a sensor-shift alarm. It is not a certified monitor.

## If Azure upload fails

This subscription creates storage accounts with public network access disabled, and it ignores attempts to turn that access on. A laptop then gets `AuthorizationFailure` when it tries to upload a model.

What you need if you want the laptop path:

- Permission to allow public network access on `strotorpdm001`, or a private endpoint from this machine into that account.

What works without that permission:

- Run registration from a compute cluster in the workspace managed virtual network, using the cluster's managed identity, as the bank-marketing project did. That path costs a small CPU cluster while the job runs. Do not leave an online endpoint up afterward.

## Stop the bill

```powershell
az ml online-endpoint delete --name rotor-pdm-mm25 --resource-group rg-rotorcraft-pdm --workspace-name mlw-rotorcraft-pdm --yes
az ml compute delete --name cpu-rotor --resource-group rg-rotorcraft-pdm --workspace-name mlw-rotorcraft-pdm --yes
```

To remove the workspace, storage, key vault, and Application Insights:

```powershell
az group delete --name rg-rotorcraft-pdm --yes
```

## Redeploy the infrastructure

```powershell
az group create --name rg-rotorcraft-pdm --location eastus
az deployment group create --resource-group rg-rotorcraft-pdm --template-file infra/main.bicep
```
