# Costs

Prices are East US Linux pay-as-you-go list prices unless noted. They are not a quote for this subscription.

## Created

| Resource | Name | Why it exists | Approximate cost |
| --- | --- | --- | --- |
| Resource group | `rg-rotorcraft-pdm` | Isolates this proof of concept | $0 |
| Storage account | `strotorpdm001` | Workspace data, public access disabled by policy | A few dollars a month if nearly empty |
| Key vault | `kvrotoroub6ur66blnza` | Required by the Azure ML workspace | Cents, plus any secret operations |
| Application Insights | `appi-rotor-pdm` | Required by the workspace | Cents unless it ingests a lot of telemetry |
| Azure ML workspace | `mlw-rotorcraft-pdm` | Experiment tracking and the model registry | No charge for the workspace itself |
| Training | This laptop | All model fitting | $0 of Azure compute |
| CPU cluster | `cpu-rotor` (`Standard_DS2_v2`, minimum 0 nodes) | Attempted model registration from inside the private network | About $0.14 per hour only while a node is up. It scales to zero. |

The previous project `bank-td-mm25` online endpoint was deleted at the start of this work. It had been billing about $0.085 an hour and did not scale to zero. Its model registry was not deleted.

## What was not left running

No GPU. No managed online endpoint was created for this project. The cluster `cpu-rotor` was used for three failed registration attempts and is set to scale to zero. If an online endpoint is created later, use one `Standard_F2s_v2` at about **$0.085 per hour**. It does not scale to zero. Delete it the same day. See [RUNBOOK.md](RUNBOOK.md).

## How to redeploy

1. `az deployment group create --resource-group rg-rotorcraft-pdm --template-file infra/main.bicep`
2. `python scripts/train.py`
3. Register the bundle in `models/rul_bundle.joblib` from a machine that can reach the private storage account.
4. Deploy one small CPU instance, invoke it once, then delete the endpoint.

Nothing in this file is permission to leave a managed online endpoint running overnight.
