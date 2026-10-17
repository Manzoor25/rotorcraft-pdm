# =============================================================
# WHAT THIS FILE DOES:
# Registers the already-trained bundles in Azure Machine Learning.
# It is meant to run on workspace compute, where the managed
# identity can reach private storage. A laptop often cannot.
#
# HOW IT CONNECTS TO OTHER FILES:
# - models/rul_bundle.joblib is the forest that won the NASA score
# - models/one_class_svm.joblib is the vibration detector
# =============================================================

import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import ManagedIdentityCredential

ROOT = Path(__file__).resolve().parents[1]


def client():
    return MLClient(
        credential=ManagedIdentityCredential(),
        subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
        resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP", "rg-rotorcraft-pdm"),
        workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME", "mlw-rotorcraft-pdm"),
    )


def register(ml_client, name, path, description):
    model = Model(
        name=name,
        path=str(path),
        type=AssetTypes.CUSTOM_MODEL,
        description=description,
    )
    registered = ml_client.models.create_or_update(model)
    print(f"registered {registered.name} version {registered.version}")


def main():
    ml_client = client()
    register(
        ml_client,
        "rotorcraft-engine-rul",
        ROOT / "models" / "rul_bundle.joblib",
        "FD001 winner by NASA score. Random forest plus 10/90 quantile bounds. Not a turboshaft certification model.",
    )
    register(
        ml_client,
        "rotorcraft-vibration-anomaly",
        ROOT / "models" / "one_class_svm.joblib",
        "One-class SVM on Airbus accelerometer summaries. Missed 49 of 297 validation anomalies.",
    )


if __name__ == "__main__":
    main()
