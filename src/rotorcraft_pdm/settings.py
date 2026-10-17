# =============================================================
# WHAT THIS FILE DOES:
# Shared names and constants. One notebook of settings, so the
# trainer, the API, and the dashboard cannot drift apart.
#
# HOW IT CONNECTS TO OTHER FILES:
# - data/cmapss.py uses the column names and the RUL cap
# - models/decision.py uses the action thresholds
# - api/app.py uses the same thresholds when it explains an action
#
# WHY THIS FILE IS NOT NAMED config.py:
# Azure ML's scoring server already has a module named config.
# Importing our own config.py there crashes the container.
# =============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CMAPSS_ZIP_NAME = "CMAPSSData.zip"
CMAPSS_SUBSETS = ("FD001", "FD002", "FD003", "FD004")

# Piecewise-linear RUL cap. Early cycles are treated as "healthy enough"
# that the remaining life is not yet observable. 125 cycles is the value
# used by Saxena and by most public C-MAPSS papers. See DECISIONS.md.
RUL_CAP = 125

# How many past cycles a sequence model sees at once.
SEQUENCE_LENGTH = 30

# Vibration is recorded at 1024 samples per second, one minute per row.
VIBRATION_SAMPLE_RATE_HZ = 1024
VIBRATION_WINDOW_SAMPLES = 1024

RANDOM_STATE = 42

# Decision-support cutoffs, in predicted cycles of remaining life.
# These are teaching thresholds, not airworthiness limits.
GROUND_BELOW_CYCLES = 15
SCHEDULE_BELOW_CYCLES = 40

AZURE_RESOURCE_GROUP = "rg-rotorcraft-pdm"
AZURE_ML_WORKSPACE = "mlw-rotorcraft-pdm"
AZURE_ML_LOCATION = "eastus"
AZURE_STORAGE_ACCOUNT = "strotorpdm001"
EXPERIMENT_NAME = "rotorcraft-pdm"
REGISTERED_RUL_MODEL = "rotorcraft-engine-rul"
REGISTERED_ANOMALY_MODEL = "rotorcraft-vibration-anomaly"

# Sensor meanings from the NASA C-MAPSS documentation. The numbers are
# the column positions after the three operational settings.
SENSOR_NAMES = {
    1: "fan_inlet_temperature",
    2: "lpc_outlet_temperature",
    3: "hpc_outlet_temperature",
    4: "lpt_outlet_temperature",
    5: "fan_inlet_pressure",
    6: "bypass_duct_pressure",
    7: "hpc_outlet_pressure",
    8: "physical_fan_speed",
    9: "physical_core_speed",
    10: "engine_pressure_ratio",
    11: "hpc_outlet_static_pressure",
    12: "fuel_flow_ratio",
    13: "corrected_fan_speed",
    14: "corrected_core_speed",
    15: "bypass_ratio",
    16: "burner_fuel_air_ratio",
    17: "bleed_enthalpy",
    18: "demanded_fan_speed",
    19: "demanded_corrected_fan_speed",
    20: "hpt_coolant_bleed",
    21: "lpt_coolant_bleed",
}
