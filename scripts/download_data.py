# =============================================================
# WHAT THIS FILE DOES:
# Downloads the public datasets into data/raw. It does not commit
# them. Git ignores that folder.
#
# HOW IT CONNECTS TO OTHER FILES:
# - data/cmapss.py reads the zip this script saves
# - data/vibration.py reads the Airbus HDF5 files
# =============================================================

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
AIRBUS = "https://www.research-collection.ethz.ch/server/api/core/bitstreams"
AIRBUS_FILES = {
    "a701c2e4-a3ca-41e4-99b9-2fa657fd72bf": "airbus_train.h5",
    "34ecff00-0cf8-4891-ad5f-b347396ac14b": "airbus_validation.h5",
    "2c21d0db-da85-4e05-be87-4ad44a3647b6": "airbus_ground_truth.csv",
    "c592926a-afa5-4105-857c-9d95cf94ed0a": "airbus_description.txt",
}


def _download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1000:
        print(f"already present: {destination.name}")
        return
    print(f"downloading {destination.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "rotorcraft-pdm/0.1"})
    with urllib.request.urlopen(request, timeout=600) as response:
        destination.write_bytes(response.read())
    print(f"saved {destination} ({destination.stat().st_size} bytes)")


def main():
    _download(CMAPSS_URL, RAW / "CMAPSSData.zip")
    for bitstream, name in AIRBUS_FILES.items():
        _download(f"{AIRBUS}/{bitstream}/content", RAW / name)


if __name__ == "__main__":
    main()
