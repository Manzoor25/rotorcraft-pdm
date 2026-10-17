# =============================================================
# WHAT THIS FILE DOES:
# Writes reports/data_quality.md from the C-MAPSS files.
# A quality report is a list of "is this file what we think it is?"
#
# HOW IT CONNECTS TO OTHER FILES:
# - data/cmapss.py loads each subset
# - The notebook repeats the same questions with charts
# =============================================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotorcraft_pdm.data.cmapss import load_subset
from rotorcraft_pdm.settings import CMAPSS_SUBSETS, REPORTS_DIR


def main():
    lines = ["# Data quality", ""]
    for subset in CMAPSS_SUBSETS:
        loaded = load_subset(subset)
        train = loaded["train"]
        test = loaded["test"]
        lines.append(f"## {subset}")
        lines.append("")
        lines.append(f"- Training rows: {len(train):,}")
        lines.append(f"- Training engines: {train['unit'].nunique()}")
        lines.append(f"- Test rows: {len(test):,}")
        lines.append(f"- Test engines: {test['unit'].nunique()}")
        lines.append(
            f"- Constant sensors dropped: {', '.join(loaded['dropped_sensors']) or 'none'}"
        )
        lines.append(f"- RUL cap: {loaded['rul_cap']}")
        lines.append(f"- Training RUL min/max: {train['rul'].min():.0f} / {train['rul'].max():.0f}")
        lines.append("")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "data_quality.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
