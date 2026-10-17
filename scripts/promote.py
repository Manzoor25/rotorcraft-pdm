# =============================================================
# WHAT THIS FILE DOES:
# Decides whether a candidate model may replace the current one.
# The only vote is the NASA score. Lower wins. A tie does not promote.
#
# HOW IT CONNECTS TO OTHER FILES:
# - .github/workflows/promote.yml calls this before any Azure update
# - reports/metrics.json is the candidate
# - a saved production score is the incumbent
# =============================================================

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotorcraft_pdm.evaluation.metrics import should_promote


def main():
    if len(sys.argv) != 3:
        print("usage: python scripts/promote.py CANDIDATE.json PRODUCTION_SCORE")
        return 2
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = [row for row in report["classical"] if row["subset"] == "FD001"]
    winner = min(rows, key=lambda row: row["nasa_score"])
    candidate = winner["nasa_score"]
    production = float(sys.argv[2])
    if should_promote(candidate, production):
        print(f"promote: candidate NASA {candidate:.1f} beats production {production:.1f}")
        return 0
    print(f"hold: candidate NASA {candidate:.1f} does not beat production {production:.1f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
