# =============================================================
# WHAT THIS FILE DOES:
# Turns reports/metrics.json into the markdown table in
# reports/results.md. The numbers are not typed by hand.
# =============================================================

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
lines = [
    "# Results",
    "",
    "Written by scripts/write_results.py from reports/metrics.json.",
    "The FD001 winner is the lowest NASA score. Lower is better.",
    "",
    "## FD001 classical models",
    "",
    "| Model | RMSE | MAE | NASA score |",
    "| --- | ---: | ---: | ---: |",
]
for row in report["classical"]:
    if row["subset"] != "FD001":
        continue
    mark = " (selected)" if row["model"] == report["fd001_winner"] else ""
    lines.append(
        f"| {row['model']}{mark} | {row['rmse']:.2f} | {row['mae']:.2f} | {row['nasa_score']:.1f} |"
    )
lines += [
    "",
    "## Sequence models on FD001",
    "",
    "| Model | RMSE | MAE | NASA score |",
    "| --- | ---: | ---: | ---: |",
]
for row in report["sequence"]:
    lines.append(f"| {row['model']} | {row['rmse']:.2f} | {row['mae']:.2f} | {row['nasa_score']:.1f} |")
lines += [
    "",
    f"10-90 interval coverage on the FD001 exam: {report['interval_coverage_10_90']:.3f}",
    "",
    "## Other subsets",
    "",
    "| Subset | Model | RMSE | NASA score |",
    "| --- | --- | ---: | ---: |",
]
for row in report["classical"]:
    if row["subset"] == "FD001":
        continue
    lines.append(f"| {row['subset']} | {row['model']} | {row['rmse']:.2f} | {row['nasa_score']:.1f} |")
path = ROOT / "reports" / "results.md"
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(path)
