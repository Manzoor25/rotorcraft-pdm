# =============================================================
# WHAT THIS FILE DOES:
# A fleet screen. Twelve fictional aircraft, each green, amber,
# or red, with a drill-down for remaining life and the recommended
# action.
#
# HOW IT CONNECTS TO OTHER FILES:
# - scripts/train.py writes reports/fleet.csv and reports/metrics.json
# - models/decision.py is the same action logic the API uses
#
# RUN IT:
#   streamlit run src/rotorcraft_pdm/dashboard/app.py
#
# This is a picture of a simulation. It is not a live Army fleet.
# =============================================================

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"


def _load_fleet():
    path = REPORTS / "fleet.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _color(action):
    if action == "ground":
        return "red"
    if action == "schedule_maintenance":
        return "orange"
    return "green"


def main():
    st.set_page_config(page_title="Rotorcraft PDM proof of concept", layout="wide")
    st.title("Rotorcraft predictive maintenance")
    st.caption(
        "Public-data proof of concept. Not affiliated with the U.S. Army or Boeing. "
        "Not an airworthiness release. Tail numbers POC-01 to POC-12 are fictional."
    )
    fleet = _load_fleet()
    if fleet is None:
        st.warning("No fleet file yet. Run: python scripts/train.py")
        return

    counts = fleet["action"].value_counts().to_dict()
    left, middle, right = st.columns(3)
    left.metric("Ground", counts.get("ground", 0))
    middle.metric("Schedule", counts.get("schedule_maintenance", 0))
    right.metric("Inspect", counts.get("inspect", 0))

    st.subheader("Fleet")
    show = fleet.copy()
    show["status"] = show["action"].map(_color)
    st.dataframe(show, use_container_width=True)

    st.subheader("One aircraft")
    tail = st.selectbox("Tail", list(fleet["tail"]))
    row = fleet.loc[fleet["tail"] == tail].iloc[0]
    st.write(
        f"Hidden remaining life in the simulation: {row['hidden_rul']:.0f} cycles. "
        f"Cautious model bound: {row['cautious_rul']:.0f} cycles. "
        f"Recommended action: {row['action']}."
    )
    chart = pd.DataFrame(
        {
            "estimate": ["hidden life", "cautious bound"],
            "cycles": [row["hidden_rul"], row["cautious_rul"]],
        }
    )
    st.bar_chart(chart.set_index("estimate"))

    metrics_path = REPORTS / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.subheader("Model exam, FD001")
        st.json(
            {
                "winner": metrics.get("fd001_winner"),
                "interval_coverage": metrics.get("interval_coverage_10_90"),
                "fleet": metrics.get("fleet"),
            }
        )


if __name__ == "__main__":
    main()
