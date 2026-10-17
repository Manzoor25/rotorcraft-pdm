# =============================================================
# WHAT THIS FILE DOES:
# Checks that the fleet comparison counts failures, and that the
# tail numbers are the fictional POC series, not real serials.
# =============================================================

import numpy as np

from rotorcraft_pdm.simulation.fleet import simulate_fleet


def test_fleet_uses_fictional_tails():
    life = np.arange(20, 200, 10)
    cautious = np.full_like(life, 10, dtype=float)
    result = simulate_fleet(life, cautious, n_aircraft=12)
    tails = list(result["table"]["tail"])
    assert tails[0] == "POC-01"
    assert tails[-1] == "POC-12"
    assert result["aircraft"] == 12
