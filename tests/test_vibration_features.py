# =============================================================
# WHAT THIS FILE DOES:
# Checks that a vibration window becomes a fixed-length vector,
# and that a spike raises the crest factor.
# =============================================================

import numpy as np

from rotorcraft_pdm.features.vibration import crest_factor, sequence_feature_vector


def test_spike_raises_crest_factor():
    calm = np.ones(256)
    spiked = calm.copy()
    spiked[10] = 50
    assert crest_factor(spiked) > crest_factor(calm)


def test_sequence_vector_has_a_stable_length():
    trace = np.sin(np.linspace(0, 40, 4096))
    vector = sequence_feature_vector(trace, window=1024)
    assert vector.shape == (9,)
    assert np.isfinite(vector).all()
