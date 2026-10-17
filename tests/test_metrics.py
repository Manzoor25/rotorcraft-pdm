# =============================================================
# WHAT THIS FILE DOES:
# Locks the NASA score asymmetry. A late error must cost more
# than an early error of the same size. If someone "simplifies"
# the formula, this test fails.
# =============================================================

from rotorcraft_pdm.evaluation.metrics import interval_coverage, nasa_score, rul_metrics


def test_late_prediction_is_penalized_more_than_early():
    early = nasa_score([100], [80])
    late = nasa_score([100], [120])
    assert late > early


def test_perfect_prediction_scores_zero():
    assert nasa_score([40, 10], [40, 10]) == 0


def test_rul_metrics_include_the_nasa_score():
    scores = rul_metrics([10, 20], [10, 20])
    assert scores["rmse"] == 0
    assert scores["nasa_score"] == 0


def test_promotion_requires_a_strictly_better_nasa_score():
    from rotorcraft_pdm.evaluation.metrics import should_promote

    assert should_promote(10, 12) is True
    assert should_promote(12, 10) is False
    assert should_promote(10, 10) is False


def test_interval_coverage_counts_hits():
    coverage = interval_coverage([5, 50], [0, 0], [10, 10])
    assert coverage == 0.5
