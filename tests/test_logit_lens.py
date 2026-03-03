import numpy as np
import pytest
from src.metrics import (
    compute_divergence_onset,
    categorize_divergence_pattern,
    paired_permutation_test,
    aggregate_trajectories,
)


class TestDivergenceOnset:
    def test_clear_divergence(self):
        probs_en = np.array([0.0, 0.1, 0.3, 0.5, 0.8])
        probs_tr = np.array([0.0, 0.1, 0.1, 0.1, 0.1])
        onset = compute_divergence_onset(probs_en, probs_tr, threshold=0.1)
        assert onset == 2  # layer 2 is where diff > 0.1

    def test_no_divergence(self):
        probs_en = np.array([0.1, 0.2, 0.3])
        probs_tr = np.array([0.1, 0.2, 0.3])
        onset = compute_divergence_onset(probs_en, probs_tr, threshold=0.1)
        assert onset == -1


class TestCategorizeDivergence:
    def test_no_divergence(self):
        probs = np.linspace(0, 1, 33)
        assert categorize_divergence_pattern(probs, probs) == "no_divergence"

    def test_mid_divergence(self):
        probs_en = np.zeros(33)
        probs_tr = np.zeros(33)
        probs_en[15] = 0.8  # big difference in mid layers
        assert categorize_divergence_pattern(probs_en, probs_tr) == "mid_divergence"


class TestPermutationTest:
    def test_significant_difference(self):
        a = np.array([10, 11, 12, 13, 14])
        b = np.array([1, 2, 3, 4, 5])
        p = paired_permutation_test(a, b)
        assert p < 0.05

    def test_no_difference(self):
        a = np.array([1, 2, 3, 4, 5])
        p = paired_permutation_test(a, a)
        assert p > 0.5


class TestAggregateTrajectories:
    def test_shape(self):
        all_probs = [np.random.rand(33) for _ in range(10)]
        mean, ci_low, ci_high = aggregate_trajectories(all_probs)
        assert mean.shape == (33,)
        assert ci_low.shape == (33,)
        assert ci_high.shape == (33,)

    def test_ci_bounds(self):
        all_probs = [np.random.rand(33) for _ in range(10)]
        mean, ci_low, ci_high = aggregate_trajectories(all_probs)
        assert np.all(ci_low <= mean)
        assert np.all(ci_high >= mean)
