from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from dcai.eval.calibration import (
    brier_score,
    calibration_report,
    ece_noise_floor,
    expected_calibration_error,
    recalibration,
    reliability_curve,
)

N_BOOT = 200


def calibrated(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 4, size=n)  # mostly low, like caries on a mostly-sound set
    return p, (rng.random(n) < p).astype(int)


def overconfident(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    p, y = calibrated(n, seed)
    return expit(3 * logit(np.clip(p, 1e-6, 1 - 1e-6))), y  # true slope 1/3


def test_brier_known_values() -> None:
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0
    assert brier_score([1.0, 0.0], [0, 1]) == 1.0
    assert brier_score([0.5, 0.5], [0, 1]) == 0.25


def test_ece_hand_computed() -> None:
    # Two uniform bins: [0.1, 0.1] observed 0.5 -> 0.4 * 2/4; [0.9, 0.9] observed 1 -> 0.1 * 2/4
    p, y = [0.1, 0.1, 0.9, 0.9], [0, 1, 1, 1]
    assert expected_calibration_error(p, y, n_bins=2, strategy="uniform") == pytest.approx(0.25)


def test_ece_near_zero_for_large_calibrated_sample() -> None:
    p, y = calibrated(50_000)
    assert expected_calibration_error(p, y) < 0.01


def test_ece_large_for_overconfident_model() -> None:
    p, y = overconfident(5000)
    assert expected_calibration_error(p, y) > 0.05


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_reliability_curve_accounts_for_every_prediction(strategy: str) -> None:
    p, y = calibrated(1000)
    curve = reliability_curve(p, y, n_bins=10, strategy=strategy)
    assert sum(b.count for b in curve.bins) == 1000
    assert all(b.lo <= b.mean_predicted <= b.hi for b in curve.bins)
    if strategy == "quantile":
        assert {b.count for b in curve.bins} <= {100}


def test_quantile_bins_collapse_on_ties() -> None:
    curve = reliability_curve([0.3] * 10, [0, 1] * 5)
    assert len(curve.bins) == 1 and curve.bins[0].observed == 0.5
    assert expected_calibration_error([0.3] * 10, [0, 1] * 5) == pytest.approx(0.2)


def test_uniform_bins_include_probability_one() -> None:
    curve = reliability_curve([0.0, 1.0], [0, 1], n_bins=4, strategy="uniform")
    assert [b.count for b in curve.bins] == [1, 1]


def test_recalibration_recovers_known_miscalibration() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 50_000)
    y = (rng.random(p.size) < expit(0.5 + 0.5 * logit(p))).astype(int)
    a, b = recalibration(p, y)
    assert a == pytest.approx(0.5, abs=0.05)
    assert b == pytest.approx(0.5, abs=0.05)


def test_recalibration_of_calibrated_model() -> None:
    a, b = recalibration(*calibrated(50_000))
    assert a == pytest.approx(0.0, abs=0.1)
    assert b == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize(
    "p, y",
    [
        ([0.2, 0.4, 0.6], [1, 1, 1]),  # one class
        ([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]),  # perfect separation
        ([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]),  # constant input: singular
    ],
)
def test_recalibration_degenerate_is_nan(p: list, y: list) -> None:
    assert np.isnan(recalibration(p, y)).all()


def test_noise_floor_is_positive_and_shrinks_with_n() -> None:
    small = ece_noise_floor(calibrated(150)[0], seed=0, n_sim=300)
    large = ece_noise_floor(calibrated(5000)[0], seed=0, n_sim=300)
    assert small > large > 0
    assert small > 0.03  # at test-set sizes like ours the floor is not negligible


def test_report_calibrated_model_within_noise_floor() -> None:
    p, y = calibrated(600, seed=3)
    groups = [f"p{i // 3}" for i in range(600)]
    r = calibration_report(p, y, groups, seed=0, n_boot=N_BOOT, n_sim=300)
    assert not r.ece_exceeds_noise_floor
    assert r.slope.lo < 1.0 < r.slope.hi
    assert r.n == 600 and r.prevalence == pytest.approx(y.mean())
    assert r.brier.value == pytest.approx(brier_score(p, y))


def test_report_flags_overconfident_model() -> None:
    p, y = overconfident(600, seed=3)
    groups = [f"p{i // 3}" for i in range(600)]
    r = calibration_report(p, y, groups, seed=0, n_boot=N_BOOT, n_sim=300)
    assert r.ece_exceeds_noise_floor
    assert r.slope.hi < 1.0


def test_report_without_estimable_recalibration() -> None:
    r = calibration_report([0.5] * 8, [0, 1] * 4, [f"p{i}" for i in range(8)],
                           seed=0, n_boot=N_BOOT, n_sim=50)
    assert r.slope is None and r.intercept is None


@pytest.mark.parametrize(
    "p, y, match",
    [
        ([], [], "non-empty"),
        ([0.1, 0.2], [0], "same length"),
        ([1.2], [1], r"\[0, 1\]"),
        ([np.nan], [1], r"\[0, 1\]"),
        ([0.5], [2], "binary"),
    ],
)
def test_validation(p: list, y: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        brier_score(p, y)


def test_bin_validation() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([0.5], [1], n_bins=0)
    with pytest.raises(ValueError, match="strategy"):
        expected_calibration_error([0.5], [1], strategy="kmeans")  # type: ignore[arg-type]


def test_report_groups_length() -> None:
    with pytest.raises(ValueError, match="groups"):
        calibration_report([0.5, 0.5], [0, 1], ["a"], seed=0)
