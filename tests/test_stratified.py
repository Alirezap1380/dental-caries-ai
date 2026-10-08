from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dcai.eval.scales import DENTEX_DEPTH, ICCMS_STAGE
from dcai.eval.stratified import auc, stratified_report

N_BOOT = 300


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_sklearn_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.random(100) < 0.3
    s = np.round(rng.random(100) + 0.3 * y, 1)  # rounding forces ties
    assert auc(s, y) == pytest.approx(roc_auc_score(y, s))


def test_auc_single_class_is_nan() -> None:
    assert np.isnan(auc(np.array([0.1, 0.2]), np.array([True, True])))
    assert np.isnan(auc(np.array([0.1, 0.2]), np.array([False, False])))


def staged_units(
    detect_prob: dict[int, float], n_per_level: dict[int, int], seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Units whose chance of scoring above 0.5 is set per level; 0 = sound."""
    rng = np.random.default_rng(seed)
    levels = np.concatenate([np.full(n, lv) for lv, n in n_per_level.items()])
    p = np.array([detect_prob[lv] for lv in levels])
    hit = rng.random(levels.size) < p
    scores = np.where(hit, rng.uniform(0.5, 1.0, levels.size), rng.uniform(0.0, 0.5, levels.size))
    groups = np.array([f"p{i // 2}" for i in range(levels.size)])
    return levels, scores, groups


def test_report_structure_and_order() -> None:
    levels, scores, groups = staged_units({0: 0.1, 1: 0.4, 2: 0.9}, {0: 300, 1: 150, 2: 150})
    r = stratified_report(levels, scores, groups, scale=DENTEX_DEPTH, threshold=0.5,
                          seed=0, n_boot=N_BOOT)
    assert [s.name for s in r.strata] == ["caries", "deep_caries"]
    assert r.n_sound == 300 and r.scale == "dentex_depth" and r.threshold == 0.5
    assert r.stratum("caries").n_units == 150
    assert r.stratum("caries").sensitivity.value == pytest.approx(
        (scores[levels == 1] >= 0.5).mean()
    )
    assert r.specificity.value == pytest.approx((scores[levels == 0] < 0.5).mean())
    assert all(s.auc_vs_sound is not None for s in r.strata)


def test_pooling_hides_the_early_lesion_failure() -> None:
    # The project thesis in miniature: deep lesions dominate the pooled figure,
    # which looks fine while the shallow stratum is failing.
    levels, scores, groups = staged_units({0: 0.1, 1: 0.25, 2: 0.95}, {0: 400, 1: 60, 2: 340})
    r = stratified_report(levels, scores, groups, scale=DENTEX_DEPTH, threshold=0.5,
                          seed=0, n_boot=N_BOOT)
    shallow = r.stratum("caries").sensitivity
    assert r.pooled_sensitivity.lo > shallow.hi
    assert r.depth_gap.lo > 0.5
    assert r.depth_gap.value == pytest.approx(
        r.stratum("deep_caries").sensitivity.value - shallow.value
    )


def test_empty_stratum_is_reported_not_dropped() -> None:
    levels, scores, groups = staged_units({0: 0.1, 3: 0.5, 5: 0.9}, {0: 100, 3: 50, 5: 50})
    r = stratified_report(levels, scores, groups, scale=ICCMS_STAGE, threshold=0.5,
                          seed=0, n_boot=N_BOOT)
    assert [s.name for s in r.strata] == ["E1", "E2", "D1", "D2", "D3"]
    e1 = r.stratum("E1")
    assert e1.n_units == 0 and e1.sensitivity is None and e1.auc_vs_sound is None
    assert r.depth_gap is not None  # D3 minus D1, the shallowest present


def test_lesion_units_have_no_specificity() -> None:
    levels, scores, groups = staged_units({1: 0.4, 2: 0.9}, {1: 50, 2: 50})
    r = stratified_report(levels, scores, groups, scale=DENTEX_DEPTH, threshold=0.5,
                          seed=0, n_boot=N_BOOT)
    assert r.specificity is None and r.n_sound == 0
    assert all(s.auc_vs_sound is None for s in r.strata)


def test_single_stratum_has_no_gap() -> None:
    levels, scores, groups = staged_units({0: 0.1, 2: 0.9}, {0: 50, 2: 50})
    r = stratified_report(levels, scores, groups, scale=DENTEX_DEPTH, threshold=0.5,
                          seed=0, n_boot=N_BOOT)
    assert r.depth_gap is None


@pytest.mark.parametrize(
    "levels, scores, groups, match",
    [
        ([1, 2], [0.5], ["a", "b"], "same length"),
        ([1, 3], [0.5, 0.5], ["a", "b"], "out of range"),
        ([1, -1], [0.5, 0.5], ["a", "b"], "out of range"),
        ([1, 2], [0.5, np.nan], ["a", "b"], "NaN"),
        ([0, 0], [0.5, 0.5], ["a", "b"], "no lesions"),
    ],
)
def test_validation(levels: list, scores: list, groups: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        stratified_report(levels, scores, groups, scale=DENTEX_DEPTH, threshold=0.5, seed=0)
