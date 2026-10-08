from __future__ import annotations

import numpy as np
import pytest

from dcai.eval.bootstrap import difference, grouped_bootstrap


def mean_of(x: np.ndarray):
    return lambda idx: float(x[idx].mean())


def test_point_estimate_is_full_data_statistic() -> None:
    x = np.arange(10.0)
    est = grouped_bootstrap(mean_of(x), np.arange(10), seed=0, n_boot=200)
    assert est.value == 4.5
    assert est.lo < 4.5 < est.hi
    assert est.n_groups == 10 and est.n_valid == 200 and est.valid_fraction == 1.0
    assert str(est).startswith("4.500 [")


def test_deterministic_given_seed() -> None:
    x = np.random.default_rng(0).normal(size=50)
    a = grouped_bootstrap(mean_of(x), np.arange(50), seed=3, n_boot=300)
    b = grouped_bootstrap(mean_of(x), np.arange(50), seed=3, n_boot=300)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_coverage_of_a_known_mean() -> None:
    # Nominal 95% intervals should cover the true mean roughly 95% of the time.
    rng = np.random.default_rng(1)
    hits = 0
    for trial in range(200):
        x = rng.normal(loc=2.0, size=60)
        est = grouped_bootstrap(mean_of(x), np.arange(60), seed=trial, n_boot=400)
        hits += est.lo <= 2.0 <= est.hi
    assert 0.88 <= hits / 200 <= 0.99


def test_grouping_widens_the_interval_for_clustered_data() -> None:
    # 30 patients x 4 near-identical images: effectively 30 observations, not 120.
    rng = np.random.default_rng(2)
    patient_effect = rng.normal(size=30)
    x = np.repeat(patient_effect, 4) + rng.normal(scale=0.05, size=120)
    groups = np.repeat(np.arange(30), 4)
    naive = grouped_bootstrap(mean_of(x), np.arange(120), seed=0, n_boot=1000)
    grouped = grouped_bootstrap(mean_of(x), groups, seed=0, n_boot=1000)
    assert (grouped.hi - grouped.lo) > 1.6 * (naive.hi - naive.lo)  # ~sqrt(4) = 2
    assert grouped.n_groups == 30


def test_whole_groups_are_resampled_together() -> None:
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    seen: list[np.ndarray] = []

    def stat(idx: np.ndarray) -> float:
        seen.append(idx.copy())
        return 0.0

    grouped_bootstrap(stat, groups, seed=0, n_boot=50)
    for idx in seen[1:]:
        counts = {g: int((groups[idx] == g).sum()) for g in "abc"}
        assert counts["a"] % 2 == 0 and counts["b"] % 3 == 0


def test_undefined_resamples_are_dropped_and_counted() -> None:
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    def rate(idx: np.ndarray) -> float:
        return float(y[idx].mean()) if y[idx].any() else float("nan")

    est = grouped_bootstrap(rate, np.arange(8), seed=0, n_boot=500)
    assert 0 < est.n_valid < 500


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"groups": []}, "non-empty"),
        ({"groups": [[1, 2]]}, "non-empty"),
        ({"groups": [1, 1, 1]}, "two groups"),
        ({"groups": [1, 2], "level": 1.0}, "level"),
        ({"groups": [1, 2], "n_boot": 0}, "n_boot"),
    ],
)
def test_input_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        grouped_bootstrap(lambda idx: 0.0, seed=0, **kwargs)


def test_undefined_on_full_data_raises() -> None:
    with pytest.raises(ValueError, match="full data"):
        grouped_bootstrap(lambda idx: float("nan"), [1, 2], seed=0)


def test_undefined_everywhere_raises() -> None:
    calls = iter(range(10**6))
    with pytest.raises(ValueError, match="every bootstrap resample"):
        grouped_bootstrap(
            lambda idx: 1.0 if next(calls) == 0 else float("nan"), [1, 2], seed=0, n_boot=5
        )


def test_paired_difference_cancels_shared_noise() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=40)
    x, y = base + 0.5, base + rng.normal(scale=0.01, size=40)
    groups = np.arange(40)
    a = grouped_bootstrap(mean_of(x), groups, seed=7, n_boot=500)
    b = grouped_bootstrap(mean_of(y), groups, seed=7, n_boot=500)
    d = difference(a, b, paired=True)
    assert d.value == pytest.approx(a.value - b.value)
    assert d.hi - d.lo < 0.1 * (a.hi - a.lo)  # paired: the shared base cancels
    assert d.lo > 0


def test_paired_difference_requires_identical_resamples() -> None:
    x = np.arange(10.0)
    a = grouped_bootstrap(mean_of(x), np.arange(10), seed=1, n_boot=50)
    b = grouped_bootstrap(mean_of(x), np.arange(10), seed=2, n_boot=50)
    with pytest.raises(ValueError, match="identical resamples"):
        difference(a, b, paired=True)


def test_independent_difference_requires_distinct_seeds() -> None:
    x = np.arange(10.0)
    a = grouped_bootstrap(mean_of(x), np.arange(10), seed=1, n_boot=50)
    b = grouped_bootstrap(mean_of(x + 1), np.arange(10), seed=1, n_boot=50)
    with pytest.raises(ValueError, match="different seeds"):
        difference(a, b, paired=False)
    c = grouped_bootstrap(mean_of(x + 1), np.arange(10), seed=2, n_boot=50)
    assert difference(c, a, paired=False).value == pytest.approx(1.0)


def test_difference_requires_matching_n_boot() -> None:
    x = np.arange(10.0)
    a = grouped_bootstrap(mean_of(x), np.arange(10), seed=1, n_boot=50)
    b = grouped_bootstrap(mean_of(x), np.arange(10), seed=2, n_boot=60)
    with pytest.raises(ValueError, match="n_boot"):
        difference(a, b, paired=False)


def test_at_level_reuses_resamples() -> None:
    x = np.random.default_rng(5).normal(size=80)
    est = grouped_bootstrap(mean_of(x), np.arange(80), seed=0, n_boot=1000)
    wider = est.at_level(1 - 0.05 / 3)
    assert wider.value == est.value and wider.level == pytest.approx(1 - 0.05 / 3)
    assert wider.lo < est.lo and wider.hi > est.hi
    assert wider.samples is est.samples
    with pytest.raises(ValueError, match="level"):
        est.at_level(1.0)
