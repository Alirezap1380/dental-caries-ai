"""Patient-grouped bootstrap confidence intervals.

Rule 5: no metric without a CI, and the resampling unit is the patient.

Images from one patient are correlated (same anatomy, same restorations, same
acquisition session). Resampling images as if they were independent treats a set
of four bitewings as four observations, and the CI comes out too narrow. The unit
of resampling has to be the unit of independence.

Every statistic here is a function of an index array into the evaluation units.
That makes paired comparisons free: two statistics bootstrapped with the same
seed over the same groups see identical resamples, so their per-resample
difference is a paired difference (`difference(a, b, paired=True)`).

Percentile intervals rather than BCa: many statistics here are undefined on
some resamples (kappa with one category, AUC with no positives), which makes
BCa's jackknife acceleration unstable. Undefined resamples are dropped and
counted in `n_valid`, never imputed, so a CI built on a fraction of its
resamples is visible as such.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class Estimate:
    value: float  # point estimate on the full data
    lo: float
    hi: float
    level: float
    n_groups: int
    n_boot: int
    n_valid: int  # resamples on which the statistic was defined
    seed: int
    groups_digest: str
    samples: np.ndarray = field(repr=False, compare=False)

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_boot

    def at_level(self, level: float) -> Estimate:
        """Same resamples, different interval level (e.g. Bonferroni for headline claims)."""
        lo, hi, n_valid = _interval(self.value, self.samples, level)
        return replace(self, lo=lo, hi=hi, level=level, n_valid=n_valid)

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def _digest(groups: np.ndarray) -> str:
    return hashlib.sha1("\x1f".join(map(str, groups)).encode()).hexdigest()


def _interval(value: float, samples: np.ndarray, level: float) -> tuple[float, float, int]:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    valid = samples[np.isfinite(samples)]
    if valid.size == 0:
        raise ValueError("statistic is undefined on every bootstrap resample")
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(valid, [alpha, 1.0 - alpha])
    return float(lo), float(hi), int(valid.size)


def grouped_bootstrap(
    statistic: Statistic,
    groups: Sequence[Hashable] | np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
    level: float = 0.95,
) -> Estimate:
    """Cluster bootstrap: resample groups with replacement, keep all their units.

    `statistic(idx)` receives indices into the units (with repeats) and returns
    a float, or NaN where the statistic is undefined.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("groups must be a non-empty 1-D sequence")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    if n_groups < 2:
        raise ValueError("need at least two groups (patients) to bootstrap")

    value = float(statistic(np.arange(groups.size)))
    if not np.isfinite(value):
        raise ValueError("statistic is undefined on the full data")

    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse))
    members = np.split(order, bounds[:-1])

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, n_groups, size=n_groups)
        samples[b] = statistic(np.concatenate([members[g] for g in picked]))

    lo, hi, n_valid = _interval(value, samples, level)
    return Estimate(
        value=value, lo=lo, hi=hi, level=level, n_groups=n_groups, n_boot=n_boot,
        n_valid=n_valid, seed=seed, groups_digest=_digest(groups), samples=samples,
    )


def difference(a: Estimate, b: Estimate, *, paired: bool) -> Estimate:
    """CI for a - b from two bootstrapped estimates.

    paired=True:  same units, same resamples (e.g. model-vs-reader minus
                  human-vs-reader on the same images). Requires identical seed
                  and groups, which is what makes the resamples line up.
    paired=False: independent samples (e.g. internal minus external test set).
                  Requires *different* seeds: reusing a seed on two datasets with
                  similar group counts correlates their resamples and shrinks
                  the CI for no reason.
    """
    if a.n_boot != b.n_boot or a.level != b.level:
        raise ValueError("estimates must share n_boot and level")
    same_resamples = a.seed == b.seed and a.groups_digest == b.groups_digest
    if paired and not same_resamples:
        raise ValueError("paired difference needs estimates from identical resamples")
    if not paired and a.seed == b.seed:
        raise ValueError("independent difference needs estimates bootstrapped with different seeds")

    samples = a.samples - b.samples
    value = a.value - b.value
    lo, hi, n_valid = _interval(value, samples, a.level)
    return Estimate(
        value=value, lo=lo, hi=hi, level=a.level, n_groups=min(a.n_groups, b.n_groups),
        n_boot=a.n_boot, n_valid=n_valid, seed=a.seed,
        groups_digest=a.groups_digest if paired else f"{a.groups_digest}-{b.groups_digest}",
        samples=samples,
    )
