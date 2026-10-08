"""Calibration: when the model says 0.8, is it right 80% of the time?

This matters for E3 because abstention and referral act on the probability
itself, not only on its ranking. A model with good AUC and bad calibration puts
its referral threshold in the wrong place.

Feed this module per-tooth probabilities (`units.ToothTable.p_lesion`), never
detector box scores. A detector chooses how many boxes it emits, so a box-score
bin has no fixed denominator and its ECE depends on NMS settings. See `units.py`.

Methodological choices:

- Equal-mass ("quantile") bins by default. Caries probabilities pile up near 0
  on a mostly-sound test set, so equal-width bins leave the upper bins with a
  handful of images each and ECE becomes noise from those few bins.
- ECE is biased upward on small samples: even a perfectly calibrated model scores
  ECE > 0, because each bin's observed rate is a noisy estimate. At this
  project's test-set sizes that floor is several points. `ece_noise_floor`
  simulates outcomes from the predicted probabilities themselves (so the model is
  perfectly calibrated by construction) and returns the 95th percentile of the ECE
  such a model gets. An observed ECE below that is not evidence of miscalibration.
- Logistic recalibration intercept and slope (Cox, 1958) are reported alongside,
  because they need no binning and say *how* the model is miscalibrated:
  slope < 1 means overconfident (too extreme), slope > 1 underconfident;
  intercept != 0 means systematic over- or under-prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import expit, logit

from dcai.eval.bootstrap import Estimate, grouped_bootstrap

Strategy = Literal["quantile", "uniform"]
_EPS = 1e-6


def _check(probs: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray):
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    if p.ndim != 1 or p.shape != y.shape or p.size == 0:
        raise ValueError("probs and y must be non-empty 1-D arrays of the same length")
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ValueError("probs must be finite and in [0, 1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be binary 0/1")
    return p, y.astype(float)


def _bin_index(p: np.ndarray, n_bins: int, strategy: Strategy) -> tuple[np.ndarray, np.ndarray]:
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        # Tied probabilities collapse edges; fewer, populated bins beat empty ones.
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return np.searchsorted(edges[1:-1], p, side="right"), edges


def brier_score(probs: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray) -> float:
    p, y = _check(probs, y)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probs: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: Strategy = "quantile",
) -> float:
    p, y = _check(probs, y)
    return _ece(p, y, n_bins, strategy)


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int, strategy: Strategy) -> float:
    idx, _ = _bin_index(p, n_bins, strategy)
    # sum over bins of (count/n) * |mean_p - mean_y| == sum |sum_p - sum_y| / n
    gap = np.bincount(idx, weights=p) - np.bincount(idx, weights=y)
    return float(np.abs(gap).sum() / p.size)


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_predicted: float
    observed: float


@dataclass(frozen=True)
class ReliabilityCurve:
    strategy: str
    bins: tuple[ReliabilityBin, ...]  # populated bins only


def reliability_curve(
    probs: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: Strategy = "quantile",
) -> ReliabilityCurve:
    p, y = _check(probs, y)
    idx, edges = _bin_index(p, n_bins, strategy)
    bins = []
    for b in range(max(len(edges) - 1, 1)):
        sel = idx == b
        if sel.any():
            lo, hi = (edges[b], edges[b + 1]) if len(edges) > 1 else (edges[0], edges[0])
            bins.append(ReliabilityBin(
                float(lo), float(hi), int(sel.sum()), float(p[sel].mean()), float(y[sel].mean())
            ))
    return ReliabilityCurve(strategy, tuple(bins))


def recalibration(
    probs: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray
) -> tuple[float, float]:
    """(intercept, slope) of logit P(y=1) = a + b * logit(p). NaN if not estimable.

    Fitted by Newton's method rather than sklearn so the result is unpenalised
    and degenerate fits (separation, constant input) come back as NaN instead
    of a convergence warning and a silently regularised number.
    """
    p, y = _check(probs, y)
    if y.min() == y.max():
        return float("nan"), float("nan")
    x = logit(np.clip(p, _EPS, 1 - _EPS))
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        mu = expit(design @ beta)
        hessian = design.T @ (design * (mu * (1 - mu))[:, None])
        try:
            step = np.linalg.solve(hessian, design.T @ (y - mu))
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta = beta + step
        # A diverging fit (separation) never meets this, and NaN fails it too.
        if np.abs(step).max() < 1e-10:
            return float(beta[0]), float(beta[1])
    return float("nan"), float("nan")


def ece_noise_floor(
    probs: Sequence[float] | np.ndarray,
    *,
    seed: int,
    n_bins: int = 10,
    strategy: Strategy = "quantile",
    n_sim: int = 1000,
    quantile: float = 0.95,
) -> float:
    """ECE a perfectly calibrated model would exceed only (1 - quantile) of the time."""
    p, _ = _check(probs, np.zeros(len(probs)))
    rng = np.random.default_rng(seed)
    sims = [_ece(p, (rng.random(p.size) < p).astype(float), n_bins, strategy) for _ in range(n_sim)]
    return float(np.quantile(sims, quantile))


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    prevalence: float
    mean_predicted: float
    n_bins: int
    strategy: str
    ece: Estimate
    ece_noise_floor: float  # 95th pct of ECE under perfect calibration at this n
    brier: Estimate
    intercept: Estimate | None  # None when the recalibration fit is not estimable
    slope: Estimate | None
    curve: ReliabilityCurve

    @property
    def ece_exceeds_noise_floor(self) -> bool:
        return self.ece.value > self.ece_noise_floor


def calibration_report(
    probs: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    seed: int,
    n_bins: int = 10,
    strategy: Strategy = "quantile",
    n_boot: int = 2000,
    n_sim: int = 1000,
) -> CalibrationReport:
    p, yf = _check(probs, y)
    groups = np.asarray(groups)
    if groups.shape != p.shape:
        raise ValueError("groups must have one entry per prediction")

    def boot(stat) -> Estimate:
        return grouped_bootstrap(stat, groups, seed=seed, n_boot=n_boot)

    intercept = slope = None
    if np.isfinite(recalibration(p, yf)).all():
        intercept = boot(lambda i: recalibration(p[i], yf[i])[0])
        slope = boot(lambda i: recalibration(p[i], yf[i])[1])

    return CalibrationReport(
        n=p.size,
        prevalence=float(yf.mean()),
        mean_predicted=float(p.mean()),
        n_bins=n_bins,
        strategy=strategy,
        ece=boot(lambda i: _ece(p[i], yf[i], n_bins, strategy)),
        ece_noise_floor=ece_noise_floor(p, seed=seed, n_bins=n_bins, strategy=strategy,
                                        n_sim=n_sim),
        brier=boot(lambda i: float(np.mean((p[i] - yf[i]) ** 2))),
        intercept=intercept,
        slope=slope,
        curve=reliability_curve(p, yf, n_bins=n_bins, strategy=strategy),
    )
