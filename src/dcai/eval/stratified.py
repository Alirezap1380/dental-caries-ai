"""Stage-stratified detection metrics.

Rule 2: never pool across lesion stage. Detection gets reliably better with
lesion depth. A pooled sensitivity is therefore mostly a statement about deep
lesions, which are the easy and less clinically valuable cases, and it can look
excellent while early-lesion detection is near chance. The headline result is
performance *as a function of* stage.

What can and cannot be stratified:
- Sensitivity and AUC are properties of the lesions in a stratum, so they get
  one estimate per level. AUC is "this stratum vs. sound units".
- Specificity and false positives belong to sound units, which have no stage.
  They are reported once, next to the strata, rather than being divided up
  among them.

`pooled_sensitivity` is included only *beside* the strata, to show how much the
pooling hides. No function here returns a pooled number on its own.

All estimates in one report are bootstrapped with the same seed over the same
patients, so any two of them can be differenced as paired (`depth_gap` is one).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from dcai.eval.bootstrap import Estimate, Statistic, difference, grouped_bootstrap
from dcai.eval.scales import OrdinalScale


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties count half). NaN if one class."""
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return (float(ranks[positive].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass(frozen=True)
class StratumResult:
    level: int
    name: str
    n_units: int
    # None when the stratum is empty in this data. An empty stratum is reported,
    # never dropped: "no E1 lesions in the test set" is itself a result.
    sensitivity: Estimate | None
    auc_vs_sound: Estimate | None  # None when there are no sound units


@dataclass(frozen=True)
class StratifiedReport:
    scale: str
    threshold: float
    strata: tuple[StratumResult, ...]
    n_sound: int
    specificity: Estimate | None
    pooled_sensitivity: Estimate
    # Sensitivity at the deepest non-empty level minus the shallowest, paired.
    # On DENTEX_DEPTH this is a lower bound on the early-lesion gap (see scales.py).
    depth_gap: Estimate | None

    def stratum(self, name: str) -> StratumResult:
        return next(s for s in self.strata if s.name == name)


def stratified_report(
    levels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    scale: OrdinalScale,
    threshold: float,
    seed: int,
    n_boot: int = 2000,
) -> StratifiedReport:
    """Per-stratum sensitivity and AUC, plus specificity on sound units.

    `levels[i]` is unit i's true grade on `scale` (0 = sound); `scores[i]` the
    model's score for it. Units are images (image-level grading) or lesions
    (from `detection.match_lesions`, which have no sound units).
    """
    levels = np.asarray(levels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    if not levels.shape == scores.shape == groups.shape or levels.ndim != 1:
        raise ValueError("levels, scores and groups must be 1-D and the same length")
    if levels.size and (levels.min() < 0 or levels.max() >= scale.n_categories):
        raise ValueError(f"levels out of range for scale {scale.name}")
    # -inf is allowed: it is how a lesion that no detection claimed is scored
    # (never detected at any threshold). NaN is always a bug upstream.
    if np.isnan(scores).any():
        raise ValueError("scores must not be NaN")
    if not (levels > 0).any():
        raise ValueError("no lesions: nothing to stratify")

    def sens_at(mask_of: Callable[[np.ndarray], np.ndarray]) -> Statistic:
        def stat(idx: np.ndarray) -> float:
            sel = mask_of(levels[idx])
            return float((scores[idx][sel] >= threshold).mean()) if sel.any() else float("nan")
        return stat

    def boot(stat: Statistic) -> Estimate:
        return grouped_bootstrap(stat, groups, seed=seed, n_boot=n_boot)

    has_sound = bool((levels == 0).any())
    strata = []
    for level in scale.lesion_levels:
        n = int((levels == level).sum())
        sens = auc_est = None
        if n:
            sens = boot(sens_at(lambda lv, level=level: lv == level))
            if has_sound:
                def auc_stat(idx: np.ndarray, level: int = level) -> float:
                    lv = levels[idx]
                    keep = (lv == level) | (lv == 0)
                    return auc(scores[idx][keep], lv[keep] == level)
                auc_est = boot(auc_stat)
        strata.append(StratumResult(level, scale.categories[level], n, sens, auc_est))

    specificity = None
    if has_sound:
        def spec(idx: np.ndarray) -> float:
            sel = levels[idx] == 0
            return float((scores[idx][sel] < threshold).mean()) if sel.any() else float("nan")
        specificity = boot(spec)

    present = [s for s in strata if s.sensitivity is not None]
    gap = (
        difference(present[-1].sensitivity, present[0].sensitivity, paired=True)
        if len(present) >= 2 else None
    )
    return StratifiedReport(
        scale=scale.name,
        threshold=threshold,
        strata=tuple(strata),
        n_sound=int((levels == 0).sum()),
        specificity=specificity,
        pooled_sensitivity=boot(sens_at(lambda lv: lv > 0)),
        depth_gap=gap,
    )
