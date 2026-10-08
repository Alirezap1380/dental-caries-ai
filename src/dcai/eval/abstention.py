"""Selective prediction: when should the model say "refer this one to a clinician"?

Two pieces:

1. Risk-coverage curves (and their area, AURC). Sort cases by confidence, accept
   the most confident fraction, and plot the error rate among accepted cases
   against that fraction. A useful confidence signal makes risk fall as coverage
   falls.

2. A clinical operating point. The decision threshold is set to meet a binding
   constraint (a sensitivity or a specificity target), and an abstention band
   around it is sized to a target coverage. Both are fitted on validation
   patients and evaluated on disjoint test patients.

Caries inverts the usual screening asymmetry. A missed early lesion is caught
later and restored instead of arrested: a bounded, partly recoverable harm. A
false positive that is acted on drills a sound tooth: permanent loss of
structure and the start of the restorative cycle. So which error binds depends on
what a flag *does*, and the operating point has to declare it:

- SECOND_READER: a dentist adjudicates every flag. A false positive costs review
  time, so a sensitivity constraint is defensible.
- AUTONOMOUS_TRIAGE: a flag drives treatment. A false positive costs tooth
  structure, so specificity must be the binding constraint. Any other choice is
  rejected at construction.

Enforced in code:
- An `OperatingPoint` needs a deployment role and a written rationale. The
  target is a clinical judgement, and the write-up has to state and defend it.
- It remembers the patients it was fitted on, and evaluating it on any of those
  patients raises `PatientLeakageError`. Tuning a threshold on the test set is
  the quietest leak there is.
- Evaluation reports `missed_positive_rate`: positives the system cleared
  *without* referring them. Sensitivity among accepted cases alone can look
  excellent while abstention hides the hard positives in the referral pile.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from dcai.data.splits import PatientLeakageError
from dcai.eval.bootstrap import Estimate, grouped_bootstrap


def _check(scores: Sequence[float] | np.ndarray, y: Sequence[int] | np.ndarray):
    s, y = np.asarray(scores, dtype=float), np.asarray(y)
    if s.ndim != 1 or s.shape != y.shape or s.size == 0:
        raise ValueError("scores and y must be non-empty 1-D arrays of the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be binary 0/1")
    return s, y.astype(bool)


@dataclass(frozen=True)
class RiskCoverageCurve:
    coverage: np.ndarray  # fraction accepted, increasing, ending at 1
    risk: np.ndarray  # error rate among accepted cases
    aurc: float


def risk_coverage(
    confidence: Sequence[float] | np.ndarray, correct: Sequence[bool] | np.ndarray
) -> RiskCoverageCurve:
    """Risk-coverage curve. Tied confidences are accepted or refused together.

    AURC is the area under the step curve. Lower is better; it is bounded below
    by the risk of an oracle that refuses its errors first.
    """
    c = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if c.ndim != 1 or c.shape != ok.shape or c.size == 0:
        raise ValueError("confidence and correct must be non-empty 1-D arrays of the same length")
    order = np.argsort(-c, kind="stable")
    c, errors = c[order], np.cumsum(~ok[order])
    # Last index of each block of tied confidences.
    ends = np.flatnonzero(np.r_[c[1:] != c[:-1], True])
    accepted = ends + 1
    risk = errors[ends] / accepted
    coverage = accepted / c.size
    widths = np.diff(np.r_[0.0, coverage])
    return RiskCoverageCurve(coverage, risk, float((widths * risk).sum()))


class DeploymentRole(str, Enum):
    SECOND_READER = "second_reader"  # a dentist adjudicates every flag
    AUTONOMOUS_TRIAGE = "autonomous_triage"  # a flag drives treatment without review


Constraint = Literal["sensitivity", "specificity"]


def _check_spec(
    role: DeploymentRole, constraint: str, target: float, coverage_target: float, rationale: str
) -> None:
    if not rationale.strip():
        raise ValueError("an operating point needs a stated clinical rationale")
    if constraint not in ("sensitivity", "specificity"):
        raise ValueError(f"constraint must be 'sensitivity' or 'specificity', got {constraint!r}")
    if not 0.0 < target < 1.0:
        raise ValueError("target must be in (0, 1)")
    if not 0.0 < coverage_target <= 1.0:
        raise ValueError("coverage_target must be in (0, 1]")
    if role is DeploymentRole.AUTONOMOUS_TRIAGE and constraint != "specificity":
        raise ValueError(
            "autonomous triage: a flag drives treatment, so a false positive costs sound tooth "
            "structure and specificity must be the binding constraint"
        )


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float  # predict positive at score >= threshold
    margin: float  # refer when |score - threshold| < margin
    deployment_role: DeploymentRole
    constraint: Constraint  # which error rate is held to `target`
    target: float
    coverage_target: float
    rationale: str
    fitted_value: float  # the constrained metric achieved on the fitting data
    fitted_on_groups: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "deployment_role", DeploymentRole(self.deployment_role))
        _check_spec(self.deployment_role, self.constraint, self.target, self.coverage_target,
                    self.rationale)


def fit_operating_point(
    scores: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    deployment_role: DeploymentRole | str,
    constraint: Constraint,
    target: float,
    rationale: str,
    coverage_target: float = 1.0,
) -> OperatingPoint:
    """Threshold meeting the binding constraint with least cost to the other error, then a band.

    Sensitivity constraint: the highest threshold with sensitivity >= target.
    Specificity constraint: the lowest threshold with specificity >= target.
    The band is the smallest margin around the threshold that refers
    (1 - coverage_target) of fitting cases: those nearest the decision boundary,
    where the model is least sure.
    """
    s, y = _check(scores, y)
    groups = np.asarray(groups)
    if groups.shape != s.shape:
        raise ValueError("groups must have one entry per score")
    role = DeploymentRole(deployment_role)
    _check_spec(role, constraint, target, coverage_target, rationale)

    if constraint == "sensitivity":
        if not y.any():
            raise ValueError("no positives to fit a sensitivity target on")
        pos = np.sort(s[y])[::-1]
        threshold = float(pos[math.ceil(target * pos.size) - 1])
        fitted = float((pos >= threshold).mean())
    else:
        if y.all():
            raise ValueError("no negatives to fit a specificity target on")
        neg = np.sort(s[~y])
        # Just above the k-th lowest negative score: those k fall below and are cleared.
        threshold = float(np.nextafter(neg[math.ceil(target * neg.size) - 1], np.inf))
        fitted = float((neg < threshold).mean())
    margin = 0.0
    if coverage_target < 1.0:
        distance = np.sort(np.abs(s - threshold))
        margin = float(distance[math.ceil((1.0 - coverage_target) * s.size) - 1])
        # `< margin` refers strictly-closer cases; nudge so the boundary case is referred too.
        margin = float(np.nextafter(margin, np.inf))
    return OperatingPoint(
        threshold=threshold,
        margin=margin,
        deployment_role=role,
        constraint=constraint,
        target=target,
        coverage_target=coverage_target,
        rationale=rationale,
        fitted_value=fitted,
        fitted_on_groups=frozenset(map(str, groups)),
    )


@dataclass(frozen=True)
class SelectiveResult:
    operating_point: OperatingPoint
    n: int
    # Each rate is None when its denominator is empty on the evaluation data, e.g.
    # every positive fell in the referral band, so none was auto-decided.
    coverage: Estimate | None  # fraction auto-decided (not referred)
    sensitivity_accepted: Estimate | None  # among positives not referred
    specificity_accepted: Estimate | None  # among negatives not referred
    missed_positive_rate: Estimate | None  # positives auto-cleared as negative / all positives
    sensitivity_no_abstention: Estimate | None  # same threshold, nothing referred
    specificity_no_abstention: Estimate | None
    # Patients shared with the fitting set. Always 0 unless the caller opted into
    # the E0 naive-split reproduction, where it is part of the result.
    patient_overlap: int = 0


def evaluate_operating_point(
    op: OperatingPoint,
    scores: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
    allow_patient_overlap: bool = False,
) -> SelectiveResult:
    """Evaluate a frozen operating point on held-out patients.

    `allow_patient_overlap` exists only so experiment E0 can reproduce the naive
    image-level protocol. The overlap is then counted and reported, not hidden.
    """
    s, y = _check(scores, y)
    groups = np.asarray(groups)
    if groups.shape != s.shape:
        raise ValueError("groups must have one entry per score")
    overlap = op.fitted_on_groups & set(map(str, groups))
    if overlap and not allow_patient_overlap:
        raise PatientLeakageError(
            f"operating point was fitted on {len(overlap)} patient(s) present in the "
            f"evaluation set, e.g. {sorted(overlap)[:3]}"
        )

    predicted = s >= op.threshold
    accepted = np.abs(s - op.threshold) >= op.margin

    def rate(num: np.ndarray, den: np.ndarray) -> Estimate | None:
        if not den.any():
            return None

        def stat(idx: np.ndarray) -> float:
            d = den[idx]
            return float(num[idx][d].mean()) if d.any() else float("nan")
        return grouped_bootstrap(stat, groups, seed=seed, n_boot=n_boot)

    everything = np.ones_like(y)
    return SelectiveResult(
        operating_point=op,
        n=s.size,
        coverage=rate(accepted, everything),
        sensitivity_accepted=rate(predicted, y & accepted),
        specificity_accepted=rate(~predicted, ~y & accepted),
        missed_positive_rate=rate(accepted & ~predicted, y),
        sensitivity_no_abstention=rate(predicted, y),
        specificity_no_abstention=rate(~predicted, ~y),
        patient_overlap=len(overlap),
    )
