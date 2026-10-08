from __future__ import annotations

import numpy as np
import pytest

from dcai.data.splits import PatientLeakageError
from dcai.eval.abstention import (
    DeploymentRole,
    OperatingPoint,
    evaluate_operating_point,
    fit_operating_point,
    risk_coverage,
)

N_BOOT = 200
ROLE = {"deployment_role": "second_reader", "constraint": "sensitivity"}
WHY = "Missed caries progress between recalls; 0.9 matches clinician sensitivity on D-stages."


def scored(n: int, seed: int, prefix: str = "p") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    s = np.clip(rng.normal(0.35 + 0.3 * y, 0.15), 0, 1)
    groups = np.array([f"{prefix}{i // 2}" for i in range(n)])
    return s, y, groups


# --- risk_coverage ----------------------------------------------------------------


def test_risk_coverage_hand_computed() -> None:
    curve = risk_coverage([0.9, 0.8, 0.7, 0.6], [True, False, True, True])
    np.testing.assert_allclose(curve.coverage, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(curve.risk, [0.0, 0.5, 1 / 3, 0.25])
    assert curve.aurc == pytest.approx(0.25 * (0 + 0.5 + 1 / 3 + 0.25))


def test_ties_are_accepted_together() -> None:
    curve = risk_coverage([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
    np.testing.assert_allclose(curve.coverage, [0.25, 0.75, 1.0])
    np.testing.assert_allclose(curve.risk, [0.0, 1 / 3, 0.5])


def test_informative_confidence_beats_random() -> None:
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.8
    informative = correct + rng.normal(scale=0.5, size=2000)
    useless = rng.random(2000)
    assert risk_coverage(informative, correct).aurc < risk_coverage(useless, correct).aurc
    assert risk_coverage(useless, correct).aurc == pytest.approx(0.2, abs=0.03)


def test_risk_coverage_validation() -> None:
    with pytest.raises(ValueError, match="same length"):
        risk_coverage([0.1], [True, False])


# --- fit_operating_point ------------------------------------------------------------


def test_threshold_meets_sensitivity_target_at_highest_possible_threshold() -> None:
    s, y, g = scored(1000, 0)
    op = fit_operating_point(s, y, g, **ROLE, target=0.9, rationale=WHY)
    pos = s[y == 1]
    assert (pos >= op.threshold).mean() >= 0.9
    assert op.fitted_value == pytest.approx((pos >= op.threshold).mean())
    # Any higher threshold misses the target.
    higher = pos[pos > op.threshold].min()
    assert (pos >= higher).mean() < 0.9
    assert op.margin == 0.0


def test_margin_refers_the_target_fraction() -> None:
    s, y, g = scored(1000, 0)
    op = fit_operating_point(s, y, g, **ROLE, target=0.9, rationale=WHY, coverage_target=0.8)
    accepted = np.abs(s - op.threshold) >= op.margin
    assert accepted.mean() == pytest.approx(0.8, abs=0.002)


@pytest.mark.parametrize("rationale", ["", "   "])
def test_rationale_is_required(rationale: str) -> None:
    s, y, g = scored(100, 0)
    with pytest.raises(ValueError, match="rationale"):
        fit_operating_point(s, y, g, **ROLE, target=0.9, rationale=rationale)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"target": 1.0}, "target"),
        ({"target": 0.9, "coverage_target": 0.0}, "coverage_target"),
        ({"target": 0.9, "constraint": "ppv"}, "constraint"),
        ({"target": 0.9, "deployment_role": "oracle"}, "oracle"),
    ],
)
def test_fit_validation(kwargs: dict, match: str) -> None:
    s, y, g = scored(100, 0)
    with pytest.raises(ValueError, match=match):
        fit_operating_point(s, y, g, rationale=WHY, **{**ROLE, **kwargs})


def test_autonomous_triage_must_bind_specificity() -> None:
    s, y, g = scored(400, 0)
    with pytest.raises(ValueError, match="specificity must be the binding constraint"):
        fit_operating_point(s, y, g, deployment_role="autonomous_triage",
                            constraint="sensitivity", target=0.9, rationale=WHY)
    op = fit_operating_point(s, y, g, deployment_role=DeploymentRole.AUTONOMOUS_TRIAGE,
                             constraint="specificity", target=0.95, rationale=WHY)
    neg = s[y == 0]
    assert (neg < op.threshold).mean() >= 0.95
    assert op.fitted_value == pytest.approx((neg < op.threshold).mean())
    # The lowest such threshold: one negative more and it would fall short.
    assert (neg < neg[neg < op.threshold].max()).mean() < 0.95


def test_specificity_needs_negatives() -> None:
    with pytest.raises(ValueError, match="no negatives"):
        fit_operating_point([0.1, 0.2], [1, 1], ["a", "b"], deployment_role="second_reader",
                            constraint="specificity", target=0.9, rationale=WHY)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"target": 0.0}, "target"),
        ({"coverage_target": 1.5}, "coverage_target"),
        ({"rationale": " "}, "rationale"),
    ],
)
def test_operating_point_validation(kwargs: dict, match: str) -> None:
    fields = {"threshold": 0.5, "margin": 0.0, "deployment_role": "second_reader",
              "constraint": "sensitivity", "target": 0.9, "coverage_target": 1.0,
              "rationale": WHY, "fitted_value": 0.9, "fitted_on_groups": frozenset()}
    fields.update(kwargs)
    with pytest.raises(ValueError, match=match):
        OperatingPoint(**fields)


def test_fit_needs_positives() -> None:
    with pytest.raises(ValueError, match="no positives"):
        fit_operating_point([0.1, 0.2], [0, 0], ["a", "b"], **ROLE, target=0.9, rationale=WHY)


@pytest.mark.parametrize(
    "s, y, g, match",
    [
        ([0.1, np.inf], [0, 1], ["a", "b"], "finite"),
        ([0.1, 0.2], [0, 2], ["a", "b"], "binary"),
        ([0.1, 0.2], [0, 1], ["a"], "groups"),
        ([], [], [], "non-empty"),
    ],
)
def test_input_validation(s: list, y: list, g: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fit_operating_point(s, y, g, **ROLE, target=0.9, rationale=WHY)


# --- evaluate_operating_point -------------------------------------------------------


def test_evaluating_on_fitting_patients_is_leakage() -> None:
    s, y, g = scored(400, 0)
    op = fit_operating_point(s, y, g, **ROLE, target=0.9, rationale=WHY)
    with pytest.raises(PatientLeakageError, match="fitted on"):
        evaluate_operating_point(op, s, y, g, seed=0, n_boot=N_BOOT)
    # E0 opt-in: allowed, and the overlap is counted rather than hidden.
    r = evaluate_operating_point(op, s, y, g, seed=0, n_boot=N_BOOT, allow_patient_overlap=True)
    assert r.patient_overlap == 200


def test_held_out_evaluation() -> None:
    s_val, y_val, g_val = scored(1000, 0, prefix="val")
    s_test, y_test, g_test = scored(1000, 1, prefix="test")
    op = fit_operating_point(s_val, y_val, g_val, **ROLE, target=0.9, rationale=WHY,
                             coverage_target=0.8)
    r = evaluate_operating_point(op, s_test, y_test, g_test, seed=0, n_boot=N_BOOT)
    assert r.n == 1000 and r.operating_point is op
    assert r.coverage.value == pytest.approx(0.8, abs=0.05)
    # Referring the borderline cases helps the accepted ones...
    assert r.sensitivity_accepted.value > r.sensitivity_no_abstention.value
    assert r.specificity_accepted.value > r.specificity_no_abstention.value
    # ...and the fraction of all positives silently cleared is what's left over.
    pos, acc = y_test == 1, np.abs(s_test - op.threshold) >= op.margin
    assert r.missed_positive_rate.value == pytest.approx(
        (pos & acc & (s_test < op.threshold)).sum() / pos.sum()
    )
    assert r.sensitivity_no_abstention.lo < 0.9 < r.sensitivity_no_abstention.hi


def test_evaluate_validation() -> None:
    s, y, g = scored(100, 0, prefix="val")
    op = fit_operating_point(s, y, g, **ROLE, target=0.9, rationale=WHY)
    with pytest.raises(ValueError, match="groups"):
        evaluate_operating_point(op, [0.5, 0.5], [0, 1], ["x"], seed=0)


def test_empty_denominator_is_reported_not_raised() -> None:
    # Every positive sits inside the referral band: no positive is auto-decided.
    s_val, y_val, g_val = scored(400, 0, prefix="val")
    op = fit_operating_point(s_val, y_val, g_val, **ROLE, target=0.9, rationale=WHY,
                             coverage_target=0.5)
    s = np.array([op.threshold] * 4 + [0.0] * 4)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    r = evaluate_operating_point(op, s, y, [f"t{i}" for i in range(8)], seed=0, n_boot=N_BOOT)
    assert r.sensitivity_accepted is None
    assert r.specificity_accepted is not None and r.sensitivity_no_abstention.value == 1.0
