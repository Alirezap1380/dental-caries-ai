from __future__ import annotations

import numpy as np
import pytest
from conftest import record
from sklearn.metrics import cohen_kappa_score

from dcai.data.schema import Annotation, CariesStage, Finding
from dcai.eval.agreement import (
    cohen_kappa,
    fleiss_agreement,
    fleiss_kappa,
    kappa_sweep,
    krippendorff_agreement,
    krippendorff_alpha,
    model_vs_readers,
    pairwise_agreement,
)
from dcai.eval.bootstrap import difference
from dcai.eval.ratings import (
    MISSING,
    RatingMatrix,
    consensus_grades,
    image_level_ratings,
    tooth_grades,
    tooth_level_ratings,
    unit_id,
)
from dcai.eval.scales import DENTEX_DEPTH, ICCMS_STAGE, OrdinalScale

N_BOOT = 300


# --- scales -----------------------------------------------------------------------


def test_depth_scale_grades() -> None:
    anns = [Annotation(Finding.CARIES, "r"), Annotation(Finding.DEEP_CARIES, "r"),
            Annotation(Finding.IMPLANT, "r")]
    assert DENTEX_DEPTH.image_grade(anns) == 2
    assert DENTEX_DEPTH.image_grade(anns[:1]) == 1
    assert DENTEX_DEPTH.image_grade([anns[2]]) == 0
    assert DENTEX_DEPTH.image_grade([]) == 0
    assert DENTEX_DEPTH.grade(Annotation(Finding.HEALTHY, "r")) == 0
    assert list(DENTEX_DEPTH.lesion_levels) == [1, 2]


def test_iccms_scale() -> None:
    assert ICCMS_STAGE.categories == ("sound", "E1", "E2", "D1", "D2", "D3")
    assert ICCMS_STAGE.grade(Annotation(Finding.CARIES, "r", stage=CariesStage.D1)) == 3
    assert ICCMS_STAGE.grade(Annotation(Finding.HEALTHY, "r")) == 0
    assert ICCMS_STAGE.grade(Annotation(Finding.IMPLANT, "r")) is None
    with pytest.raises(ValueError, match="unstaged"):
        ICCMS_STAGE.grade(Annotation(Finding.CARIES, "r"))


def test_scale_must_start_with_sound() -> None:
    with pytest.raises(ValueError, match="sound"):
        OrdinalScale("bad", ("caries", "deep"), lambda a: None)


# --- cohen_kappa ------------------------------------------------------------------


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
@pytest.mark.parametrize("seed", range(5))
def test_cohen_matches_sklearn(weights: str | None, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, size=80)
    b = np.where(rng.random(80) < 0.6, a, rng.integers(0, 3, size=80))
    expected = cohen_kappa_score(a, b, labels=[0, 1, 2], weights=weights)
    assert cohen_kappa(a, b, 3, weights) == pytest.approx(expected)


def test_cohen_perfect_and_chance() -> None:
    a = np.array([0, 1, 2, 0, 1, 2])
    assert cohen_kappa(a, a, 3) == pytest.approx(1.0)
    assert cohen_kappa(np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0]), 2) < 0


def test_cohen_ignores_missing() -> None:
    a = np.array([0, 1, MISSING, 1])
    b = np.array([0, 1, 0, MISSING])
    assert cohen_kappa(a, b, 2) == pytest.approx(1.0)


def test_cohen_undefined_cases_are_nan() -> None:
    assert np.isnan(cohen_kappa(np.zeros(5, int), np.zeros(5, int), 2))
    assert np.isnan(cohen_kappa(np.array([MISSING]), np.array([0]), 2))


def test_unknown_weights() -> None:
    with pytest.raises(ValueError, match="unknown weights"):
        cohen_kappa(np.array([0, 1]), np.array([0, 1]), 2, "cubic")  # type: ignore[arg-type]


# --- fleiss_kappa -----------------------------------------------------------------

# Fleiss (1971) worked example as reproduced on Wikipedia: 10 units, 14 raters,
# 5 categories, kappa = 0.210.
FLEISS_COUNTS = np.array([
    [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
])


def test_fleiss_textbook_example() -> None:
    ratings = np.array([np.repeat(np.arange(5), row) for row in FLEISS_COUNTS])
    assert fleiss_kappa(ratings, 5) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_two_raters_close_to_cohen() -> None:
    a = np.array([0, 1, 1, 0, 2, 2, 1, 0])
    b = np.array([0, 1, 0, 1, 2, 2, 1, 0])
    assert fleiss_kappa(np.stack([a, b], 1), 3) == pytest.approx(cohen_kappa(a, b, 3))


def test_fleiss_rejects_incomplete_design() -> None:
    with pytest.raises(ValueError, match="every rater on every unit"):
        fleiss_kappa(np.array([[0, MISSING], [1, 1]]), 2)


def test_fleiss_degenerate() -> None:
    with pytest.raises(ValueError, match="two raters"):
        fleiss_kappa(np.array([[0], [1]]), 2)
    assert np.isnan(fleiss_kappa(np.empty((0, 3), int), 2))
    assert np.isnan(fleiss_kappa(np.zeros((4, 3), int), 2))


# --- krippendorff_alpha -------------------------------------------------------------

# Krippendorff (2011), "Computing Krippendorff's alpha-reliability": 4 coders,
# 12 units, values 1-5, with missing data. Published: nominal 0.743,
# ordinal 0.815, interval 0.849.
_ = MISSING
KRIPP = np.array([
    [1, 2, 3, 3, 2, 1, 4, 1, 2, _, _, _],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, _, 3],
    [_, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, _],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, _],
]).T
KRIPP = np.where(KRIPP == MISSING, MISSING, KRIPP - 1)  # values 1-5 -> categories 0-4


@pytest.mark.parametrize(
    "metric, expected", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)]
)
def test_krippendorff_published_example(metric: str, expected: float) -> None:
    assert krippendorff_alpha(KRIPP, 5, metric) == pytest.approx(expected, abs=5e-4)


def test_krippendorff_close_to_fleiss_on_complete_design() -> None:
    m, _truth = panel(n_patients=500)
    nominal = krippendorff_alpha(m.ratings, 3, "nominal")
    assert nominal == pytest.approx(fleiss_kappa(m.ratings, 3), abs=0.005)


def test_krippendorff_degenerate() -> None:
    assert np.isnan(krippendorff_alpha(np.array([[0, MISSING], [1, MISSING]]), 2))
    assert np.isnan(krippendorff_alpha(np.zeros((4, 3), int), 2))
    with pytest.raises(ValueError, match="unknown metric"):
        krippendorff_alpha(KRIPP, 5, "ratio")  # type: ignore[arg-type]


# --- units --------------------------------------------------------------------------


def reads(*by_image: dict[str, list[tuple[Finding, int]]], teeth=(16, 36, 46)) -> list:
    """One record per dict: reader -> [(finding, tooth)] (empty list = read, found nothing)."""
    recs = []
    for i, by_reader in enumerate(by_image):
        anns = [Annotation(f, r, tooth_fdi=t) for r, marks in by_reader.items() for f, t in marks]
        recs.append(record(f"i{i}", f"p{i}", readers=set(by_reader), annotations=anns,
                           teeth_present=set(teeth)))
    return recs


def test_tooth_ratings_pair_readers_on_the_same_tooth() -> None:
    recs = reads(
        {"r1": [(Finding.CARIES, 36), (Finding.DEEP_CARIES, 36)], "r2": [(Finding.CARIES, 46)]},
        {"r1": []},
    )
    m = tooth_level_ratings(recs, DENTEX_DEPTH)
    assert m.raters == ("r1", "r2")
    assert m.item_ids[:3] == (unit_id("i0", 16), unit_id("i0", 36), unit_id("i0", 46))
    # Most severe lesion per tooth; unflagged teeth are sound; unread is MISSING.
    np.testing.assert_array_equal(
        m.ratings, [[0, 0], [2, 0], [0, 1], [0, MISSING], [0, MISSING], [0, MISSING]]
    )
    assert m.groups == ("ds/p0",) * 3 + ("ds/p1",) * 3


def test_tooth_ratings_need_an_inventory() -> None:
    recs = [record("i0", "p0")]
    with pytest.raises(ValueError, match="no tooth inventory"):
        tooth_level_ratings(recs, DENTEX_DEPTH)


def test_lesion_without_tooth_number_cannot_be_placed() -> None:
    rec = record("i0", "p0", teeth_present={36},
                 annotations=[Annotation(Finding.CARIES, "consensus")])
    with pytest.raises(ValueError, match="no tooth_fdi"):
        tooth_grades(rec, "consensus", DENTEX_DEPTH)


def test_non_scale_findings_ignored_on_teeth() -> None:
    rec = record("i0", "p0", teeth_present={36},
                 annotations=[Annotation(Finding.IMPLANT, "consensus")])
    assert tooth_grades(rec, "consensus", DENTEX_DEPTH) == {36: 0}


def test_image_level_fallback() -> None:
    recs = reads({"r1": [(Finding.CARIES, 36)], "r2": []}, {"r1": [(Finding.DEEP_CARIES, 16)]})
    m = image_level_ratings(recs, DENTEX_DEPTH)
    np.testing.assert_array_equal(m.ratings, [[1, 0], [2, MISSING]])
    assert image_level_ratings(recs, DENTEX_DEPTH, raters=["r2"]).raters == ("r2",)


def test_rating_matrix_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        RatingMatrix(("a",), ("g",), ("r",), np.zeros((2, 1)), DENTEX_DEPTH)
    with pytest.raises(ValueError, match="groups"):
        RatingMatrix(("a",), (), ("r",), np.zeros((1, 1)), DENTEX_DEPTH)
    with pytest.raises(ValueError, match="duplicate"):
        RatingMatrix(("a",), ("g",), ("r", "r"), np.zeros((1, 2)), DENTEX_DEPTH)
    with pytest.raises(ValueError, match="out of range"):
        RatingMatrix(("a",), ("g",), ("r",), np.full((1, 1), 3), DENTEX_DEPTH)


def test_with_rater_requires_full_coverage() -> None:
    m = tooth_level_ratings(reads({"r1": []}), DENTEX_DEPTH)
    m2 = m.with_rater("model", dict.fromkeys(m.item_ids, 1))
    np.testing.assert_array_equal(m2.column("model"), [1, 1, 1])
    with pytest.raises(ValueError, match="no grade"):
        m.with_rater("model", {m.item_ids[0]: 1})


# --- consensus ----------------------------------------------------------------------


def test_consensus_is_strict_majority() -> None:
    ratings = np.array([
        [0, 1, 1],  # majority caries
        [1, 2, MISSING],  # 1-1 split: the milder call
        [2, 2, 0],
        [1, MISSING, MISSING],  # one reader only
        [0, 1, 2],  # no majority on any one level: lower median
    ])
    np.testing.assert_array_equal(consensus_grades(ratings), [1, 1, 2, MISSING, 1])
    assert consensus_grades(ratings, min_raters=1)[3] == 1


# --- simulated reader panels ------------------------------------------------------


def noisy_reader(truth: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    return np.where(rng.random(truth.size) < noise, rng.integers(0, 3, truth.size), truth)


def panel(
    n_patients: int = 200, n_readers: int = 4, noise: float = 0.3, seed: int = 0
) -> tuple[RatingMatrix, np.ndarray]:
    """Readers who each see the truth with independent noise; two units per patient."""
    rng = np.random.default_rng(seed)
    n = n_patients * 2
    truth = rng.choice(3, size=n, p=[0.6, 0.25, 0.15])
    m = RatingMatrix(
        item_ids=tuple(f"i{i}" for i in range(n)),
        groups=tuple(f"p{i // 2}" for i in range(n)),
        raters=tuple(f"r{j}" for j in range(n_readers)),
        ratings=np.stack([noisy_reader(truth, rng, noise) for _ in range(n_readers)], 1),
        scale=DENTEX_DEPTH,
    )
    return m, truth


def test_pairwise_agreement_all_pairs() -> None:
    m, _ = panel()
    pairs = pairwise_agreement(m, seed=0, n_boot=N_BOOT)
    assert len(pairs) == 6
    for p in pairs:
        assert p.kappa is not None and p.n_items == 400
        assert 0.3 < p.kappa.value < 0.9


def test_pairwise_with_too_little_overlap() -> None:
    ratings = np.array([[0, MISSING], [1, MISSING], [MISSING, 1], [2, 2]])
    m = RatingMatrix(("a", "b", "c", "d"), ("p0", "p1", "p2", "p3"), ("r1", "r2"),
                     ratings, DENTEX_DEPTH)
    (pair,) = pairwise_agreement(m, seed=0, n_boot=N_BOOT)
    assert pair.n_items == 1 and pair.kappa is None


def test_fleiss_and_krippendorff_estimates() -> None:
    m, _ = panel()
    f = fleiss_agreement(m, seed=0, n_boot=N_BOOT)
    a = krippendorff_agreement(m, seed=0, n_boot=N_BOOT, metric="nominal")
    assert f.lo < f.value < f.hi
    assert a.value == pytest.approx(f.value, abs=0.01)


def test_incomplete_design_routes_to_krippendorff() -> None:
    m, _ = panel()
    r = m.ratings.copy()
    r[::7, 0] = MISSING
    incomplete = RatingMatrix(m.item_ids, m.groups, m.raters, r, m.scale)
    with pytest.raises(ValueError, match="Krippendorff"):
        fleiss_agreement(incomplete, seed=0, n_boot=N_BOOT)
    est = krippendorff_agreement(incomplete, seed=0, n_boot=N_BOOT)
    assert est.lo < est.value < est.hi


# --- the ceiling ----------------------------------------------------------------------


def with_model(m: RatingMatrix, grades: np.ndarray) -> RatingMatrix:
    return m.with_rater("model", dict(zip(m.item_ids, grades.tolist(), strict=True)))


def test_consensus_like_model_is_not_flagged() -> None:
    # A model that approximates a 3-reader consensus beats every individual
    # reader by construction. Against individuals it would look superhuman;
    # against the matched-variance LOO-consensus ceiling it must not be flagged.
    m, truth = panel(seed=1)
    rng = np.random.default_rng(42)
    fresh = np.stack([noisy_reader(truth, rng, 0.3) for _ in range(3)], 1)
    m = with_model(m, consensus_grades(fresh))
    results = model_vs_readers(m, "model", seed=0, n_boot=N_BOOT)
    assert [c.reader for c in results] == ["r0", "r1", "r2", "r3"]
    for c in results:
        # The old comparator (paired model - individual) would fire...
        assert difference(c.model_kappa, c.individual_kappa, paired=True).lo > 0
        assert not c.exceeds_ceiling  # the corrected one does not
        assert c.difference.value == pytest.approx(c.model_kappa.value - c.ceiling_kappa.value)


def test_model_copying_one_reader_is_flagged() -> None:
    m, _ = panel(seed=2)
    m = with_model(m, m.column("r0"))
    by_reader = {c.reader: c for c in model_vs_readers(m, "model", seed=0, n_boot=N_BOOT)}
    assert by_reader["r0"].exceeds_ceiling
    assert not by_reader["r1"].exceeds_ceiling


def test_individual_level_model_is_below_ceiling() -> None:
    m, truth = panel(seed=3)
    m = with_model(m, noisy_reader(truth, np.random.default_rng(5), 0.3))
    assert all(c.below_ceiling for c in model_vs_readers(m, "model", seed=0, n_boot=N_BOOT))


def test_ceiling_uses_only_units_with_enough_consensus() -> None:
    m, truth = panel(seed=4)
    r = m.ratings.copy()
    r[:100, 1:3] = MISSING  # first 100 units: only r0 and r3 read them
    m = RatingMatrix(m.item_ids, m.groups, m.raters, r, m.scale)
    m = with_model(m, truth)
    by_reader = {c.reader: c for c in model_vs_readers(m, "model", seed=0, n_boot=N_BOOT)}
    assert by_reader["r0"].n_items == 300  # LOO consensus of r3 alone is not enough
    assert by_reader["r1"].n_items == 300  # r1 did not read them


def test_ceiling_needs_three_humans() -> None:
    m, truth = panel(n_readers=2)
    m = with_model(m, truth)
    with pytest.raises(ValueError, match="at least 3 human readers"):
        model_vs_readers(m, "model", seed=0, n_boot=N_BOOT)


def test_model_must_be_a_rater() -> None:
    m, _ = panel()
    with pytest.raises(ValueError, match="not a rater"):
        model_vs_readers(m, "model", seed=0, n_boot=N_BOOT)


# --- kappa sweep ------------------------------------------------------------------


def score_model(truth: np.ndarray, seed: int, separation: float = 1.0):
    """A model with a good ranking whose grades depend on the threshold."""
    rng = np.random.default_rng(seed)
    p = np.clip(0.15 + separation * 0.3 * (truth > 0) + rng.normal(0, 0.12, truth.size), 0, 1)

    def grades_at(t: float) -> np.ndarray:
        return np.where(p >= t, np.where(truth == 2, 2, 1), 0)

    return grades_at


def test_sweep_separates_threshold_choice_from_model_quality() -> None:
    m, truth = panel(seed=6)
    grades_at = score_model(truth, seed=1)
    # A sensitivity-first operating point far below where the readers call lesions.
    sweep = kappa_sweep(m, grades_at, operating_threshold=0.12, seed=0, n_boot=N_BOOT)
    assert sweep.operating_threshold in sweep.thresholds
    assert sweep.model_kappa.shape == sweep.thresholds.shape
    assert sweep.at_best.value == pytest.approx(sweep.model_kappa.max())
    assert sweep.operating_point_costs_agreement
    assert sweep.best_minus_operating.value == pytest.approx(
        sweep.at_best.value - sweep.at_operating.value)
    assert sweep.reaches_ceiling  # the ranking is good: the threshold was the problem


def test_sweep_weak_model_never_reaches_ceiling() -> None:
    m, truth = panel(seed=7)
    rng = np.random.default_rng(3)
    p = rng.random(truth.size)
    sweep = kappa_sweep(m, lambda t: np.where(p >= t, 1, 0), operating_threshold=0.5,
                        seed=0, n_boot=N_BOOT)
    assert not sweep.reaches_ceiling


def test_sweep_at_best_operating_point_has_no_gap() -> None:
    m, truth = panel(seed=8)
    grades_at = score_model(truth, seed=2)
    first = kappa_sweep(m, grades_at, operating_threshold=0.5, seed=0, n_boot=N_BOOT)
    again = kappa_sweep(m, grades_at, operating_threshold=first.best_threshold, seed=0,
                        n_boot=N_BOOT)
    assert again.best_minus_operating.value == pytest.approx(0.0)
    assert not again.operating_point_costs_agreement


def test_sweep_needs_three_readers() -> None:
    m, truth = panel(n_readers=2)
    with pytest.raises(ValueError, match="at least 3 human readers"):
        kappa_sweep(m, lambda t: truth, operating_threshold=0.5, seed=0)
