from __future__ import annotations

import numpy as np
import pytest
from conftest import record

from dcai.data.schema import Annotation, Finding
from dcai.eval.ratings import MAJORITY, check_inventory, tooth_level_ratings
from dcai.eval.scales import DENTEX_DEPTH
from dcai.eval.subgroups import age_bands, subgroup_report
from dcai.eval.units import ToothPrediction, tooth_table

SOUND = (0.9, 0.08, 0.02)
LESION = (0.2, 0.5, 0.3)


def three_reader_records() -> list:
    marks = [
        Annotation(Finding.CARIES, "r1", tooth_fdi=36),
        Annotation(Finding.CARIES, "r2", tooth_fdi=36),
        Annotation(Finding.DEEP_CARIES, "r3", tooth_fdi=46),
    ]
    return [
        record("i0", "p0", readers={"r1", "r2", "r3"}, annotations=marks,
               teeth_present={36, 46}, age=40.0, sex="female", site_id="A"),
        record("i1", "p1", teeth_present={11}),
    ]


def preds(probs_by_unit: dict[tuple[str, int], tuple[float, ...]]) -> list[ToothPrediction]:
    return [ToothPrediction(i, t, p) for (i, t), p in probs_by_unit.items()]


ALL = {("i0", 36): LESION, ("i0", 46): SOUND, ("i1", 11): SOUND}


# --- ToothPrediction ----------------------------------------------------------------


@pytest.mark.parametrize(
    "fdi, probs, match",
    [
        (19, SOUND, "FDI"),
        (36, (0.5, 0.6, -0.1), "non-negative"),
        (36, (0.5, 0.2, 0.2), "sum to 1"),
        (36, (1.0,), "non-negative"),
    ],
)
def test_prediction_validation(fdi: int, probs: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ToothPrediction("i0", fdi, probs)


# --- tooth_table ------------------------------------------------------------------


def test_majority_reference_and_metadata() -> None:
    t = tooth_table(three_reader_records(), preds(ALL), scale=DENTEX_DEPTH)
    assert t.reference_name == MAJORITY
    assert t.unit_ids == ("i0#36", "i0#46", "i1#11")
    # 2 of 3 readers say caries on 36; only 1 of 3 flags 46.
    assert t.reference.tolist() == [1, 0, 0]
    assert t.has_lesion.tolist() == [1, 0, 0]
    np.testing.assert_allclose(t.p_lesion, [0.8, 0.1, 0.1])
    assert t.predicted_grade(0.5).tolist() == [1, 0, 0]
    assert t.model_grades(0.5) == {"i0#36": 1, "i0#46": 0, "i1#11": 0}
    # At a low operating threshold the sound-leaning teeth are called too, at
    # their most probable lesion grade (caries, not deep).
    assert t.predicted_grade(0.05).tolist() == [1, 1, 1]
    assert t.groups.tolist() == ["ds/p0", "ds/p0", "ds/p1"]
    assert t.sexes.tolist() == ["female", "female", None]
    assert t.sites.tolist() == ["A", "A", None]
    assert t.ages[0] == 40.0 and np.isnan(t.ages[2])
    assert t.n == 3


def test_single_reader_reference() -> None:
    rec = three_reader_records()[0]
    rec.teeth_present = frozenset({11, 21, 36, 46})  # a realistic, mostly sound dentition
    p = preds({("i0", 11): SOUND, ("i0", 21): SOUND, ("i0", 36): LESION, ("i0", 46): SOUND})
    t = tooth_table([rec], p, scale=DENTEX_DEPTH, reference="r3")
    assert t.reference.tolist() == [0, 0, 0, 2]


def test_box_derived_inventory_is_refused() -> None:
    # Inventory == teeth with findings: sound teeth have vanished from the denominator.
    rec = three_reader_records()[0]
    with pytest.raises(ValueError, match="derived from annotation boxes"):
        tooth_table([rec], preds({("i0", 36): LESION, ("i0", 46): SOUND}), scale=DENTEX_DEPTH)
    with pytest.raises(ValueError, match="derived from annotation boxes"):
        tooth_level_ratings([rec], DENTEX_DEPTH)
    assert check_inventory(three_reader_records()) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p[:-1], "no prediction"),
        (lambda p: [*p, ToothPrediction("i1", 21, SOUND)], "not in the inventory"),
        (lambda p: [*p, p[0]], "duplicate"),
        (lambda p: [ToothPrediction("i0", 36, (0.5, 0.5)), *p[1:]], "2 probs"),
    ],
)
def test_table_validation(mutate, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        tooth_table(three_reader_records(), mutate(preds(ALL)), scale=DENTEX_DEPTH)


# --- subgroups ----------------------------------------------------------------------


def test_age_bands() -> None:
    bands = age_bands([10, 18, 34, 35, 64, 65, 90, np.nan])
    assert bands.tolist() == ["<18", "18-34", "18-34", "35-49", "50-64", "65+", "65+", None]


def subgroup_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    levels = rng.choice(3, size=n, p=[0.6, 0.25, 0.15])
    scores = np.clip(0.2 + 0.3 * levels + rng.normal(0, 0.2, n), 0, 1)
    groups = np.array([f"p{i // 4}" for i in range(n)])
    return levels, scores, groups


def test_subgroup_report_is_stratified_per_subgroup() -> None:
    levels, scores, groups = subgroup_data()
    sex = np.where(np.arange(400) % 8 < 4, "female", "male").astype(object)
    r = subgroup_report(sex, levels, scores, groups, attribute="sex", scale=DENTEX_DEPTH,
                        threshold=0.5, seed=0, n_boot=200)
    assert r.not_computable is None and r.coverage == 1.0
    assert [c.value for c in r.cells] == ["female", "male"]
    for c in r.cells:
        assert c.report is not None
        assert [s.name for s in c.report.strata] == ["caries", "deep_caries"]


def test_missing_attribute_is_not_computable() -> None:
    levels, scores, groups = subgroup_data()
    r = subgroup_report([None] * 400, levels, scores, groups, attribute="age",
                        scale=DENTEX_DEPTH, threshold=0.5, seed=0, n_boot=200)
    assert r.cells == () and r.coverage == 0.0
    assert "not computable on this data" in r.not_computable


def test_partial_coverage_and_small_cells() -> None:
    levels, scores, groups = subgroup_data()
    values = np.array([None] * 400, dtype=object)
    values[:200] = "big"
    values[200:204] = "one_patient"  # p50 only
    sound = np.flatnonzero(levels == 0)[-8:]
    values[sound] = "no_lesions"
    r = subgroup_report(values, levels, scores, groups, attribute="x", scale=DENTEX_DEPTH,
                        threshold=0.5, seed=0, n_boot=200)
    assert r.coverage == pytest.approx((200 + 4 + 8) / 400)
    cells = {c.value: c for c in r.cells}
    assert cells["big"].report is not None
    assert cells["one_patient"].reason == "fewer than two patients"
    assert cells["no_lesions"].reason == "no lesions in this subgroup"


def test_subgroup_length_check() -> None:
    with pytest.raises(ValueError, match="same length"):
        subgroup_report(["a"], [1, 2], [0.5, 0.5], ["p", "q"], attribute="x",
                        scale=DENTEX_DEPTH, threshold=0.5, seed=0)
