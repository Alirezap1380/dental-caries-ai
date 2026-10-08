from __future__ import annotations

import json
import math

import numpy as np
import pytest
from conftest import record

from dcai.data.schema import (
    CONSENSUS,
    VALID_FDI,
    Annotation,
    BBox,
    CariesStage,
    Finding,
    InventorySource,
    Modality,
    PatientIdSource,
    RadiographRecord,
    Sex,
    iter_annotations,
    summarise,
)
from dcai.data.synthetic import make_records

# --- BBox ---------------------------------------------------------------------


def test_bbox_geometry() -> None:
    b = BBox(0, 0, 10, 20)
    assert b.area == 200
    assert b.center == (5.0, 10.0)
    np.testing.assert_array_equal(b.as_array(), [0, 0, 10, 20])


@pytest.mark.parametrize("coords", [(0, 0, 0, 10), (0, 0, 10, 0), (5, 5, 1, 10)])
def test_bbox_rejects_degenerate(coords: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        BBox(*coords)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_bbox_rejects_non_finite(bad: float) -> None:
    # NaN fails every comparison, so without the explicit check it would pass
    # the degenerate-box test and poison every downstream IoU.
    with pytest.raises(ValueError, match="non-finite"):
        BBox(0, 0, bad, 10)


def test_bbox_iou() -> None:
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(BBox(20, 20, 30, 30)) == 0.0
    assert a.iou(BBox(10, 0, 20, 10)) == 0.0  # touching edges, zero-area overlap
    # 5x10 overlap, union 150
    assert a.iou(BBox(5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert a.iou(BBox(5, 0, 15, 10)) == BBox(5, 0, 15, 10).iou(a)


# --- CariesStage --------------------------------------------------------------


def test_stage_initial_is_enamel_only() -> None:
    assert {s for s in CariesStage if s.is_initial} == {CariesStage.E1, CariesStage.E2}
    assert not CariesStage.NONE.is_initial


# --- Annotation ---------------------------------------------------------------


def test_valid_fdi_set() -> None:
    assert len(VALID_FDI) == 32
    assert min(VALID_FDI) == 11 and max(VALID_FDI) == 48


@pytest.mark.parametrize("fdi", [0, 10, 19, 50, 51, 85, 49])
def test_annotation_rejects_invalid_fdi(fdi: int) -> None:
    with pytest.raises(ValueError, match="FDI"):
        Annotation(Finding.CARIES, "r1", tooth_fdi=fdi)


@pytest.mark.parametrize("conf", [-0.1, 1.1, math.nan])
def test_annotation_rejects_bad_confidence(conf: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Annotation(Finding.CARIES, "r1", confidence=conf)


def test_annotation_requires_annotator() -> None:
    with pytest.raises(ValueError, match="annotator_id"):
        Annotation(Finding.CARIES, "")


def test_annotation_coerces_strings_to_enums() -> None:
    # A raw "caries" would compare equal to Finding.CARIES but fail `is` checks.
    a = Annotation("caries", "r1", stage=3)  # type: ignore[arg-type]
    assert a.finding is Finding.CARIES
    assert a.stage is CariesStage.D1


def test_unstaged_caries_is_allowed_and_distinct_from_sound() -> None:
    a = Annotation(Finding.CARIES, "r1")
    assert a.stage is None


def test_caries_with_stage_none_is_rejected() -> None:
    with pytest.raises(ValueError, match="sound surface"):
        Annotation(Finding.CARIES, "r1", stage=CariesStage.NONE)


@pytest.mark.parametrize("stage", [CariesStage.E1, CariesStage.E2])
def test_deep_caries_cannot_be_enamel(stage: CariesStage) -> None:
    with pytest.raises(ValueError, match="deep_caries"):
        Annotation(Finding.DEEP_CARIES, "r1", stage=stage)


def test_deep_caries_dentine_stage_ok() -> None:
    assert Annotation(Finding.DEEP_CARIES, "r1", stage=CariesStage.D3).stage is CariesStage.D3


def test_healthy_only_takes_stage_none() -> None:
    assert Annotation(Finding.HEALTHY, "r1", stage=CariesStage.NONE).stage is CariesStage.NONE
    with pytest.raises(ValueError, match="healthy"):
        Annotation(Finding.HEALTHY, "r1", stage=CariesStage.D1)


def test_stage_on_non_caries_finding_rejected() -> None:
    with pytest.raises(ValueError, match="only defined"):
        Annotation(Finding.PERIAPICAL_LESION, "r1", stage=CariesStage.D2)


def test_annotation_round_trip() -> None:
    a = Annotation(
        Finding.CARIES, "r1", bbox=BBox(1, 2, 3, 4), tooth_fdi=36,
        stage=CariesStage.E1, confidence=0.4, meta={"surface": "mesial"},
    )
    b = Annotation.from_dict(json.loads(json.dumps(a.to_dict())))
    assert b == a


# --- RadiographRecord ---------------------------------------------------------


def test_record_requires_readers() -> None:
    with pytest.raises(ValueError, match="readers is empty"):
        record("i1", "p1", readers=frozenset())


@pytest.mark.parametrize("field_name", ["image_id", "path", "patient_id", "dataset"])
def test_record_requires_identifiers(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        record(**{"image_id": "i1", "patient_id": "p1", field_name: ""})


def test_annotation_from_non_reader_rejected() -> None:
    with pytest.raises(ValueError, match="not in readers"):
        record("i1", "p1", readers={"r1"}, annotations=[Annotation(Finding.CARIES, "r2")])


def test_reader_with_no_findings_is_a_negative_not_a_missing_read() -> None:
    r = record(
        "i1", "p1", readers={"r1", "r2"}, annotations=[Annotation(Finding.CARIES, "r1")]
    )
    assert r.findings_by("r2") == []
    assert r.has(Finding.CARIES, "r1") and not r.has(Finding.CARIES, "r2")
    with pytest.raises(KeyError, match="did not read"):
        r.findings_by("r3")


def test_has_refuses_to_pool_readers() -> None:
    r = record("i1", "p1", readers={"r1", "r2"})
    assert r.is_multi_annotator
    with pytest.raises(ValueError, match="pass annotator_id"):
        r.has(Finding.CARIES)


def test_has_single_reader_needs_no_annotator() -> None:
    r = record("i1", "p1", annotations=[Annotation(Finding.CARIES, CONSENSUS)])
    assert not r.is_multi_annotator
    assert r.has(Finding.CARIES)
    assert not r.has(Finding.DEEP_CARIES)


@pytest.mark.parametrize("age", [-1.0, 121.0, math.nan])
def test_record_rejects_implausible_age(age: float) -> None:
    with pytest.raises(ValueError, match="age"):
        record("i1", "p1", age=age)


@pytest.mark.parametrize("dim", ["width", "height"])
def test_record_rejects_non_positive_dimensions(dim: str) -> None:
    with pytest.raises(ValueError, match=dim):
        record("i1", "p1", **{dim: 0})


def test_record_coerces_enums() -> None:
    r = record(
        "i1", "p1", modality="panoramic", patient_id_source="assumed_unique", sex="female",
        readers=["a", "b"],
    )
    assert r.modality is Modality.PANORAMIC
    assert r.patient_id_source is PatientIdSource.ASSUMED_UNIQUE
    assert r.sex is Sex.FEMALE
    assert r.readers == frozenset({"a", "b"})


def test_teeth_present_validation() -> None:
    r = record("i1", "p1", teeth_present=[11, 36],
               annotations=[Annotation(Finding.CARIES, CONSENSUS, tooth_fdi=36)])
    assert r.teeth_present == frozenset({11, 36})
    assert r.teeth_present_source is InventorySource.ANNOTATED  # conftest default
    with pytest.raises(ValueError, match="set together"):
        record("i1", "p1", teeth_present=[11], teeth_present_source=None)
    with pytest.raises(ValueError, match="set together"):
        record("i1", "p1", teeth_present_source="predicted")
    with pytest.raises(ValueError, match="invalid FDI"):
        record("i1", "p1", teeth_present=[11, 19])
    with pytest.raises(ValueError, match="not in teeth_present"):
        record("i1", "p1", teeth_present=[11],
               annotations=[Annotation(Finding.CARIES, CONSENSUS, tooth_fdi=36)])


def test_group_key_is_namespaced_by_dataset() -> None:
    assert record("i1", "17", dataset="dentex").group_key != record(
        "i2", "17", dataset="tufts"
    ).group_key


def test_record_round_trip_is_json_safe() -> None:
    recs = make_records(n_patients=5, readers=("r1", "r2"), caries_rate=0.9, seed=3)
    recs.append(record("x", "y", teeth_present={11, 21}))
    for r in recs:
        restored = RadiographRecord.from_dict(json.loads(json.dumps(r.to_dict())))
        assert restored == r


# --- summarise ----------------------------------------------------------------


def test_summarise_counts() -> None:
    recs = [
        record(
            "a1", "p1", site_id="A", readers={"r1", "r2"},
            annotations=[
                Annotation(Finding.CARIES, "r1", stage=CariesStage.E1),
                Annotation(Finding.CARIES, "r2"),
                Annotation(Finding.PERIAPICAL_LESION, "r2"),
            ],
        ),
        record("a2", "p1", site_id="A"),
        record("b1", "p1", dataset="other", site_id="B",
               patient_id_source=PatientIdSource.ASSUMED_UNIQUE),
    ]
    s = summarise(recs)
    assert s.n_images == 3
    assert s.n_patients == 2  # same patient_id, different datasets
    assert s.images_per_patient == 1.5
    assert s.max_images_per_patient == 2
    assert s.n_annotations == 3
    assert s.sites == {"A": 2, "B": 1}
    assert s.findings == {"caries": 2, "periapical_lesion": 1}
    assert s.caries_stages == {"E1": 1, "UNSTAGED": 1}
    assert s.multi_reader_images == 1
    assert s.patient_id_sources == {"assumed_unique": 1, "provided": 2}
    assert s.inventory_sources == {"none": 3}


def test_summarise_empty() -> None:
    s = summarise([])
    assert s.n_images == 0 and s.images_per_patient == 0.0 and s.max_images_per_patient == 0


def test_iter_annotations() -> None:
    recs = make_records(n_patients=10, readers=("r1", "r2"), caries_rate=0.5, seed=4)
    pairs = list(iter_annotations(recs))
    assert len(pairs) == sum(len(r.annotations) for r in recs)
    assert all(a in r.annotations for r, a in pairs)
