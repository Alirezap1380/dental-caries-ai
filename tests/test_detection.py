from __future__ import annotations

import numpy as np
import pytest
from conftest import record

from dcai.data.schema import Annotation, BBox, Finding, RadiographRecord
from dcai.eval.detection import (
    Detection,
    _weighted_ap,
    average_precision,
    detection_report,
    froc_curve,
    iou_matrix,
    match_lesions,
    stratified_average_precision,
)
from dcai.eval.scales import DENTEX_DEPTH

N_BOOT = 200
A = BBox(0, 0, 10, 10)
B = BBox(100, 100, 110, 110)
BG = BBox(300, 300, 310, 310)  # background: no lesion here


def lesion(box: BBox, finding: Finding = Finding.CARIES, reader: str = "r1") -> Annotation:
    return Annotation(finding, reader, bbox=box)


def image(i: int, *lesions: Annotation, readers: set[str] | None = None) -> RadiographRecord:
    return record(f"i{i}", f"p{i}", readers=readers or {"r1"}, annotations=list(lesions))


def det(i: int, box: BBox, score: float, finding: Finding = Finding.CARIES) -> Detection:
    return Detection(f"i{i}", box, score, finding)


def match(records: list, dets: list, **kw):
    return match_lesions(records, dets, reader="r1", scale=DENTEX_DEPTH, **kw)


# --- geometry ---------------------------------------------------------------------


def test_iou_matrix_matches_bbox_iou() -> None:
    boxes = [A, B, BBox(5, 0, 15, 10), BBox(0, 0, 20, 20)]
    arr = np.array([b.as_array() for b in boxes])
    m = iou_matrix(arr, arr)
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            assert m[i, j] == pytest.approx(a.iou(b))


def test_detection_score_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        Detection("i0", A, float("nan"))


# --- matching ---------------------------------------------------------------------


def test_hit_miss_and_false_positive() -> None:
    m = match([image(0, lesion(A), lesion(B))], [det(0, A, 0.9), det(0, BG, 0.4)])
    assert m.gt_score.tolist() == [0.9, -np.inf]
    assert m.det_level.tolist() == [1, 0]


def test_low_overlap_is_not_a_hit() -> None:
    shifted = BBox(6, 0, 16, 10)  # IoU 4/16 = 0.25
    m = match([image(0, lesion(A))], [det(0, shifted, 0.9)])
    assert m.gt_score.tolist() == [-np.inf]
    m_loose = match([image(0, lesion(A))], [det(0, shifted, 0.9)], iou_threshold=0.2)
    assert m_loose.gt_score.tolist() == [0.9]


def test_duplicate_detection_is_a_false_positive() -> None:
    m = match([image(0, lesion(A))], [det(0, A, 0.6), det(0, A, 0.9)])
    assert m.gt_score.tolist() == [0.9]  # higher score claims it
    assert sorted(m.det_level.tolist()) == [0, 1]


def test_detection_claims_its_best_overlap() -> None:
    near_a = BBox(0, 0, 10, 11)
    m = match([image(0, lesion(BBox(0, 0, 10, 14)), lesion(A))], [det(0, near_a, 0.9)])
    assert m.gt_score.tolist() == [-np.inf, 0.9]


def test_detection_is_class_agnostic_within_target() -> None:
    m = match([image(0, lesion(A, Finding.DEEP_CARIES))], [det(0, A, 0.9, Finding.CARIES)])
    assert m.gt_score.tolist() == [0.9] and m.gt_level.tolist() == [2]


def test_non_target_findings_are_dropped_on_both_sides() -> None:
    rec = image(0, lesion(A), lesion(B, Finding.PERIAPICAL_LESION))
    m = match([rec], [det(0, B, 0.9, Finding.PERIAPICAL_LESION)])
    assert m.gt_level.tolist() == [1] and m.det_score.size == 0


def test_images_without_detections_still_count() -> None:
    m = match([image(0, lesion(A)), image(1, lesion(A)), image(2)], [det(0, BG, 0.9)])
    assert m.n_images == 3
    r = detection_report(m, threshold=0.5, seed=0, n_boot=N_BOOT)
    assert r.fp_per_image.value == pytest.approx(1 / 3)


def test_matching_is_against_one_reader() -> None:
    rec = image(0, lesion(A, reader="r1"), lesion(B, reader="r2"), readers={"r1", "r2"})
    m1 = match_lesions([rec], [], reader="r1", scale=DENTEX_DEPTH)
    m2 = match_lesions([rec], [], reader="r2", scale=DENTEX_DEPTH)
    assert m1.gt_level.size == m2.gt_level.size == 1
    with pytest.raises(KeyError, match="did not read"):
        match_lesions([rec], [], reader="r3", scale=DENTEX_DEPTH)


@pytest.mark.parametrize(
    "records, dets, kw, match_text",
    [
        ([image(0)], [det(9, A, 0.5)], {}, "not in records"),
        ([image(0), image(0)], [], {}, "duplicate"),
        ([image(0, Annotation(Finding.CARIES, "r1"))], [], {}, "without a box"),
        ([image(0)], [], {"iou_threshold": 0.0}, "iou_threshold"),
    ],
)
def test_match_validation(records: list, dets: list, kw: dict, match_text: str) -> None:
    with pytest.raises(ValueError, match=match_text):
        match(records, dets, **kw)


# --- detection_report / froc --------------------------------------------------------


def two_depth_set(n: int = 60) -> tuple[list, list]:
    """Each image has one shallow and one deep lesion; deep ones are found more often."""
    rng = np.random.default_rng(0)
    recs, dets = [], []
    for i in range(n):
        recs.append(image(i, lesion(A), lesion(B, Finding.DEEP_CARIES)))
        if rng.random() < 0.3:
            dets.append(det(i, A, rng.uniform(0.5, 1)))
        if rng.random() < 0.9:
            dets.append(det(i, B, rng.uniform(0.5, 1)))
        if rng.random() < 0.5:
            dets.append(det(i, BG, rng.uniform(0, 1)))
    return recs, dets


def test_detection_report_is_stratified() -> None:
    recs, dets = two_depth_set()
    m = match(recs, dets)
    r = detection_report(m, threshold=0.5, seed=0, n_boot=N_BOOT)
    assert r.n_images == 60 and r.n_lesions == 120 and r.n_detections == len(dets)
    shallow = r.sensitivity.stratum("caries").sensitivity
    deep = r.sensitivity.stratum("deep_caries").sensitivity
    assert shallow.value == pytest.approx((m.gt_score[m.gt_level == 1] >= 0.5).mean())
    assert deep.value > shallow.value and r.sensitivity.depth_gap.lo > 0
    assert r.sensitivity.specificity is None  # lesions have no sound units
    expected_fp = sum(1 for d in dets if d.bbox == BG and d.score >= 0.5) / 60
    assert r.fp_per_image.value == pytest.approx(expected_fp)


def test_froc_is_monotone_and_ordered_by_depth() -> None:
    m = match(*two_depth_set())
    f = froc_curve(m)
    for sens in f.sensitivity.values():
        assert np.all(np.diff(sens) >= 0)
    assert np.all(f.sensitivity["deep_caries"][-1] >= f.sensitivity["caries"][-1])


def test_froc_empty_stratum_is_nan() -> None:
    m = match([image(0, lesion(A)), image(1)], [det(0, A, 0.9)])
    f = froc_curve(m, fp_rates=[1.0])
    assert f.sensitivity["caries"].tolist() == [1.0]
    assert np.isnan(f.sensitivity["deep_caries"]).all()


# --- average precision ----------------------------------------------------------------


def test_ap_hand_computed() -> None:
    # TP, FP, TP with three lesions: precision 1, 1/2, 2/3 at recall 1/3, 1/3, 2/3.
    # Envelope 1, 2/3, 2/3 -> AP = 1/3 * 1 + 1/3 * 2/3.
    ap = average_precision(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1]), n_pos=3)
    assert ap == pytest.approx(1 / 3 + 2 / 9)


def test_ap_edge_cases() -> None:
    assert np.isnan(average_precision(np.array([0.5]), np.array([1]), n_pos=0))
    assert average_precision(np.array([]), np.array([]), n_pos=2) == 0.0
    assert _weighted_ap(np.array([0.5]), np.array([True]), np.array([0.0]), 1.0) == 0.0


def test_weighted_ap_equals_materialised_duplicates() -> None:
    rng = np.random.default_rng(1)
    s, tp = rng.random(20), rng.random(20) < 0.5
    w = rng.integers(0, 3, 20).astype(float)
    dup_s, dup_tp = np.repeat(s, w.astype(int)), np.repeat(tp, w.astype(int))
    assert _weighted_ap(s, tp, w, 7.0) == pytest.approx(average_precision(dup_s, dup_tp, 7))


def test_stratified_ap_ignores_other_strata() -> None:
    # Background FP (0.95), then the deep lesion (0.9), then the shallow one (0.8).
    recs = [image(0, lesion(A), lesion(B, Finding.DEEP_CARIES)), image(1)]
    dets = [det(0, BG, 0.95), det(0, B, 0.9), det(0, A, 0.8)]
    r = stratified_average_precision(match(recs, dets), seed=0, n_boot=N_BOOT)
    shallow, deep = r.strata
    # Each stratum pays for the background FP, but not for hits on the other stratum.
    assert shallow.ap.value == pytest.approx(0.5)  # without ignore it would be 1/3
    assert deep.ap.value == pytest.approx(0.5)
    assert r.pooled.value == pytest.approx(2 / 3)
    assert shallow.n_lesions == deep.n_lesions == 1


def test_stratified_ap_prefers_real_lesion_over_ignored() -> None:
    # One detection overlapping both a deep lesion (better IoU) and a shallow one.
    big = BBox(0, 0, 10, 12)
    recs = [image(0, lesion(BBox(0, 0, 10, 14)), lesion(A, Finding.DEEP_CARIES)), image(1)]
    r = stratified_average_precision(match(recs, [det(0, big, 0.9)]), seed=0, n_boot=N_BOOT)
    assert r.strata[0].ap.value == pytest.approx(1.0)  # shallow stratum still gets its hit


def test_stratified_ap_empty_stratum() -> None:
    recs = [image(0, lesion(A)), image(1, lesion(B))]
    r = stratified_average_precision(match(recs, [det(0, A, 0.9)]), seed=0, n_boot=N_BOOT)
    assert r.strata[1].n_lesions == 0 and r.strata[1].ap is None
    assert r.strata[0].ap.value == pytest.approx(0.5)


def test_stratified_ap_ci_on_realistic_set() -> None:
    r = stratified_average_precision(match(*two_depth_set()), seed=0, n_boot=N_BOOT)
    shallow, deep = r.strata
    assert shallow.ap.hi < deep.ap.lo
    assert shallow.ap.value < r.pooled.value < deep.ap.value


def test_stratified_ap_with_detections_on_lesion_free_image() -> None:
    # Regression, found by the end-to-end run: an image with boxes but no
    # lesions at all crashed the ignore lookup.
    recs = [image(0, lesion(A)), image(1, lesion(B)), image(2)]
    dets = [det(0, A, 0.9), det(2, BG, 0.95)]
    r = stratified_average_precision(match(recs, dets), seed=0, n_boot=N_BOOT)
    assert r.strata[0].ap.value == pytest.approx(0.5 * 0.5)  # FP first, then 1 of 2 found
