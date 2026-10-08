"""Lesion-level detection metrics, stratified by depth.

Design choices:

- **One reader at a time.** Lesions come from `record.findings_by(reader)`, so a
  model is scored against each annotator separately (rule 3). Asking for a reader
  who did not read an image raises; it is never treated as "no lesions".
- **Detection is class-agnostic within the target.** A model that boxes a deep
  lesion but labels it "caries" has still *detected* it. Depth is the axis we
  stratify on, taken from the reader's grade; whether the model also gets the
  depth right is a separate staging question, not a detection miss.
- **Greedy COCO-style matching.** Detections, highest score first, claim the
  unclaimed lesion they overlap most (IoU >= threshold). Because of the score
  order, the matching restricted to detections above any threshold t is exactly
  the matching those detections would get on their own. One matching therefore
  serves every operating point and the FROC curve.
- **Per-stratum AP uses COCO's ignore mechanism**, the same one behind
  AP_small/medium/large: lesions from other strata are ignored, and a detection
  that lands on one counts as neither TP nor FP. False positives on background
  have no stage and count against every stratum, because every stratum's AP has
  to pay for them.
- **Images with no detections still count.** They are in the denominator of
  false positives per image, and their lesions are misses.

`pooled` numbers appear only beside the strata, as in `stratified.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from dcai.data.schema import CARIES_FINDINGS, BBox, Finding, RadiographRecord
from dcai.eval.bootstrap import Estimate, grouped_bootstrap
from dcai.eval.scales import OrdinalScale
from dcai.eval.stratified import StratifiedReport, stratified_report


@dataclass(frozen=True)
class Detection:
    image_id: str
    bbox: BBox
    score: float
    finding: Finding = Finding.CARIES

    def __post_init__(self) -> None:
        if not np.isfinite(self.score):
            raise ValueError(f"detection score must be finite, got {self.score}")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (n, 4) and (m, 4) xyxy boxes."""
    a, b = a.reshape(-1, 4), b.reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def _greedy_match(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    gt_boxes: np.ndarray,
    ignore: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Index of the lesion each detection claims, or -1."""
    assign = np.full(len(det_scores), -1)
    if len(det_scores) == 0 or len(gt_boxes) == 0:
        return assign
    iou = iou_matrix(det_boxes, gt_boxes)
    claimed = np.zeros(len(gt_boxes), dtype=bool)
    for d in np.argsort(-det_scores, kind="stable"):
        candidates = ~claimed & (iou[d] >= iou_threshold)
        # Prefer a real lesion over an ignored one, as COCO does.
        real = candidates & ~ignore
        pool = real if real.any() else candidates
        if pool.any():
            g = int(np.argmax(np.where(pool, iou[d], -1.0)))
            claimed[g] = True
            assign[d] = g
    return assign


@dataclass(frozen=True)
class _Image:
    det_boxes: np.ndarray
    det_scores: np.ndarray
    gt_boxes: np.ndarray
    gt_levels: np.ndarray


@dataclass(frozen=True, eq=False)
class LesionMatches:
    reader: str
    scale: OrdinalScale
    iou_threshold: float
    image_ids: tuple[str, ...]
    image_groups: np.ndarray  # patient group per image
    gt_image: np.ndarray  # image index per lesion
    gt_level: np.ndarray  # reader's grade per lesion (>= 1)
    gt_score: np.ndarray  # score of the claiming detection; -inf if never claimed
    det_image: np.ndarray  # image index per detection
    det_score: np.ndarray
    det_level: np.ndarray  # level of the claimed lesion; 0 = false positive
    _images: tuple[_Image, ...] = field(repr=False)

    @property
    def n_images(self) -> int:
        return len(self.image_ids)


def match_lesions(
    records: Sequence[RadiographRecord],
    detections: Sequence[Detection],
    *,
    reader: str,
    scale: OrdinalScale,
    iou_threshold: float = 0.5,
    target: frozenset[Finding] = CARIES_FINDINGS,
) -> LesionMatches:
    """Match detections to one reader's lesions. Non-target detections are dropped."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    index = {r.image_id: i for i, r in enumerate(records)}
    if len(index) != len(records):
        raise ValueError("duplicate image_id in records")
    unknown = {d.image_id for d in detections} - index.keys()
    if unknown:
        raise ValueError(f"detections for images not in records: {sorted(unknown)[:3]}")

    dets_by_image: list[list[Detection]] = [[] for _ in records]
    for d in detections:
        if d.finding in target:
            dets_by_image[index[d.image_id]].append(d)

    images: list[_Image] = []
    gt_image, gt_level, gt_score = [], [], []
    det_image, det_score, det_level = [], [], []
    for i, rec in enumerate(records):
        lesions = [a for a in rec.findings_by(reader) if a.finding in target]
        if any(a.bbox is None for a in lesions):
            raise ValueError(f"{rec.image_id}: reader {reader!r} has a lesion without a box")
        gt_boxes = np.array([a.bbox.as_array() for a in lesions]).reshape(-1, 4)
        levels = np.array([scale.grade(a) for a in lesions], dtype=np.int64)
        dets = dets_by_image[i]
        boxes = np.array([d.bbox.as_array() for d in dets]).reshape(-1, 4)
        scores = np.array([d.score for d in dets], dtype=float)

        assign = _greedy_match(boxes, scores, gt_boxes, np.zeros(len(lesions), bool),
                               iou_threshold)
        claimed_by = np.full(len(lesions), -np.inf)
        claimed_by[assign[assign >= 0]] = scores[assign >= 0]

        images.append(_Image(boxes, scores, gt_boxes, levels))
        gt_image += [i] * len(lesions)
        gt_level += levels.tolist()
        gt_score += claimed_by.tolist()
        det_image += [i] * len(dets)
        det_score += scores.tolist()
        det_level += [int(levels[g]) if g >= 0 else 0 for g in assign]

    return LesionMatches(
        reader=reader,
        scale=scale,
        iou_threshold=iou_threshold,
        image_ids=tuple(r.image_id for r in records),
        image_groups=np.array([r.group_key for r in records]),
        gt_image=np.array(gt_image, dtype=np.int64),
        gt_level=np.array(gt_level, dtype=np.int64),
        gt_score=np.array(gt_score, dtype=float),
        det_image=np.array(det_image, dtype=np.int64),
        det_score=np.array(det_score, dtype=float),
        det_level=np.array(det_level, dtype=np.int64),
        _images=tuple(images),
    )


@dataclass(frozen=True)
class DetectionReport:
    reader: str
    threshold: float
    iou_threshold: float
    n_images: int
    n_lesions: int
    n_detections: int
    sensitivity: StratifiedReport  # lesion-level, one estimate per depth stratum
    fp_per_image: Estimate  # background false positives: not stage-attributable


def detection_report(
    m: LesionMatches, *, threshold: float, seed: int, n_boot: int = 2000
) -> DetectionReport:
    fp_counts = np.bincount(
        m.det_image[(m.det_level == 0) & (m.det_score >= threshold)], minlength=m.n_images
    ).astype(float)
    return DetectionReport(
        reader=m.reader,
        threshold=threshold,
        iou_threshold=m.iou_threshold,
        n_images=m.n_images,
        n_lesions=m.gt_level.size,
        n_detections=m.det_score.size,
        sensitivity=stratified_report(
            m.gt_level, m.gt_score, m.image_groups[m.gt_image],
            scale=m.scale, threshold=threshold, seed=seed, n_boot=n_boot,
        ),
        fp_per_image=grouped_bootstrap(
            lambda idx: float(fp_counts[idx].mean()), m.image_groups, seed=seed, n_boot=n_boot
        ),
    )


@dataclass(frozen=True)
class FrocCurve:
    fp_per_image: np.ndarray
    sensitivity: dict[str, np.ndarray]  # per stratum name, aligned with fp_per_image


def froc_curve(
    m: LesionMatches, fp_rates: Sequence[float] = (0.125, 0.25, 0.5, 1, 2, 4, 8)
) -> FrocCurve:
    """Per-stratum sensitivity at fixed false-positive rates. For figures only.

    These are point estimates on purpose. Numbers quoted in the write-up come
    from `detection_report` at a threshold fixed in advance, which has CIs.
    """
    thresholds = np.r_[np.inf, np.unique(m.det_score)[::-1]]
    fp = np.array([((m.det_level == 0) & (m.det_score >= t)).sum() for t in thresholds])
    fp = fp / m.n_images
    out: dict[str, np.ndarray] = {}
    for level in m.scale.lesion_levels:
        mask = m.gt_level == level
        sens = np.array([
            (m.gt_score[mask] >= t).mean() if mask.any() else np.nan for t in thresholds
        ])
        # Lowest threshold whose FP rate is within budget: fp is non-decreasing.
        idx = np.searchsorted(fp, fp_rates, side="right") - 1
        out[m.scale.categories[level]] = sens[idx]
    return FrocCurve(np.asarray(fp_rates, dtype=float), out)


def _weighted_ap(scores: np.ndarray, tp: np.ndarray, weights: np.ndarray, n_pos: float) -> float:
    """All-point interpolated AP. Weights let a bootstrap repeat images without copying."""
    if n_pos <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    w = weights[order]
    tp_cum = np.cumsum(w * tp[order])
    fp_cum = np.cumsum(w * ~tp[order])
    keep = w > 0
    if not keep.any():
        return 0.0
    tp_cum, fp_cum = tp_cum[keep], fp_cum[keep]
    recall = tp_cum / n_pos
    precision = tp_cum / (tp_cum + fp_cum)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float((np.diff(np.r_[0.0, recall]) * envelope).sum())


def average_precision(scores: np.ndarray, tp: np.ndarray, n_pos: int) -> float:
    scores = np.asarray(scores, dtype=float)
    return _weighted_ap(scores, np.asarray(tp, bool), np.ones_like(scores), n_pos)


@dataclass(frozen=True)
class StratumAP:
    level: int
    name: str
    n_lesions: int
    ap: Estimate | None  # None for an empty stratum


@dataclass(frozen=True)
class APReport:
    reader: str
    iou_threshold: float
    strata: tuple[StratumAP, ...]
    pooled: Estimate  # beside the strata only: this is the number papers report


def stratified_average_precision(m: LesionMatches, *, seed: int, n_boot: int = 2000) -> APReport:
    def ap_estimate(det_img, det_scores, tp, counted, pos_per_image) -> Estimate:
        det_img, det_scores, tp = det_img[counted], det_scores[counted], tp[counted]

        def stat(idx: np.ndarray) -> float:
            w = np.bincount(idx, minlength=m.n_images).astype(float)
            return _weighted_ap(det_scores, tp, w[det_img], float(w @ pos_per_image))

        return grouped_bootstrap(stat, m.image_groups, seed=seed, n_boot=n_boot)

    strata = []
    for level in m.scale.lesion_levels:
        n = int((m.gt_level == level).sum())
        if n == 0:
            strata.append(StratumAP(level, m.scale.categories[level], 0, None))
            continue
        tp_parts, counted_parts = [], []
        for img in m._images:
            ignore = img.gt_levels != level
            assign = _greedy_match(img.det_boxes, img.det_scores, img.gt_boxes, ignore,
                                   m.iou_threshold)
            hit = assign >= 0
            hit_ignored = np.zeros(assign.size, dtype=bool)
            hit_ignored[hit] = ignore[assign[hit]]
            tp_parts.append((assign >= 0) & ~hit_ignored)
            counted_parts.append(~hit_ignored)
        pos = np.array([(img.gt_levels == level).sum() for img in m._images], dtype=float)
        strata.append(StratumAP(
            level, m.scale.categories[level], n,
            ap_estimate(m.det_image, m.det_score, np.concatenate(tp_parts),
                        np.concatenate(counted_parts), pos),
        ))

    pos_all = np.array([len(img.gt_levels) for img in m._images], dtype=float)
    pooled = ap_estimate(m.det_image, m.det_score, m.det_level > 0,
                         np.ones(m.det_score.size, bool), pos_all)
    return APReport(m.reader, m.iou_threshold, tuple(strata), pooled)
