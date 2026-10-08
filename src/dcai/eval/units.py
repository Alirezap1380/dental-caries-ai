"""The tooth table: one row per tooth, with the reference grade and the model's probabilities.

This is the shared evaluation unit for calibration, stratified classification
metrics, abstention and subgroups, the same unit `agreement.py` uses.

Calibration in particular has to live here and not on detector boxes. ECE assumes
a fixed set of cases per confidence bin, but a detector chooses how many boxes to
emit: change the NMS threshold and the denominator changes. Calibrating raw box
scores is therefore ill-posed. A per-tooth probability answers a fixed question
for a fixed set of units, "does tooth 36 have a lesion?", so it is well-posed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dcai.data.schema import VALID_FDI, RadiographRecord
from dcai.eval.ratings import (
    MAJORITY,
    check_inventory,
    consensus_grades,
    inventory,
    tooth_grades,
    tooth_level_ratings,
    unit_id,
)
from dcai.eval.scales import OrdinalScale


@dataclass(frozen=True)
class ToothPrediction:
    image_id: str
    tooth_fdi: int
    probs: tuple[float, ...]  # distribution over the scale's categories, sound first

    def __post_init__(self) -> None:
        if self.tooth_fdi not in VALID_FDI:
            raise ValueError(f"invalid FDI {self.tooth_fdi}")
        p = np.asarray(self.probs, dtype=float)
        if p.ndim != 1 or p.size < 2 or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError(f"probs must be finite and non-negative, got {self.probs}")
        if not np.isclose(p.sum(), 1.0, atol=1e-6):
            raise ValueError(f"probs must sum to 1, got {p.sum()}")


@dataclass(frozen=True, eq=False)
class ToothTable:
    scale: OrdinalScale
    reference_name: str
    unit_ids: tuple[str, ...]
    image_ids: np.ndarray
    groups: np.ndarray  # patient group_key per tooth
    sites: np.ndarray  # site_id per tooth (object; None if unknown)
    ages: np.ndarray  # float, NaN if unknown
    sexes: np.ndarray  # object; None if unknown
    reference: np.ndarray  # reference-standard grade per tooth
    probs: np.ndarray  # (n_teeth, n_categories)

    @property
    def n(self) -> int:
        return len(self.unit_ids)

    @property
    def p_lesion(self) -> np.ndarray:
        return 1.0 - self.probs[:, 0]

    @property
    def has_lesion(self) -> np.ndarray:
        return (self.reference > 0).astype(int)

    def predicted_grade(self, threshold: float) -> np.ndarray:
        """The model's call as deployed: sound below the operating threshold,
        otherwise its most probable lesion grade.

        Not a plain argmax over all categories: the deployed decision is made at
        the operating point, and agreement must judge the model it would ship.
        """
        lesion_grade = 1 + self.probs[:, 1:].argmax(axis=1)
        return np.where(self.p_lesion >= threshold, lesion_grade, 0)

    def model_grades(self, threshold: float) -> dict[str, int]:
        return dict(zip(self.unit_ids, self.predicted_grade(threshold).tolist(), strict=True))


def tooth_table(
    records: Sequence[RadiographRecord],
    predictions: Sequence[ToothPrediction],
    *,
    scale: OrdinalScale,
    reference: str = MAJORITY,
) -> ToothTable:
    """Align predictions with a reference standard on every tooth present.

    `reference` is one reader's id, or MAJORITY for the strict-majority consensus
    of each image's readers. Predictions must cover exactly the inventory:
    a missing tooth is a pipeline bug, and so is a prediction for an absent tooth.
    """
    check_inventory(records)
    preds: dict[str, ToothPrediction] = {}
    for p in predictions:
        key = unit_id(p.image_id, p.tooth_fdi)
        if key in preds:
            raise ValueError(f"duplicate prediction for {key}")
        if len(p.probs) != scale.n_categories:
            raise ValueError(f"{key}: {len(p.probs)} probs for a {scale.n_categories}-level scale")
        preds[key] = p

    if reference == MAJORITY:
        m = tooth_level_ratings(records, scale)
        ref_by_unit = dict(zip(m.item_ids, consensus_grades(m.ratings, min_raters=1).tolist(),
                               strict=True))
    else:
        ref_by_unit = {
            unit_id(r.image_id, fdi): g
            for r in records for fdi, g in tooth_grades(r, reference, scale).items()
        }

    units, rows = [], []
    for rec in records:
        for fdi in inventory(rec):
            units.append(unit_id(rec.image_id, fdi))
            rows.append(rec)
    missing = [u for u in units if u not in preds]
    if missing:
        raise ValueError(f"no prediction for {len(missing)} tooth/teeth, e.g. {missing[:3]}")
    extra = preds.keys() - set(units)
    if extra:
        raise ValueError(f"predictions for teeth not in the inventory: {sorted(extra)[:3]}")
    ref = np.array([ref_by_unit[u] for u in units], dtype=np.int64)

    return ToothTable(
        scale=scale,
        reference_name=reference,
        unit_ids=tuple(units),
        image_ids=np.array([r.image_id for r in rows], dtype=object),
        groups=np.array([r.group_key for r in rows], dtype=object),
        sites=np.array([r.site_id for r in rows], dtype=object),
        ages=np.array([np.nan if r.age is None else r.age for r in rows], dtype=float),
        sexes=np.array([None if r.sex is None else r.sex.value for r in rows], dtype=object),
        reference=ref,
        probs=np.array([preds[u].probs for u in units], dtype=float).reshape(len(units), -1),
    )
