"""Graded decisions on a shared unit: the input to agreement and to the tooth table.

**The unit is the tooth.** Kappa needs paired decisions on a shared unit.
Box-level agreement would first have to IoU-match one reader's boxes to
another's, which makes the agreement figure a function of an arbitrary matching
threshold. FDI numbering gives a canonical unit with no matching step: reader A's
grade for tooth 36 against reader B's grade for tooth 36. A reader's grade for a
tooth is the most severe lesion they placed on it, or 0 (sound).

The unit set is the image's tooth inventory (`RadiographRecord.teeth_present`),
not "teeth somebody flagged". Without an inventory, teeth every reader called
sound would disappear and a missing tooth would count as sound, so tooth-level
construction raises instead of guessing. Image-level ratings remain available as
a coarser fallback for datasets with no tooth numbering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from dcai.data.schema import RadiographRecord
from dcai.eval.scales import OrdinalScale

MISSING = -1  # rater did not read this unit: distinct from grading it 0 (sound)

# Reference-standard sentinel: the strict-majority consensus of every reader of
# the image (see `consensus_grades`). For a single-reader dataset it is that reader.
MAJORITY = "__majority__"


@dataclass(frozen=True, eq=False)
class RatingMatrix:
    item_ids: tuple[str, ...]
    groups: tuple[str, ...]  # patient group_key per item: the bootstrap unit
    raters: tuple[str, ...]
    ratings: np.ndarray  # (n_items, n_raters) int grades on `scale`, MISSING if unread
    scale: OrdinalScale

    def __post_init__(self) -> None:
        r = np.asarray(self.ratings)
        if r.shape != (len(self.item_ids), len(self.raters)):
            raise ValueError(f"ratings shape {r.shape} does not match items x raters")
        if len(self.groups) != len(self.item_ids):
            raise ValueError("groups must have one entry per item")
        if len(set(self.raters)) != len(self.raters):
            raise ValueError(f"duplicate rater names: {self.raters}")
        if r.size and (r.min() < MISSING or r.max() >= self.scale.n_categories):
            raise ValueError(f"ratings out of range for scale {self.scale.name}")
        object.__setattr__(self, "ratings", r.astype(np.int64))

    def column(self, rater: str) -> np.ndarray:
        return self.ratings[:, self.raters.index(rater)]

    def with_rater(self, name: str, grades: Mapping[str, int]) -> RatingMatrix:
        """Add a rater (typically the model) that must grade every item."""
        missing = [i for i in self.item_ids if i not in grades]
        if missing:
            raise ValueError(f"{name} has no grade for {len(missing)} item(s), e.g. {missing[:3]}")
        col = np.array([grades[i] for i in self.item_ids], dtype=np.int64)[:, None]
        return RatingMatrix(
            self.item_ids, self.groups, (*self.raters, name),
            np.hstack([self.ratings, col]), self.scale,
        )


def unit_id(image_id: str, tooth_fdi: int) -> str:
    return f"{image_id}#{tooth_fdi}"


def inventory(record: RadiographRecord) -> list[int]:
    if record.teeth_present is None:
        raise ValueError(
            f"{record.image_id} has no tooth inventory (teeth_present); "
            "tooth-level units cannot be formed"
        )
    return sorted(record.teeth_present)


def tooth_grades(record: RadiographRecord, reader: str, scale: OrdinalScale) -> dict[int, int]:
    """Reader's grade for every tooth in the inventory (0 where they marked nothing)."""
    grades = dict.fromkeys(inventory(record), 0)
    for a in record.findings_by(reader):
        g = scale.grade(a)
        if not g:
            continue
        if a.tooth_fdi is None:
            raise ValueError(
                f"{record.image_id}: {reader!r} has a {a.finding.value} with no tooth_fdi; "
                "it cannot be placed on a tooth"
            )
        grades[a.tooth_fdi] = max(grades[a.tooth_fdi], g)
    return grades


def check_inventory(records: Sequence[RadiographRecord], *, min_unflagged: float = 0.2) -> float:
    """Refuse an inventory that looks rebuilt from annotation boxes.

    In a real dentition most teeth carry no finding at all. If almost every
    inventoried tooth has a finding from some reader, `teeth_present` was very
    likely built from the diagnosis boxes, so sound teeth are missing. Specificity
    and kappa computed on it would be meaningless. Returns the unflagged fraction.
    """
    n_teeth = n_flagged = 0
    for rec in records:
        teeth = set(inventory(rec))
        n_teeth += len(teeth)
        n_flagged += len({a.tooth_fdi for a in rec.annotations if a.tooth_fdi in teeth})
    unflagged = 1.0 - n_flagged / n_teeth if n_teeth else 0.0
    if unflagged < min_unflagged:
        raise ValueError(
            f"only {unflagged:.0%} of inventoried teeth carry no finding: teeth_present looks "
            "derived from annotation boxes (sound teeth missing). Source it from enumeration "
            "labels or an enumeration model, not from findings."
        )
    return unflagged


def _all_readers(records: Sequence[RadiographRecord]) -> list[str]:
    return sorted(set().union(*(r.readers for r in records)))


def tooth_level_ratings(
    records: Sequence[RadiographRecord],
    scale: OrdinalScale,
    raters: Sequence[str] | None = None,
) -> RatingMatrix:
    """One row per tooth present, one column per reader; MISSING where unread."""
    check_inventory(records)
    raters = list(raters) if raters is not None else _all_readers(records)
    item_ids, groups, rows = [], [], []
    for rec in records:
        per_reader = {
            rater: tooth_grades(rec, rater, scale) for rater in raters if rater in rec.readers
        }
        for fdi in inventory(rec):
            item_ids.append(unit_id(rec.image_id, fdi))
            groups.append(rec.group_key)
            rows.append([per_reader[r][fdi] if r in per_reader else MISSING for r in raters])
    return RatingMatrix(
        tuple(item_ids), tuple(groups), tuple(raters),
        np.array(rows, dtype=np.int64).reshape(len(item_ids), len(raters)), scale,
    )


def image_level_ratings(
    records: Sequence[RadiographRecord],
    scale: OrdinalScale,
    raters: Sequence[str] | None = None,
) -> RatingMatrix:
    """Fallback unit for data without tooth numbering: most severe grade per image."""
    raters = list(raters) if raters is not None else _all_readers(records)
    ratings = np.array(
        [
            [
                scale.image_grade(rec.findings_by(rater)) if rater in rec.readers else MISSING
                for rater in raters
            ]
            for rec in records
        ],
        dtype=np.int64,
    ).reshape(len(records), len(raters))
    return RatingMatrix(
        tuple(r.image_id for r in records), tuple(r.group_key for r in records),
        tuple(raters), ratings, scale,
    )


def consensus_grades(ratings: np.ndarray, *, min_raters: int = 2) -> np.ndarray:
    """Strict-majority consensus per unit: the lower median of the available grades.

    The result is the most severe grade that a strict majority of readers assigned
    or exceeded. With an even number of readers who split, it takes the milder
    call: a lesion enters the reference standard only when most readers saw it.
    Units with fewer than `min_raters` available grades get MISSING.
    """
    ratings = np.asarray(ratings)
    out = np.full(ratings.shape[0], MISSING, dtype=np.int64)
    for i, row in enumerate(ratings):
        avail = np.sort(row[row != MISSING])
        if avail.size >= min_raters and avail.size > 0:
            out[i] = avail[(avail.size - 1) // 2]
    return out
