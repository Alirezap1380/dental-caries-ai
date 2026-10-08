"""Ordinal severity scales that lesions are graded and stratified on.

Index 0 is always "sound". Agreement (`agreement.py`) grades each reader's read
of an image on a scale; stratified metrics (`stratified.py`) report one result
per non-zero level. Neither ever collapses the scale to a single number.

Two scales:

- DENTEX_DEPTH: sound < caries < deep_caries. DENTEX's only depth information.
  Its "caries" level pools enamel and shallow-dentine lesions, so its sensitivity
  sits above that of true E1 lesions, assuming sensitivity rises with depth.
  The shallow-vs-deep gap measured on this scale is therefore a *lower bound*
  on the early-lesion gap, and should be written up as one.
- ICCMS_STAGE: sound < E1 < E2 < D1 < D2 < D3, for staged bitewing sets
  (the ACTA-Bw25 track). Unstaged lesions cannot be placed on it and raise.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from dcai.data.schema import CARIES_FINDINGS, Annotation, CariesStage, Finding


@dataclass(frozen=True)
class OrdinalScale:
    name: str
    categories: tuple[str, ...]
    # Grade of one annotation, or None if the annotation is not on this scale
    # (e.g. an implant is neither sound nor carious).
    grade: Callable[[Annotation], int | None]

    def __post_init__(self) -> None:
        if len(self.categories) < 2 or self.categories[0] != "sound":
            raise ValueError("a scale needs >= 2 categories, starting with 'sound'")

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    @property
    def lesion_levels(self) -> range:
        return range(1, self.n_categories)

    def image_grade(self, annotations: Iterable[Annotation]) -> int:
        """Most severe grade a reader assigned anywhere in the image (0 if none)."""
        grades = [g for a in annotations if (g := self.grade(a)) is not None]
        return max(grades, default=0)


def _depth_grade(a: Annotation) -> int | None:
    if a.finding is Finding.CARIES:
        return 1
    if a.finding is Finding.DEEP_CARIES:
        return 2
    return 0 if a.finding is Finding.HEALTHY else None


def _iccms_grade(a: Annotation) -> int | None:
    if a.finding is Finding.HEALTHY:
        return 0
    if a.finding not in CARIES_FINDINGS:
        return None
    if a.stage is None:
        raise ValueError(
            f"unstaged {a.finding.value} cannot be graded on ICCMS_STAGE; use DENTEX_DEPTH"
        )
    return int(a.stage)


DENTEX_DEPTH = OrdinalScale("dentex_depth", ("sound", "caries", "deep_caries"), _depth_grade)
ICCMS_STAGE = OrdinalScale(
    "iccms_stage", ("sound", *(s.name for s in CariesStage if s is not CariesStage.NONE)),
    _iccms_grade,
)
