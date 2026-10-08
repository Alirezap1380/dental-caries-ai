"""Unified annotation schema.

Every dataset in this project (DENTEX, Tufts, bitewing sets) gets normalised into
these records before anything else touches it. The point is that the evaluation
harness never has to know which dataset a record came from -- except where it
deliberately does, via `site_id`, which is what the shortcut probes key on.

Four fields carry most of the methodological weight:

- `patient_id`:  splits are grouped on this, never on image. Bitewings come in
                 sets of four from one mouth; panoramics can have repeat visits.
                 Splitting on image leaks the patient and inflates every metric.
- `patient_id_source`: whether that id came from the dataset or was assumed. A
                 patient-level leakage check over ids that are really image ids
                 passes vacuously, and the write-up has to be able to say so.
- `readers`:     every annotator who *reviewed* the image, including those who
                 marked nothing. Without it, "reader B saw this image and found no
                 lesion" is indistinguishable from "reader B never saw it", and
                 inter-observer agreement silently drops exactly the negative
                 cases that agreement statistics depend on.
- `annotator_id`: kept per-annotation rather than collapsed to consensus, so
                 inter-observer agreement can be measured and used as a ceiling.

Validation is strict and happens at construction: a loader that maps labels
wrongly should fail on the first bad record, not produce a plausible metric later.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# Annotator id for datasets that only ship an adjudicated label. It is a normal
# reader id, not a special case -- but a loader has to choose it explicitly.
CONSENSUS = "consensus"


class Modality(str, Enum):
    PANORAMIC = "panoramic"
    BITEWING = "bitewing"
    PERIAPICAL = "periapical"
    INTRAORAL_PHOTO = "intraoral_photo"


class Finding(str, Enum):
    """Diagnosis classes, harmonised across datasets.

    DENTEX ships four: caries, deep caries, periapical lesion, impacted tooth.
    Bitewing sets usually ship caries graded by depth. We keep both and map
    dataset-specific labels onto this enum in each loader.
    """

    CARIES = "caries"
    DEEP_CARIES = "deep_caries"
    PERIAPICAL_LESION = "periapical_lesion"
    IMPACTED_TOOTH = "impacted_tooth"
    RESTORATION = "restoration"
    ROOT_CANAL_FILLING = "root_canal_filling"
    IMPLANT = "implant"
    CROWN = "crown"
    MISSING_TOOTH = "missing_tooth"
    HEALTHY = "healthy"


CARIES_FINDINGS = frozenset({Finding.CARIES, Finding.DEEP_CARIES})


class CariesStage(int, Enum):
    """Lesion depth, roughly aligned to ICCMS / radiographic scoring.

    This is the axis the headline result is stratified on. Models are reliably
    good at E2/D-stages and poor at E1 -- and E1 is the stage where a lesion can
    still be arrested without drilling, so it is where detection is worth most.
    Reporting a single pooled mAP hides exactly the failure that matters.

    An *unstaged* lesion is `Annotation.stage = None`, never `NONE`. `NONE` means
    "sound surface"; defaulting a DENTEX caries box (which carries no depth grade)
    to it would file real lesions under the healthy stratum.
    """

    NONE = 0  # sound surface
    E1 = 1  # outer half of enamel
    E2 = 2  # inner half of enamel, not past the EDJ
    D1 = 3  # outer third of dentine
    D2 = 4  # middle third of dentine
    D3 = 5  # inner third of dentine / pulpal involvement

    @property
    def is_initial(self) -> bool:
        """True for lesions that are candidates for non-operative management."""
        return self in (CariesStage.E1, CariesStage.E2)

    @property
    def is_enamel(self) -> bool:
        return self in (CariesStage.E1, CariesStage.E2)


class PatientIdSource(str, Enum):
    """Where a record's `patient_id` came from."""

    PROVIDED = "provided"  # the dataset ships a patient identifier
    DERIVED = "derived"  # reconstructed by us, e.g. from filenames or visit metadata
    # No identifier available: each image is treated as its own patient. Splits
    # still run, but the leakage guarantee is unverifiable for these records.
    ASSUMED_UNIQUE = "assumed_unique"


class InventorySource(str, Enum):
    """Where a record's `teeth_present` came from.

    Never reconstruct it from diagnosis boxes: that keeps only diseased teeth,
    sound teeth vanish from every denominator, and tooth-level specificity and
    kappa become meaningless. `dcai.eval.ratings.check_inventory` trips on it.
    """

    ANNOTATED = "annotated"  # human enumeration labels covering every visible tooth
    # An enumeration/segmentation model. Its errors then sit in every tooth-level
    # denominator, and the write-up has to say so.
    PREDICTED = "predicted"


class Sex(str, Enum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"


# FDI two-digit tooth notation, e.g. 11..18, 21..28, 31..38, 41..48.
VALID_FDI = frozenset(
    int(f"{quadrant}{tooth}") for quadrant in (1, 2, 3, 4) for tooth in range(1, 9)
)


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in absolute pixel coordinates, xyxy order."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Checked explicitly because NaN fails every comparison, so the
        # degenerate-box test below would let a NaN box straight through.
        if not all(math.isfinite(v) for v in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError(f"non-finite box coordinates: {self}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box: {self}")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / (self.area + other.area - inter)


@dataclass
class Annotation:
    """One finding, from one annotator.

    Multi-annotator datasets produce several Annotations per lesion. We do NOT
    merge them at load time -- `dcai.eval.agreement` needs them separate.

    `annotator_id` is required (no "consensus" default) so that a loader cannot
    collapse readers by forgetting to set it.
    """

    finding: Finding
    annotator_id: str
    bbox: BBox | None = None
    tooth_fdi: int | None = None
    stage: CariesStage | None = None
    confidence: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerced because Finding is a str enum: "caries" == Finding.CARIES, but
        # identity checks (`is`) against a raw string fail silently.
        self.finding = Finding(self.finding)
        if self.stage is not None:
            self.stage = CariesStage(self.stage)

        if not self.annotator_id:
            raise ValueError("annotator_id must be a non-empty string")
        if self.tooth_fdi is not None and self.tooth_fdi not in VALID_FDI:
            raise ValueError(
                f"tooth_fdi={self.tooth_fdi} is not valid FDI notation "
                "(expected 11-18, 21-28, 31-38, 41-48)"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        self._check_stage()

    def _check_stage(self) -> None:
        if self.stage is None:
            return
        if self.finding in CARIES_FINDINGS:
            if self.stage is CariesStage.NONE:
                raise ValueError(
                    f"{self.finding.value} annotated with stage NONE (sound surface); "
                    "use stage=None for an unstaged lesion"
                )
            # "Deep caries" is a dentine lesion by definition; an enamel stage here
            # means a loader has the label mapping backwards.
            if self.finding is Finding.DEEP_CARIES and self.stage.is_enamel:
                raise ValueError(f"deep_caries cannot have enamel stage {self.stage.name}")
        elif self.finding is Finding.HEALTHY:
            if self.stage is not CariesStage.NONE:
                raise ValueError(f"healthy finding cannot have lesion stage {self.stage.name}")
        else:
            raise ValueError(
                f"stage is only defined for caries/healthy findings, "
                f"got {self.finding.value} with stage {self.stage.name}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding": self.finding.value,
            "annotator_id": self.annotator_id,
            "bbox": None if self.bbox is None else self.bbox.as_array().tolist(),
            "tooth_fdi": self.tooth_fdi,
            "stage": None if self.stage is None else int(self.stage),
            "confidence": self.confidence,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Annotation:
        return cls(
            finding=Finding(d["finding"]),
            annotator_id=d["annotator_id"],
            bbox=None if d.get("bbox") is None else BBox(*d["bbox"]),
            tooth_fdi=d.get("tooth_fdi"),
            stage=None if d.get("stage") is None else CariesStage(d["stage"]),
            confidence=d.get("confidence"),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class RadiographRecord:
    """One image plus everything known about it."""

    image_id: str
    path: str
    patient_id: str
    patient_id_source: PatientIdSource
    dataset: str
    modality: Modality
    readers: frozenset[str]
    annotations: list[Annotation] = field(default_factory=list)

    # Acquisition provenance. `site_id` is the shortcut-probe target: if a model
    # can predict this from pixels, any apparent diagnostic skill is suspect
    # whenever site correlates with label prevalence.
    site_id: str | None = None
    device: str | None = None

    # Demographics, for subgroup reporting. Often absent; that absence is itself
    # worth stating in the write-up rather than quietly ignoring.
    age: float | None = None
    sex: Sex | None = None

    # FDI numbers of the teeth visible in the image (e.g. from DENTEX's enumeration
    # labels). Tooth-level metrics need it: a tooth no reader flagged is otherwise
    # invisible, and a missing tooth would be counted as a sound one. None means
    # no inventory, and tooth-level evaluation then raises rather than guessing.
    teeth_present: frozenset[int] | None = None
    teeth_present_source: InventorySource | None = None

    width: int | None = None
    height: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modality = Modality(self.modality)
        self.patient_id_source = PatientIdSource(self.patient_id_source)
        self.readers = frozenset(self.readers)
        if self.sex is not None:
            self.sex = Sex(self.sex)
        if self.teeth_present is not None:
            self.teeth_present = frozenset(self.teeth_present)
        if self.teeth_present_source is not None:
            self.teeth_present_source = InventorySource(self.teeth_present_source)
        if (self.teeth_present is None) != (self.teeth_present_source is None):
            raise ValueError(
                f"{self.image_id}: teeth_present and teeth_present_source must be set together"
            )

        for name in ("image_id", "path", "patient_id", "dataset"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if not self.readers:
            raise ValueError(
                f"{self.image_id}: readers is empty. List every annotator who reviewed "
                f"the image, including those who found nothing (use {CONSENSUS!r} for "
                "adjudicated-only datasets)"
            )
        stray = {a.annotator_id for a in self.annotations} - self.readers
        if stray:
            raise ValueError(
                f"{self.image_id}: annotations from {sorted(stray)} who are not in "
                f"readers={sorted(self.readers)}"
            )
        if self.teeth_present is not None:
            bad = self.teeth_present - VALID_FDI
            if bad:
                raise ValueError(f"{self.image_id}: invalid FDI in teeth_present: {sorted(bad)}")
            absent = {
                a.tooth_fdi for a in self.annotations
                if a.tooth_fdi is not None and a.tooth_fdi not in self.teeth_present
            }
            if absent:
                raise ValueError(
                    f"{self.image_id}: findings on teeth {sorted(absent)} not in teeth_present"
                )
        if self.age is not None and not (math.isfinite(self.age) and 0 <= self.age <= 120):
            raise ValueError(f"{self.image_id}: implausible age {self.age}")
        for name in ("width", "height"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{self.image_id}: {name} must be positive, got {v}")

    @property
    def group_key(self) -> str:
        """Split grouping key.

        Namespaced by dataset because patient ids are dataset-local: patient "17"
        in DENTEX and patient "17" in Tufts are different people.
        """
        return f"{self.dataset}/{self.patient_id}"

    @property
    def is_multi_annotator(self) -> bool:
        return len(self.readers) > 1

    def findings_by(self, annotator_id: str) -> list[Annotation]:
        """Findings by one reader. Empty list means they read it and found nothing."""
        if annotator_id not in self.readers:
            # A KeyError, not an empty list: "did not read" must never be
            # mistaken for "read and found nothing".
            raise KeyError(f"{annotator_id!r} did not read {self.image_id}")
        return [a for a in self.annotations if a.annotator_id == annotator_id]

    def has(self, finding: Finding, annotator_id: str | None = None) -> bool:
        if annotator_id is None:
            if self.is_multi_annotator:
                # Pooling across readers is an implicit "any reader" consensus,
                # which is the collapse rule 3 forbids. Make the caller choose.
                raise ValueError(
                    f"{self.image_id} has readers {sorted(self.readers)}; pass annotator_id"
                )
            (annotator_id,) = self.readers
        return any(a.finding is finding for a in self.findings_by(annotator_id))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for caching normalised records."""
        return {
            "image_id": self.image_id,
            "path": self.path,
            "patient_id": self.patient_id,
            "patient_id_source": self.patient_id_source.value,
            "dataset": self.dataset,
            "modality": self.modality.value,
            "readers": sorted(self.readers),
            "annotations": [a.to_dict() for a in self.annotations],
            "site_id": self.site_id,
            "device": self.device,
            "age": self.age,
            "sex": None if self.sex is None else self.sex.value,
            "teeth_present": None if self.teeth_present is None else sorted(self.teeth_present),
            "teeth_present_source": (
                None if self.teeth_present_source is None else self.teeth_present_source.value
            ),
            "width": self.width,
            "height": self.height,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RadiographRecord:
        return cls(
            image_id=d["image_id"],
            path=d["path"],
            patient_id=d["patient_id"],
            patient_id_source=PatientIdSource(d["patient_id_source"]),
            dataset=d["dataset"],
            modality=Modality(d["modality"]),
            readers=frozenset(d["readers"]),
            annotations=[Annotation.from_dict(a) for a in d.get("annotations", [])],
            site_id=d.get("site_id"),
            device=d.get("device"),
            age=d.get("age"),
            sex=None if d.get("sex") is None else Sex(d["sex"]),
            teeth_present=(
                None if d.get("teeth_present") is None else frozenset(d["teeth_present"])
            ),
            teeth_present_source=(
                None if d.get("teeth_present_source") is None
                else InventorySource(d["teeth_present_source"])
            ),
            width=d.get("width"),
            height=d.get("height"),
            meta=dict(d.get("meta", {})),
        )


@dataclass(frozen=True)
class DatasetSummary:
    n_images: int
    n_patients: int
    images_per_patient: float
    # If this is 1, image-level and patient-level splits are identical and the
    # E0 -> E1 "patient re-split" drop is zero by construction, not by virtue.
    max_images_per_patient: int
    n_annotations: int
    sites: dict[str, int]
    findings: dict[str, int]
    # Caries annotations only, keyed by stage name; "UNSTAGED" counts lesions
    # with no depth grade (e.g. DENTEX), which cannot enter stage-stratified metrics.
    caries_stages: dict[str, int]
    multi_reader_images: int
    patient_id_sources: dict[str, int]
    inventory_sources: dict[str, int]  # images per teeth_present source ("none" if absent)


def summarise(records: Sequence[RadiographRecord]) -> DatasetSummary:
    """Quick structural summary -- run this before training anything.

    The ratios here are the first place a leak or an imbalance shows up.
    """
    per_patient: dict[str, int] = {}
    sites: dict[str, int] = {}
    findings: dict[str, int] = {}
    stages: dict[str, int] = {}
    id_sources: dict[str, int] = {}
    inventories: dict[str, int] = {}

    for r in records:
        per_patient[r.group_key] = per_patient.get(r.group_key, 0) + 1
        id_sources[r.patient_id_source.value] = id_sources.get(r.patient_id_source.value, 0) + 1
        inv = "none" if r.teeth_present_source is None else r.teeth_present_source.value
        inventories[inv] = inventories.get(inv, 0) + 1
        if r.site_id is not None:
            sites[r.site_id] = sites.get(r.site_id, 0) + 1
        for a in r.annotations:
            findings[a.finding.value] = findings.get(a.finding.value, 0) + 1
            if a.finding in CARIES_FINDINGS:
                key = "UNSTAGED" if a.stage is None else a.stage.name
                stages[key] = stages.get(key, 0) + 1

    n_patients = len(per_patient)
    return DatasetSummary(
        n_images=len(records),
        n_patients=n_patients,
        images_per_patient=len(records) / n_patients if n_patients else 0.0,
        max_images_per_patient=max(per_patient.values(), default=0),
        n_annotations=sum(len(r.annotations) for r in records),
        sites=dict(sorted(sites.items())),
        findings=dict(sorted(findings.items())),
        caries_stages=dict(sorted(stages.items())),
        multi_reader_images=sum(1 for r in records if r.is_multi_annotator),
        patient_id_sources=dict(sorted(id_sources.items())),
        inventory_sources=dict(sorted(inventories.items())),
    )


def iter_annotations(
    records: Iterable[RadiographRecord],
) -> Iterable[tuple[RadiographRecord, Annotation]]:
    for r in records:
        for a in r.annotations:
            yield r, a
