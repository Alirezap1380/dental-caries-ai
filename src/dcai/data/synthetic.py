"""Synthetic records for testing the harness before any real data arrives.

Structure only, no pixels: the generator reproduces the properties the splits and
metrics have to cope with -- several images per patient, patients nested in sites,
multiple readers who disagree, unstaged lesions -- so the leakage and agreement
logic is exercised on the same shapes the real datasets have.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dcai.data.schema import (
    Annotation,
    BBox,
    CariesStage,
    Finding,
    InventorySource,
    Modality,
    PatientIdSource,
    RadiographRecord,
    Sex,
)

_LESION_STAGES = [s for s in CariesStage if s is not CariesStage.NONE]
_TEETH = [11, 14, 16, 21, 24, 26, 31, 34, 36, 41, 44, 46]


def make_records(
    *,
    n_patients: int = 40,
    images_per_patient: int | tuple[int, int] = (1, 4),
    sites: Sequence[str] = ("site_a", "site_b", "site_c"),
    readers: Sequence[str] = ("consensus",),
    caries_rate: float = 0.3,
    stage_fraction: float = 0.8,
    dataset: str = "synthetic",
    modality: Modality = Modality.BITEWING,
    patient_id_source: PatientIdSource = PatientIdSource.PROVIDED,
    seed: int = 0,
) -> list[RadiographRecord]:
    """Generate structurally realistic records.

    `images_per_patient` is an exact count or an inclusive (lo, hi) range.
    `stage_fraction` is the share of caries lesions that carry a depth grade;
    the rest are unstaged, as in DENTEX.
    """
    rng = np.random.default_rng(seed)
    lo, hi = (
        (images_per_patient, images_per_patient)
        if isinstance(images_per_patient, int)
        else images_per_patient
    )
    records: list[RadiographRecord] = []

    for p in range(n_patients):
        patient_id = f"p{p:04d}"
        # A patient is imaged at one site; site varies between patients, not within.
        site = sites[p % len(sites)] if sites else None
        age = float(rng.integers(18, 80))
        sex = Sex.FEMALE if rng.random() < 0.5 else Sex.MALE

        for i in range(int(rng.integers(lo, hi + 1))):
            image_id = f"{dataset}-{patient_id}-{i}"
            annotations: list[Annotation] = []
            for reader in readers:
                if rng.random() < caries_rate:
                    x, y = rng.uniform(0, 400, size=2)
                    stage = (
                        _LESION_STAGES[int(rng.integers(len(_LESION_STAGES)))]
                        if rng.random() < stage_fraction
                        else None
                    )
                    annotations.append(
                        Annotation(
                            finding=Finding.CARIES,
                            annotator_id=reader,
                            bbox=BBox(x, y, x + 40, y + 40),
                            tooth_fdi=_TEETH[int(rng.integers(len(_TEETH)))],
                            stage=stage,
                        )
                    )
            records.append(
                RadiographRecord(
                    image_id=image_id,
                    path=f"/synthetic/{image_id}.png",
                    patient_id=patient_id,
                    patient_id_source=patient_id_source,
                    dataset=dataset,
                    modality=modality,
                    readers=frozenset(readers),
                    annotations=annotations,
                    site_id=site,
                    age=age,
                    sex=sex,
                    width=512,
                    height=512,
                )
            )
    return records


# --- Reader study: a latent truth, imperfect readers, tooth inventories ------------


@dataclass(frozen=True)
class ReaderProfile:
    name: str
    # P(reader marks a true lesion), for (caries, deep caries). Shallow lesions are
    # missed far more often, which is where real inter-observer disagreement lives.
    sensitivity: tuple[float, float]
    false_positive: float  # P(reader marks a sound tooth as caries)
    misgrade: float  # P(reader calls a lesion they saw at the other depth)


DEFAULT_READERS = (
    ReaderProfile("r1", (0.70, 0.94), 0.015, 0.20),
    ReaderProfile("r2", (0.60, 0.92), 0.010, 0.25),
    ReaderProfile("r3", (0.78, 0.95), 0.025, 0.20),  # liberal
    ReaderProfile("r4", (0.55, 0.90), 0.008, 0.30),  # conservative
)


@dataclass(frozen=True)
class ReaderStudy:
    records: list[RadiographRecord]
    # Latent truth no reader has access to: (image_id, fdi) -> 0 sound, 1 caries, 2 deep.
    truth: dict[tuple[str, int], int]
    lesion_boxes: dict[tuple[str, int], BBox]  # true lesion location, lesion teeth only


def tooth_box(fdi: int) -> BBox:
    """Where a tooth sits on a synthetic 1100 x 500 panoramic."""
    quadrant, tooth = divmod(fdi, 10)
    x = (8 - tooth) * 62 + 10 if quadrant in (1, 4) else 560 + (tooth - 1) * 62
    y = 60 if quadrant in (1, 2) else 280
    return BBox(x, y, x + 52, y + 150)


def _inside(box: BBox, rng: np.random.Generator, size: float = 20.0) -> BBox:
    x = rng.uniform(box.x1, box.x2 - size)
    y = rng.uniform(box.y1, box.y2 - size)
    return BBox(x, y, x + size, y + size)


def _jitter(box: BBox, rng: np.random.Generator, px: float = 2.0) -> BBox:
    dx1, dy1, dx2, dy2 = rng.uniform(-px, px, size=4)
    return BBox(box.x1 + dx1, box.y1 + dy1, box.x2 + dx2, box.y2 + dy2)


ALL_TEETH = tuple(sorted(int(f"{q}{t}") for q in (1, 2, 3, 4) for t in range(1, 9)))
_FINDING_BY_GRADE = {1: Finding.CARIES, 2: Finding.DEEP_CARIES}


def make_reader_study(
    *,
    n_patients: int,
    seed: int,
    images_per_patient: tuple[int, int] = (1, 2),
    readers: Sequence[ReaderProfile] = DEFAULT_READERS,
    full_read_fraction: float = 0.8,
    sites: Sequence[str] = ("site_a", "site_b", "site_c"),
    demographics: bool = True,
    caries_rate: float = 0.06,
    deep_rate: float = 0.03,
    dataset: str = "synthetic",
) -> ReaderStudy:
    """A DENTEX-shaped panoramic reader study with a known latent truth.

    Deep lesions and missing teeth become more common with age, so subgroup
    analyses see the age/depth confounding they will meet on real data. A
    patient's repeat image shares their dentition. That shared dentition is
    exactly what leaks under an image-level split.
    `full_read_fraction` of images are read by every reader; the rest by all
    but one, so the design is incomplete as real reader studies are.
    """
    rng = np.random.default_rng(seed)
    lo, hi = images_per_patient
    records: list[RadiographRecord] = []
    truth: dict[tuple[str, int], int] = {}
    boxes: dict[tuple[str, int], BBox] = {}

    for p in range(n_patients):
        patient_id = f"p{p:04d}"
        site = sites[p % len(sites)]
        age = float(rng.integers(18, 81))
        sex = Sex.FEMALE if rng.random() < 0.5 else Sex.MALE
        wisdom = [t for t in ALL_TEETH if t % 10 == 8 and rng.random() < 0.5]
        n_missing = min(rng.poisson(age / 25), 10)
        others = [t for t in ALL_TEETH if t % 10 != 8]
        missing = set(wisdom) | set(rng.choice(others, size=n_missing, replace=False).tolist())
        present = frozenset(ALL_TEETH) - missing

        p_deep = deep_rate * (0.4 + age / 50)
        dentition = {
            t: int(rng.choice(3, p=[1 - caries_rate - p_deep, caries_rate, p_deep]))
            for t in present
        }
        lesion_at = {t: _inside(tooth_box(t), rng) for t, g in dentition.items() if g}

        for i in range(int(rng.integers(lo, hi + 1))):
            image_id = f"{dataset}-{patient_id}-{i}"
            readers_here = list(readers)
            if rng.random() > full_read_fraction:
                readers_here.pop(int(rng.integers(len(readers_here))))

            annotations: list[Annotation] = []
            for reader in readers_here:
                for t in sorted(present):
                    g = dentition[t]
                    if g and rng.random() < reader.sensitivity[g - 1]:
                        called = 3 - g if rng.random() < reader.misgrade else g
                        box = _jitter(lesion_at[t], rng)
                    elif not g and rng.random() < reader.false_positive:
                        called, box = 1, _inside(tooth_box(t), rng)
                    else:
                        continue
                    annotations.append(Annotation(
                        _FINDING_BY_GRADE[called], reader.name, bbox=box, tooth_fdi=t
                    ))
            for t in present:
                truth[(image_id, t)] = dentition[t]
                if t in lesion_at:
                    boxes[(image_id, t)] = lesion_at[t]

            records.append(RadiographRecord(
                image_id=image_id,
                path=f"/synthetic/{image_id}.png",
                patient_id=patient_id,
                patient_id_source=PatientIdSource.PROVIDED,
                dataset=dataset,
                modality=Modality.PANORAMIC,
                readers=frozenset(r.name for r in readers_here),
                annotations=annotations,
                site_id=site,
                age=age if demographics else None,
                sex=sex if demographics else None,
                teeth_present=present,
                teeth_present_source=InventorySource.ANNOTATED,
                width=1100,
                height=500,
            ))
    return ReaderStudy(records, truth, boxes)
