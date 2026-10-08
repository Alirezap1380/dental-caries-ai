"""Patient-level splitting, with a leakage check that cannot be skipped.

Random image-level splits put the same mouth in train and test: bitewings come in
sets of four, panoramics have repeat visits, and a model that has seen a patient's
anatomy, restorations and acquisition quirks will "recognise" them at test time.
That is the single most common source of inflated numbers in this literature.

How this module enforces the rule rather than documenting it:

- Every split is a `Split`, and constructing one runs `check_leakage`. No function
  here returns partitions that have not been checked.
- Failures raise `PatientLeakageError` with an explicit `raise`, never an `assert`
  statement: `python -O` strips asserts, which would silently disable the one
  check this project is built on.
- The naive image-level split (experiment E0) is the only exception. It must be
  built with `kind=IMAGE_NAIVE, allow_leakage=True`, and instead of raising it
  *measures* the leak, so E0 results carry their own contamination figure.
  `allow_leakage` on any other kind is rejected, so it cannot be used to silence
  a failing patient split.
- Grouping is on `RadiographRecord.group_key` (dataset/patient_id).
- Results depend only on the set of records and the seed, not on input order:
  loaders that glob the filesystem see different orders on different machines.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dcai.data.schema import PatientIdSource, RadiographRecord

# Maps one patient's records to a stratum label, e.g. "has any staged caries" or
# site_id. Stratification is per patient, because the patient is the unit we split.
Stratifier = Callable[[Sequence[RadiographRecord]], Hashable]

DEFAULT_FRACTIONS: Mapping[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}


class PatientLeakageError(AssertionError):
    """A patient, or a single image, appears in more than one partition."""


class NoLeakageToMeasureWarning(UserWarning):
    """The naive split leaked nothing, so E0 vs. E1 compares a split with itself."""


class SplitKind(str, Enum):
    PATIENT = "patient"
    SITE_HOLDOUT = "site_holdout"
    IMAGE_NAIVE = "image_naive"


@dataclass(frozen=True)
class LeakageReport:
    n_groups: int
    leaked_groups: frozenset[str]
    # Groups whose patient_id was assumed rather than provided. The check passes
    # over them by construction, so it is evidence of nothing for these records.
    unverifiable_groups: int

    @property
    def is_clean(self) -> bool:
        return not self.leaked_groups

    @property
    def leaked_fraction(self) -> float:
        return len(self.leaked_groups) / self.n_groups if self.n_groups else 0.0


def check_leakage(
    partitions: Mapping[str, Sequence[RadiographRecord]],
    *,
    raise_on_leak: bool = True,
) -> LeakageReport:
    """Check that no patient spans two partitions.

    A duplicated image always raises, even when `raise_on_leak=False`: the same
    image in two places is a bug, not the patient leak that E0 sets out to measure.
    """
    image_home: dict[str, str] = {}
    group_homes: dict[str, set[str]] = defaultdict(set)
    unverifiable: set[str] = set()

    for name, recs in partitions.items():
        for r in recs:
            prev = image_home.get(r.image_id)
            if prev is not None:
                raise PatientLeakageError(
                    f"image {r.image_id!r} appears more than once (in {prev!r} and {name!r})"
                )
            image_home[r.image_id] = name
            group_homes[r.group_key].add(name)
            if r.patient_id_source is PatientIdSource.ASSUMED_UNIQUE:
                unverifiable.add(r.group_key)

    leaked = {g: sorted(homes) for g, homes in group_homes.items() if len(homes) > 1}
    if leaked and raise_on_leak:
        examples = "; ".join(f"{g} in {homes}" for g, homes in sorted(leaked.items())[:5])
        raise PatientLeakageError(
            f"{len(leaked)} patient(s) appear in more than one partition: {examples}"
        )
    return LeakageReport(
        n_groups=len(group_homes),
        leaked_groups=frozenset(leaked),
        unverifiable_groups=len(unverifiable),
    )


@dataclass(frozen=True, eq=False)
class Split:
    """Named, disjoint partitions of a record set: train/val/test, or one CV fold."""

    partitions: Mapping[str, Sequence[RadiographRecord]]
    kind: SplitKind
    name: str = ""
    seed: int | None = None
    allow_leakage: bool = False
    leakage: LeakageReport = field(init=False)

    def __post_init__(self) -> None:
        if self.allow_leakage and self.kind is not SplitKind.IMAGE_NAIVE:
            raise ValueError(
                f"allow_leakage is reserved for {SplitKind.IMAGE_NAIVE.value} splits, "
                f"got kind={self.kind.value}"
            )
        parts = {k: tuple(v) for k, v in self.partitions.items()}
        empty = [k for k, v in parts.items() if not v]
        if empty:
            raise ValueError(f"empty partition(s) {empty} in split {self.name!r}")
        object.__setattr__(self, "partitions", parts)
        object.__setattr__(
            self, "leakage", check_leakage(parts, raise_on_leak=not self.allow_leakage)
        )

    def __getitem__(self, partition: str) -> tuple[RadiographRecord, ...]:
        return tuple(self.partitions[partition])

    def image_ids(self, partition: str) -> list[str]:
        return [r.image_id for r in self.partitions[partition]]


def _group(records: Sequence[RadiographRecord]) -> dict[str, list[RadiographRecord]]:
    """Group by patient, validating the input and fixing a canonical order."""
    if not records:
        raise ValueError("no records to split")
    seen: set[str] = set()
    for r in records:
        if r.image_id in seen:
            raise ValueError(f"duplicate image_id {r.image_id!r} in input records")
        seen.add(r.image_id)

    groups: dict[str, list[RadiographRecord]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r.image_id):
        groups[r.group_key].append(r)
    return dict(sorted(groups.items()))


def _check_fractions(fractions: Mapping[str, float]) -> np.ndarray:
    if len(fractions) < 2:
        raise ValueError("need at least two partitions")
    if any(f <= 0 for f in fractions.values()):
        raise ValueError(f"fractions must be positive, got {dict(fractions)}")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions.values())}")
    bounds = np.cumsum(list(fractions.values()))
    bounds[-1] = 1.0  # absorb float drift so the last partition's upper edge is exact
    return bounds


def _allocate(sizes: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Assign items, in order, to partitions so each gets ~its share of images.

    Weighted by image count rather than patient count: a patient with a full
    bitewing set is four images of test data, not one.
    """
    midpoints = (np.cumsum(sizes) - sizes / 2) / sizes.sum()
    return np.searchsorted(bounds, midpoints, side="right")


def patient_split(
    records: Sequence[RadiographRecord],
    *,
    seed: int,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    stratify: Stratifier | None = None,
) -> Split:
    """Random patient-level split into named partitions."""
    bounds = _check_fractions(fractions)
    groups = _group(records)
    names = list(fractions)
    rng = np.random.default_rng(seed)

    strata: dict[Hashable, list[str]] = defaultdict(list)
    for key, recs in groups.items():
        strata[stratify(recs) if stratify else None].append(key)

    parts: dict[str, list[RadiographRecord]] = {n: [] for n in names}
    for stratum in sorted(strata, key=repr):
        keys = [strata[stratum][i] for i in rng.permutation(len(strata[stratum]))]
        sizes = np.array([len(groups[k]) for k in keys], dtype=float)
        for key, idx in zip(keys, _allocate(sizes, bounds)):
            parts[names[idx]].extend(groups[key])

    return Split(partitions=parts, kind=SplitKind.PATIENT, name="patient", seed=seed)


def patient_kfold(
    records: Sequence[RadiographRecord],
    *,
    seed: int,
    n_splits: int = 5,
    stratify: Stratifier | None = None,
) -> list[Split]:
    """Patient-grouped k-fold CV. Each record is in exactly one test fold."""
    groups = _group(records)
    if not 2 <= n_splits <= len(groups):
        raise ValueError(f"n_splits={n_splits} needs 2 <= n_splits <= n_patients={len(groups)}")

    flat = [r for recs in groups.values() for r in recs]
    group_label = {k: (stratify(recs) if stratify else 0) for k, recs in groups.items()}
    encoding = {lab: i for i, lab in enumerate(sorted(set(group_label.values()), key=repr))}
    y = np.array([encoding[group_label[r.group_key]] for r in flat])

    # With a constant y this still balances fold sizes: StratifiedGroupKFold
    # minimises the spread of each class's share across folds.
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        Split(
            partitions={"train": [flat[j] for j in tr], "test": [flat[j] for j in te]},
            kind=SplitKind.PATIENT,
            name=f"fold{i}",
            seed=seed,
        )
        for i, (tr, te) in enumerate(
            cv.split(np.zeros(len(flat)), y, groups=[r.group_key for r in flat])
        )
    ]

    # Across folds: test sets must be patient-disjoint and cover everything once,
    # otherwise pooled CV metrics double-count some patients and omit others.
    check_leakage({f.name: f["test"] for f in folds})
    n_tested = sum(len(f["test"]) for f in folds)
    if n_tested != len(flat):
        raise PatientLeakageError(f"CV test folds cover {n_tested} of {len(flat)} records")
    return folds


def leave_one_site_out(records: Sequence[RadiographRecord]) -> list[Split]:
    """One split per site, holding that site out as the test set.

    A patient imaged at two sites would leak across the holdout; `Split` raises
    on that like any other leak.
    """
    _group(records)
    missing = [r.image_id for r in records if r.site_id is None]
    if missing:
        raise ValueError(f"{len(missing)} record(s) have no site_id, e.g. {missing[:3]}")
    sites = sorted({r.site_id for r in records if r.site_id is not None})
    if len(sites) < 2:
        raise ValueError(f"need at least two sites to hold one out, got {sites}")

    return [
        Split(
            partitions={
                "train": [r for r in records if r.site_id != site],
                "test": [r for r in records if r.site_id == site],
            },
            kind=SplitKind.SITE_HOLDOUT,
            name=f"holdout={site}",
        )
        for site in sites
    ]


def naive_image_split(
    records: Sequence[RadiographRecord],
    *,
    seed: int,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
) -> Split:
    """E0 ONLY: the leaky image-level random split the literature typically uses.

    Exists so the inflated baseline can be reproduced and then demolished. The
    returned split's `leakage` report is part of the E0 result and must be
    reported alongside its metrics.
    """
    bounds = _check_fractions(fractions)
    ordered = [r for recs in _group(records).values() for r in recs]
    names = list(fractions)
    rng = np.random.default_rng(seed)

    shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
    parts: dict[str, list[RadiographRecord]] = {n: [] for n in names}
    for r, idx in zip(shuffled, _allocate(np.ones(len(shuffled)), bounds)):
        parts[names[idx]].append(r)

    split = Split(
        partitions=parts,
        kind=SplitKind.IMAGE_NAIVE,
        name="image_naive",
        seed=seed,
        allow_leakage=True,
    )
    if split.leakage.is_clean:
        warnings.warn(
            "naive image-level split leaked no patients (every patient has one image, "
            "or ids are assumed unique). Image- and patient-level splits coincide on "
            "this data, so the E0 -> E1 drop is zero by construction.",
            NoLeakageToMeasureWarning,
            stacklevel=2,
        )
    return split
