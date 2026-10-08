"""Subgroup performance by age and sex.

Two rules carry over from the rest of the harness:

- Rule 2 applies inside subgroups. Lesion depth is confounded with age (older
  patients carry more deep lesions, which are easier to detect), so pooled
  sensitivity per age band would invent an "age effect" out of depth. Each
  subgroup therefore gets a full depth-stratified report. The cells get small:
  that is the honest consequence, and the CIs show it.
- Missing metadata is reported, not skipped. If no unit carries the attribute,
  the report says it is not computable on this data. If only some units carry
  it, the coverage is stated alongside the results.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dcai.eval.scales import OrdinalScale
from dcai.eval.stratified import StratifiedReport, stratified_report

DEFAULT_AGE_EDGES = (18, 35, 50, 65)


def age_bands(ages: Sequence[float] | np.ndarray, edges: Sequence[int] = DEFAULT_AGE_EDGES):
    """Map ages to band labels like "35-49"; NaN -> None."""
    ages = np.asarray(ages, dtype=float)
    labels = [f"<{edges[0]}"]
    labels += [f"{lo}-{hi - 1}" for lo, hi in itertools.pairwise(edges)]
    labels.append(f"{edges[-1]}+")
    idx = np.searchsorted(np.asarray(edges), ages, side="right")
    return np.array(
        [None if np.isnan(a) else labels[i] for a, i in zip(ages, idx, strict=True)], dtype=object
    )


@dataclass(frozen=True)
class SubgroupCell:
    value: str
    n_units: int
    n_groups: int
    report: StratifiedReport | None
    reason: str | None  # why `report` is None


@dataclass(frozen=True)
class SubgroupReport:
    attribute: str
    n_units: int
    coverage: float  # fraction of units with this attribute known
    cells: tuple[SubgroupCell, ...]
    not_computable: str | None  # set when no unit carries the attribute


def subgroup_report(
    values: Sequence[str | None] | np.ndarray,
    levels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    attribute: str,
    scale: OrdinalScale,
    threshold: float,
    seed: int,
    n_boot: int = 2000,
) -> SubgroupReport:
    values = np.asarray(values, dtype=object)
    levels, scores, groups = np.asarray(levels), np.asarray(scores), np.asarray(groups)
    if not values.shape == levels.shape == scores.shape == groups.shape:
        raise ValueError("values, levels, scores and groups must be the same length")
    known = np.array([v is not None for v in values], dtype=bool)
    if not known.any():
        return SubgroupReport(
            attribute, values.size, 0.0, (),
            f"no {attribute} metadata on any of {values.size} units: not computable on this data",
        )

    cells = []
    for value in sorted({v for v in values[known]}):
        sel = values == value
        n_groups = len(set(groups[sel]))
        report, reason = None, None
        if n_groups < 2:
            reason = "fewer than two patients"
        elif not (levels[sel] > 0).any():
            reason = "no lesions in this subgroup"
        else:
            report = stratified_report(
                levels[sel], scores[sel], groups[sel],
                scale=scale, threshold=threshold, seed=seed, n_boot=n_boot,
            )
        cells.append(SubgroupCell(str(value), int(sel.sum()), n_groups, report, reason))
    return SubgroupReport(attribute, values.size, float(known.mean()), tuple(cells), None)
