"""Versioned cross-dataset label mappings (configs/label_mapping_*.yaml).

Rule 4 says external validation is the real number. But "external" only means
something if both datasets label the same thing. A mapping is a set of
methodological claims, so it lives in a reviewed, versioned file with a rationale
per class, and this module enforces what may be reported from it:

- Only `unambiguous` mappings are reportable, and only once the file has been
  checked against the real annotation files (`verified_against_files`).
- A mapping can be `report_as: domain_shift_delta`. The merged any-caries class
  pools across depth, which rule 2 forbids in performance claims, so it is never
  reportable standalone. Its only route out is `dcai.eval.domain_shift`: a
  delta against the internal result on the same pooled definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class MappingStatus(str, Enum):
    UNAMBIGUOUS = "unambiguous"
    AMBIGUOUS = "ambiguous"


class ReportAs(str, Enum):
    STANDALONE = "standalone"
    DOMAIN_SHIFT_DELTA = "domain_shift_delta"  # only as internal - external, same definition


@dataclass(frozen=True)
class Condition:
    axis: str
    values: frozenset[str]


@dataclass(frozen=True)
class ClassMapping:
    target: str
    status: MappingStatus
    conditions: tuple[Condition, ...]
    rationale: str
    report_as: ReportAs = ReportAs.STANDALONE

    def matches(self, description: Mapping[str, str]) -> bool:
        """True if a source annotation's axis values satisfy every condition."""
        return all(description.get(c.axis) in c.values for c in self.conditions)


@dataclass(frozen=True)
class LabelMapping:
    version: str
    verified_against_files: bool
    source_dataset: str
    source_axes: tuple[str, ...]
    target_dataset: str
    target_classes: tuple[str, ...]
    merged: dict[str, tuple[str, ...]]
    mappings: tuple[ClassMapping, ...]
    unmapped_findings: str

    def __post_init__(self) -> None:
        if not self.version or self.version.count(".") != 2:
            raise ValueError(f"version must be MAJOR.MINOR.PATCH, got {self.version!r}")
        for name, parts in self.merged.items():
            unknown = set(parts) - set(self.target_classes)
            if unknown:
                raise ValueError(f"merged class {name!r} lists unknown targets {sorted(unknown)}")
        allowed = set(self.target_classes) | set(self.merged)
        seen: set[str] = set()
        for m in self.mappings:
            if m.target not in allowed:
                raise ValueError(f"mapping target {m.target!r} is not a target or merged class")
            if m.target in seen:
                raise ValueError(f"target {m.target!r} is mapped twice")
            seen.add(m.target)
            if not m.rationale.strip():
                raise ValueError(f"mapping for {m.target!r} has no rationale")
            if not m.conditions:
                raise ValueError(f"mapping for {m.target!r} has no conditions")
            for c in m.conditions:
                if c.axis not in self.source_axes:
                    raise ValueError(f"mapping for {m.target!r} uses unknown axis {c.axis!r}")
        if self.unmapped_findings != "exclude_from_negatives":
            raise ValueError("unmapped_findings must be 'exclude_from_negatives'")

    @classmethod
    def from_yaml(cls, path: str | Path) -> LabelMapping:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            version=str(raw["version"]),
            verified_against_files=bool(raw["verified_against_files"]),
            source_dataset=raw["source"]["dataset"],
            source_axes=tuple(raw["source"]["axes"]),
            target_dataset=raw["target"]["dataset"],
            target_classes=tuple(raw["target"]["classes"]),
            merged={k: tuple(v) for k, v in raw["target"].get("merged", {}).items()},
            mappings=tuple(
                ClassMapping(
                    target=m["target"],
                    status=MappingStatus(m["status"]),
                    conditions=tuple(Condition(c["axis"], frozenset(c["values"]))
                                     for c in m["conditions"]),
                    rationale=m.get("rationale", ""),
                    report_as=ReportAs(m.get("report_as", "standalone")),
                )
                for m in raw["mappings"]
            ),
            unmapped_findings=raw["unmapped_findings"],
        )

    def _usable(self, report_as: ReportAs) -> tuple[str, ...]:
        if not self.verified_against_files:
            return ()
        return tuple(
            m.target for m in self.mappings
            if m.status is MappingStatus.UNAMBIGUOUS and m.report_as is report_as
        )

    def reportable(self) -> tuple[str, ...]:
        """Classes whose external results may be reported standalone. Empty until verified."""
        return self._usable(ReportAs.STANDALONE)

    def delta_only(self) -> tuple[str, ...]:
        """Classes reportable only as a domain-shift delta. Empty until verified."""
        return self._usable(ReportAs.DOMAIN_SHIFT_DELTA)

    def mapping_for(self, target: str) -> ClassMapping:
        for m in self.mappings:
            if m.target == target:
                return m
        raise KeyError(f"no mapping for {target!r}")

    def classify(self, description: Mapping[str, str]) -> tuple[str, ...]:
        """Every mapped target a source annotation satisfies (may be several, or none).

        None means the finding is unmapped and falls under `unmapped_findings`.
        """
        return tuple(m.target for m in self.mappings if m.matches(description))
