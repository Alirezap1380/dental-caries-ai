"""Domain shift on a pooled class: the only route out for delta-only mappings.

Rule 2 forbids pooling across lesion depth in *performance claims*. A cross-dataset
comparison on a pooled class is a different kind of number: a *domain-shift
measurement*. Internal and external pool identically under one mapped
definition, so their difference is like for like even though neither side is
depth-stratified. The depth-stratified result stays on the internal side, where
the labels support it.

The pooled external number on its own would read as a performance claim, so it is
never exposed: `DomainShift` keeps only the delta, the definition it was measured
on, and the sample sizes. The bootstrap is independent (different patients on
each side). What is matched is the class definition, not the resamples.
"""

from __future__ import annotations

from dataclasses import dataclass

from dcai.data.label_mapping import LabelMapping, ReportAs
from dcai.eval.bootstrap import Estimate, difference


@dataclass(frozen=True)
class DomainShift:
    target: str
    definition: str  # mapping file version + target: what "the same class" means
    metric: str
    delta: Estimate  # internal - external; positive = worse externally
    n_groups_internal: int
    n_groups_external: int

    def __str__(self) -> str:
        return (f"{self.metric} on '{self.target}' ({self.definition}): internal − external "
                f"= {self.delta}")


def domain_shift(
    internal: Estimate,
    external: Estimate,
    *,
    mapping: LabelMapping,
    target: str,
    metric: str,
) -> DomainShift:
    """Matched-definition delta for a `domain_shift_delta` class.

    Raises unless the mapping is verified and marks `target` as delta-only.
    Standalone-reportable classes do not belong here: report them directly.
    """
    m = mapping.mapping_for(target)
    if m.report_as is not ReportAs.DOMAIN_SHIFT_DELTA:
        raise ValueError(f"{target!r} is not a domain-shift-delta class; report it directly")
    if target not in mapping.delta_only():
        raise ValueError(
            f"{target!r} is not reportable yet: the mapping must be unambiguous and verified "
            "against the source files"
        )
    return DomainShift(
        target=target,
        definition=f"{mapping.source_dataset}->{mapping.target_dataset} mapping "
                   f"v{mapping.version}",
        metric=metric,
        delta=difference(internal, external, paired=False),
        n_groups_internal=internal.n_groups,
        n_groups_external=external.n_groups,
    )
