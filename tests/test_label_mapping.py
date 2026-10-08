from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dcai.data.label_mapping import LabelMapping, MappingStatus

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "label_mapping_tufts_dentex.yaml"


def mutated(tmp_path: Path, edit) -> Path:
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_shipped_mapping_loads_and_reports_nothing_until_verified() -> None:
    m = LabelMapping.from_yaml(CONFIG)
    assert m.source_dataset == "tufts" and m.target_dataset == "dentex"
    assert not m.verified_against_files
    assert m.reportable() == () and m.delta_only() == ()
    assert all(x.rationale.strip() for x in m.mappings)


def test_verified_mapping_reports_only_unambiguous_unblocked(tmp_path: Path) -> None:
    m = LabelMapping.from_yaml(mutated(tmp_path, lambda r: r.update(verified_against_files=True)))
    assert m.reportable() == ("periapical_lesion",)
    assert m.delta_only() == ("any_caries",)  # unambiguous, but never standalone
    statuses = {x.target: x.status for x in m.mappings}
    assert statuses["any_caries"] is MappingStatus.UNAMBIGUOUS
    assert statuses["deep_caries"] is MappingStatus.AMBIGUOUS


def test_classify() -> None:
    m = LabelMapping.from_yaml(CONFIG)
    apical = {"anatomical_location": "periapical", "radiodensity": "radiolucent"}
    assert m.classify(apical) == ("periapical_lesion",)
    assert m.classify({**apical, "radiodensity": "radiopaque"}) == ()  # condensing osteitis
    deep = {"abnormality_category": "caries", "effect_on_surrounding_structure": "pulp_involvement"}
    assert set(m.classify(deep)) == {"any_caries", "deep_caries"}
    assert m.classify({"abnormality_category": "cyst"}) == ()  # unmapped


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda r: r.update(version="1"), "MAJOR.MINOR.PATCH"),
        (lambda r: r["mappings"][0].update(rationale="  "), "no rationale"),
        (lambda r: r["mappings"][0].update(conditions=[]), "no conditions"),
        (lambda r: r["mappings"][0]["conditions"][0].update(axis="colour"), "unknown axis"),
        (lambda r: r["mappings"][0].update(target="restoration"), "not a target"),
        (lambda r: r["mappings"].append(dict(r["mappings"][0])), "mapped twice"),
        (lambda r: r["target"]["merged"].update(x=["implant"]), "unknown targets"),
        (lambda r: r.update(unmapped_findings="count_as_negative"), "exclude_from_negatives"),
        (lambda r: r["mappings"][0].update(status="probably"), "probably"),
        (lambda r: r["mappings"][0].update(report_as="headline"), "headline"),
    ],
)
def test_validation(tmp_path: Path, edit, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LabelMapping.from_yaml(mutated(tmp_path, edit))


# --- domain shift: the only route out for delta-only classes -------------------------


def _est(value: float, seed: int):
    import numpy as np

    from dcai.eval.bootstrap import grouped_bootstrap

    x = np.random.default_rng(seed).normal(value, 0.05, 60)
    return grouped_bootstrap(lambda idx: float(x[idx].mean()), np.arange(60), seed=seed,
                             n_boot=200)


def test_domain_shift_reports_only_the_delta(tmp_path: Path) -> None:
    from dcai.eval.domain_shift import domain_shift

    m = LabelMapping.from_yaml(mutated(tmp_path, lambda r: r.update(verified_against_files=True)))
    shift = domain_shift(_est(0.80, 1), _est(0.60, 2), mapping=m, target="any_caries",
                         metric="tooth-level sensitivity")
    assert shift.delta.value == pytest.approx(0.20, abs=0.02)
    assert "v0.2.0" in shift.definition and "internal − external" in str(shift)
    # The pooled external estimate is not retained anywhere on the result.
    assert not any(isinstance(v, type(shift.delta)) and v is not shift.delta
                   for v in vars(shift).values())


def test_domain_shift_refuses_standalone_classes_and_unverified_mappings(tmp_path: Path) -> None:
    from dcai.eval.domain_shift import domain_shift

    verified = LabelMapping.from_yaml(
        mutated(tmp_path, lambda r: r.update(verified_against_files=True)))
    with pytest.raises(ValueError, match="report it directly"):
        domain_shift(_est(0.8, 1), _est(0.6, 2), mapping=verified, target="periapical_lesion",
                     metric="sens")
    with pytest.raises(ValueError, match="verified"):
        domain_shift(_est(0.8, 1), _est(0.6, 2), mapping=LabelMapping.from_yaml(CONFIG),
                     target="any_caries", metric="sens")
    with pytest.raises(KeyError, match="no mapping"):
        verified.mapping_for("restoration")
