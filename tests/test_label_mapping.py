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
    assert m.reportable() == ()
    assert m.blocked() == {"any_caries": "rule_2"}
    assert all(x.rationale.strip() for x in m.mappings)


def test_verified_mapping_reports_only_unambiguous_unblocked(tmp_path: Path) -> None:
    m = LabelMapping.from_yaml(mutated(tmp_path, lambda r: r.update(verified_against_files=True)))
    assert m.reportable() == ("periapical_lesion",)
    statuses = {x.target: x.status for x in m.mappings}
    assert statuses["any_caries"] is MappingStatus.UNAMBIGUOUS  # unambiguous, but blocked
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
    ],
)
def test_validation(tmp_path: Path, edit, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LabelMapping.from_yaml(mutated(tmp_path, edit))
