"""End-to-end: the harness composes on synthetic data and renders a report.

Unit tests prove each piece; this proves the pieces fit. It found a real bug
(stratified AP on an image with boxes but no lesions) that no unit test covered.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from dcai.eval.bootstrap import grouped_bootstrap
from dcai.figures import write_figures
from dcai.pipeline import StudyConfig, SyntheticConfig, evaluate, synthetic_inputs
from dcai.report import (
    LEAK_THRESHOLD,
    MIN_CELL,
    _Formatter,
    render_markdown,
    small_cell_flags,
    tripwire,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic.yaml"
E0_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic_e0_leaky.yaml"


@pytest.fixture(scope="module")
def small_run():
    cfg = dataclasses.replace(
        StudyConfig.from_yaml(CONFIG),
        n_boot=60,
        synthetic=SyntheticConfig(internal_patients=150, external_patients=60,
                                  external_demographics=False),
    )
    internal, external = synthetic_inputs(cfg)
    return cfg, evaluate(internal, external, cfg)


def test_config_loads() -> None:
    cfg = StudyConfig.from_yaml(CONFIG)
    assert cfg.scale.name == "dentex_depth"
    assert sum(cfg.splits.values()) == pytest.approx(1.0)
    assert cfg.operating_point.rationale.strip()


def test_pipeline_composes(small_run) -> None:
    _cfg, r = small_run
    assert r.splits.patient_leakage.is_clean
    assert r.splits.naive_leakage.leaked_fraction > 0
    assert len(r.headline) == 3
    assert all(h.estimate is None or h.estimate.level == pytest.approx(1 - 0.05 / 3)
               for h in r.headline)
    assert {c.reader for c in r.agreement.ceiling} == {"r1", "r2", "r3", "r4"}
    assert r.internal.stratified.strata and r.external.stratified.strata
    # Incomplete reading design: Fleiss is refused and said so, Krippendorff carries it.
    assert r.agreement.fleiss is None and "complete design" in r.agreement.fleiss_reason
    assert any("no age metadata" in n for n in r.not_computable)
    assert r.splits.test_images_with_seen_patient == 0.0
    assert r.internal.selective.patient_overlap == 0
    assert r.agreement.sweep is not None
    # The clean run must not trip the wire: if it does, the generator or the
    # harness has started leaking.
    assert tripwire(r) == []
    assert all("(n=" in x for x in small_cell_flags(r))


def test_report_and_figures(small_run, tmp_path: Path) -> None:
    _, r = small_run
    figures = write_figures(r, tmp_path)
    for rel in figures.values():
        assert (tmp_path / rel).read_text().lstrip().startswith("<?xml")
    md = render_markdown(r, synthetic=True, figures=figures)
    for heading in ("SYNTHETIC DATA", "How to read this report", "Leakage tripwire",
                    "Headline claims", "1. Data and splits", "2. Reader agreement",
                    "3. Depth-stratified", "4. Lesion-level detection", "5. Calibration",
                    "6. Abstention", "7. Subgroups", "8. Not computable"):
        assert heading in md
    assert "uncorrected for multiple comparisons" in md
    assert "not computable on this data" in md
    assert isinstance(tripwire(r), list)


def test_tripwire_flags_high_performance() -> None:
    f = _Formatter()
    high = grouped_bootstrap(lambda idx: 0.97, ["a", "b", "c"], seed=0, n_boot=10)
    low = grouped_bootstrap(lambda idx: 0.80, ["a", "b", "c"], seed=0, n_boot=10)
    assert f.metric(high, "test AUC").endswith("⚠")
    assert not f.metric(low, "test sens").endswith("⚠")
    assert f.metric(None, "missing") == "—"
    assert f.flags == ["test AUC: 0.970 [0.970, 0.970]"]
    assert LEAK_THRESHOLD == 0.95
    # A 3-of-3 cell is still marked and listed, but not as the real alarm.
    assert f.metric(high, "tiny cell sens", n=3).endswith("⚠")
    assert f.small_cell_flags == ["tiny cell sens (n=3): 0.970 [0.970, 0.970]"]
    assert f.metric(high, "big cell sens", n=MIN_CELL).endswith("⚠")
    assert len(f.flags) == 2


def test_synthetic_inputs_need_synthetic_section() -> None:
    cfg = dataclasses.replace(StudyConfig.from_yaml(CONFIG), synthetic=None)
    with pytest.raises(ValueError, match="synthetic section"):
        synthetic_inputs(cfg)


def test_two_reader_complete_design_reports_what_it_cannot_compute(tmp_path: Path) -> None:
    from dcai.data.synthetic import DEFAULT_READERS, make_reader_study
    from dcai.pipeline import DatasetInputs
    from dcai.simulate import EXTERNAL, INTERNAL, simulate_model

    cfg = dataclasses.replace(StudyConfig.from_yaml(CONFIG), n_boot=40)
    sets = []
    for name, profile, seed in (("internal", INTERNAL, 1), ("external", EXTERNAL, 2)):
        study = make_reader_study(n_patients=90, seed=seed, readers=DEFAULT_READERS[:2],
                                  full_read_fraction=1.0, dataset=name)
        preds, dets = simulate_model(study, profile, seed=seed)
        sets.append(DatasetInputs(name, study.records, preds, dets))
    r = evaluate(*sets, cfg)
    assert r.agreement.fleiss is not None  # complete design: Fleiss is computable
    assert r.agreement.ceiling == () and "at least 3 human readers" in r.agreement.ceiling_reason
    md = render_markdown(r, synthetic=True, figures=write_figures(r, tmp_path))
    assert "Ceiling not computable" in md


@pytest.fixture(scope="module")
def leaky_run():
    base = StudyConfig.from_yaml(E0_CONFIG)
    cfg = dataclasses.replace(
        base, n_boot=60,
        synthetic=dataclasses.replace(base.synthetic, internal_patients=150, external_patients=60),
    )
    internal, external = synthetic_inputs(cfg)
    return evaluate(internal, external, cfg)


def test_leaky_e0_fixture_trips_the_wire(leaky_run, tmp_path: Path) -> None:
    """The tripwire's whole purpose is catching leakage, so it is tested on a real leak.

    Naive image-level split, several images per patient, and a model that
    memorises its training patients: the report must raise flags and populate
    the leakage section. If this test ever passes quietly, the tripwire is broken.
    """
    r = leaky_run
    assert r.config.split_kind.value == "image_naive"
    assert not r.splits.patient_leakage.is_clean
    assert r.splits.patient_leakage.leaked_fraction > 0.5
    assert r.splits.test_images_with_seen_patient > 0.8
    assert r.internal.selective.patient_overlap > 0

    flags = tripwire(r)
    assert any(flag.startswith("internal") for flag in flags)
    md = render_markdown(r, synthetic=True, figures=write_figures(r, tmp_path))
    assert "E0: NAIVE IMAGE-LEVEL SPLIT" in md
    assert "THIS SPLIT LEAKS" in md
    assert "No performance estimate exceeds" not in md
    assert md.count("⚠") > len(flags)  # listed at the top and marked inline
    assert any("E0 naive image-level split" in n for n in r.not_computable)


def test_leak_inflates_internal_against_external(leaky_run) -> None:
    internal = leaky_run.internal.stratified.stratum("caries").auc_vs_sound
    external = leaky_run.external.stratified.stratum("caries").auc_vs_sound
    assert internal.lo > external.hi


def test_dentex_shaped_run_reports_what_it_cannot_compute(tmp_path: Path) -> None:
    """One consensus reader, no external set, inventory on only some images."""
    import dataclasses as dc

    from dcai.data.synthetic import ReaderProfile, make_reader_study
    from dcai.pipeline import DatasetInputs
    from dcai.simulate import INTERNAL, simulate_model

    cfg = dataclasses.replace(StudyConfig.from_yaml(CONFIG), n_boot=40)
    study = make_reader_study(n_patients=160, seed=7, full_read_fraction=1.0,
                              readers=(ReaderProfile("consensus", (0.7, 0.95), 0.01, 0.2),))
    preds, dets = simulate_model(study, INTERNAL, seed=7)
    records = [r if i % 3 else dc.replace(r, teeth_present=None, teeth_present_source=None)
               for i, r in enumerate(study.records)]
    r = evaluate(DatasetInputs("internal", records, preds, dets), None, cfg)
    assert r.external is None and r.agreement.krippendorff is None
    assert any("no external dataset" in n for n in r.not_computable)
    assert any("no human tooth inventory" in n for n in r.not_computable)
    assert any("rule 3 has no data on this cohort" in n for n in r.not_computable)
    assert [h.reason for h in r.headline][1:] == ["no external dataset"] * 2
    md = render_markdown(r, synthetic=True, figures=write_figures(r, tmp_path))
    assert "sensitivity (external)" not in md and "one reader only" in md


def test_boxes_can_be_scored_at_the_operating_threshold(tmp_path: Path) -> None:
    raw = CONFIG.read_text().replace("score_threshold: 0.3", "score_threshold: operating_point")
    path = tmp_path / "c.yaml"
    path.write_text(raw)
    cfg = dataclasses.replace(
        StudyConfig.from_yaml(path), n_boot=40,
        synthetic=SyntheticConfig(internal_patients=120, external_patients=50,
                                  external_demographics=False))
    assert cfg.box_score_threshold is None
    r = evaluate(*synthetic_inputs(cfg), cfg)
    assert r.internal.detection[0].report.threshold == r.operating_point.threshold
    assert "(the operating threshold)" in render_markdown(r, synthetic=True)


def test_cohort_note_leads_tooth_level_sections_and_captions(small_run, tmp_path: Path) -> None:
    _, r = small_run
    md = render_markdown(r, synthetic=True, figures=write_figures(r, tmp_path),
                         cohort_note="one acquisition cluster")
    for heading in ("## 3.", "## 4.", "## 5.", "## 6."):
        section = md[md.index(heading):]
        first_paragraph = section.split("\n\n")[1]
        assert first_paragraph == "**Cohort:** one acquisition cluster"
    assert "*Sensitivity by lesion depth. Cohort: one acquisition cluster*" in md
    assert "*Reliability diagram. Cohort: one acquisition cluster*" in md
    assert md.count("Cohort:") == 6
