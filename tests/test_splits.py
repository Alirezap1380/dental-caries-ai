from __future__ import annotations

import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import pytest
from conftest import record

from dcai.data.schema import CariesStage, Finding, PatientIdSource, RadiographRecord
from dcai.data.splits import (
    LeakageReport,
    NoLeakageToMeasureWarning,
    PatientLeakageError,
    Split,
    SplitKind,
    check_leakage,
    leave_one_site_out,
    naive_image_split,
    patient_kfold,
    patient_split,
)
from dcai.data.synthetic import make_records

SRC = Path(__file__).resolve().parents[1] / "src"


def groups_of(recs: tuple[RadiographRecord, ...]) -> set[str]:
    return {r.group_key for r in recs}


# --- check_leakage --------------------------------------------------------------


def test_leak_is_detected_and_names_the_patient() -> None:
    parts = {"train": [record("a", "p1"), record("b", "p2")], "test": [record("c", "p1")]}
    with pytest.raises(PatientLeakageError, match="ds/p1"):
        check_leakage(parts)


def test_leakage_error_is_an_assertion_error() -> None:
    assert issubclass(PatientLeakageError, AssertionError)


def test_leak_reported_without_raising_when_asked() -> None:
    parts = {"train": [record("a", "p1"), record("b", "p2")], "test": [record("c", "p1")]}
    rep = check_leakage(parts, raise_on_leak=False)
    assert rep.leaked_groups == {"ds/p1"}
    assert rep.n_groups == 2
    assert rep.leaked_fraction == 0.5
    assert not rep.is_clean


def test_duplicate_image_always_raises() -> None:
    # Even in measuring mode: the same image twice is a bug, not the E0 leak.
    parts = {"train": [record("a", "p1")], "test": [record("a", "p1")]}
    with pytest.raises(PatientLeakageError, match="image 'a'"):
        check_leakage(parts, raise_on_leak=False)


def test_same_patient_id_in_different_datasets_is_not_leakage() -> None:
    parts = {"train": [record("a", "p1", dataset="dentex")],
             "test": [record("b", "p1", dataset="tufts")]}
    assert check_leakage(parts).is_clean


def test_assumed_ids_are_counted_as_unverifiable() -> None:
    assumed = PatientIdSource.ASSUMED_UNIQUE
    parts = {"train": [record("a", "a", patient_id_source=assumed)],
             "test": [record("b", "b", patient_id_source=assumed), record("c", "p3")]}
    assert check_leakage(parts).unverifiable_groups == 2


def test_empty_report() -> None:
    assert LeakageReport(0, frozenset(), 0).leaked_fraction == 0.0


def test_leak_check_survives_python_O() -> None:
    # `python -O` strips assert statements. The check must use explicit raises.
    code = (
        "from dcai.data.splits import Split, SplitKind, PatientLeakageError\n"
        "from dcai.data.schema import RadiographRecord as R\n"
        "mk = lambda i, p: R(image_id=i, path=i, patient_id=p, patient_id_source='provided',"
        " dataset='d', modality='bitewing', readers={'c'})\n"
        "try:\n"
        "    Split({'train': [mk('a', 'p')], 'test': [mk('b', 'p')]}, SplitKind.PATIENT)\n"
        "except PatientLeakageError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], env={"PYTHONPATH": str(SRC)}, capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


# --- Split ----------------------------------------------------------------------


def test_split_constructor_runs_the_check() -> None:
    with pytest.raises(PatientLeakageError):
        Split({"train": [record("a", "p1")], "test": [record("b", "p1")]}, SplitKind.PATIENT)


@pytest.mark.parametrize("kind", [SplitKind.PATIENT, SplitKind.SITE_HOLDOUT])
def test_allow_leakage_reserved_for_naive_split(kind: SplitKind) -> None:
    with pytest.raises(ValueError, match="reserved"):
        Split({"train": [record("a", "p1")], "test": [record("b", "p2")]}, kind,
              allow_leakage=True)


def test_split_rejects_empty_partition() -> None:
    with pytest.raises(ValueError, match="empty partition"):
        Split({"train": [record("a", "p1")], "test": []}, SplitKind.PATIENT)


def test_split_accessors() -> None:
    s = Split({"train": [record("a", "p1")], "test": [record("b", "p2")]}, SplitKind.PATIENT)
    assert s.image_ids("test") == ["b"]
    assert isinstance(s["train"], tuple)


# --- patient_split --------------------------------------------------------------


def test_patient_split_keeps_bitewing_sets_together(bitewing_sets: list) -> None:
    s = patient_split(bitewing_sets, seed=0)
    assert s.leakage.is_clean
    for part in s.partitions.values():
        per_patient = Counter(r.group_key for r in part)
        assert set(per_patient.values()) == {4}


def test_patient_split_covers_every_record_once(mixed_records: list) -> None:
    s = patient_split(mixed_records, seed=0)
    ids = [i for name in s.partitions for i in s.image_ids(name)]
    assert sorted(ids) == sorted(r.image_id for r in mixed_records)


def test_patient_split_proportions_by_image(mixed_records: list) -> None:
    s = patient_split(mixed_records, seed=0)
    n = len(mixed_records)
    for name, target in {"train": 0.7, "val": 0.15, "test": 0.15}.items():
        assert len(s[name]) / n == pytest.approx(target, abs=0.05)


@pytest.mark.parametrize("seed", range(25))
def test_patient_split_clean_across_seeds(seed: int, mixed_records: list) -> None:
    s = patient_split(mixed_records, seed=seed)
    assert s.leakage.is_clean
    names = list(s.partitions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert groups_of(s[a]).isdisjoint(groups_of(s[b]))


def test_patient_split_deterministic_and_order_independent(mixed_records: list) -> None:
    a = patient_split(mixed_records, seed=7)
    b = patient_split(list(reversed(mixed_records)), seed=7)
    c = patient_split(mixed_records, seed=8)
    assert a.image_ids("test") == b.image_ids("test")
    assert a.image_ids("test") != c.image_ids("test")


def test_patient_split_custom_fractions(mixed_records: list) -> None:
    s = patient_split(mixed_records, seed=0, fractions={"train": 0.5, "test": 0.5})
    assert list(s.partitions) == ["train", "test"]


@pytest.mark.parametrize(
    "fractions, match",
    [
        ({"train": 1.0}, "at least two"),
        ({"train": 0.8, "test": 0.1}, "sum to 1"),
        ({"train": 1.1, "test": -0.1}, "positive"),
    ],
)
def test_patient_split_validates_fractions(fractions: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        patient_split(make_records(n_patients=10), seed=0, fractions=fractions)


def test_patient_split_too_few_patients_for_partitions() -> None:
    with pytest.raises(ValueError, match="empty partition"):
        patient_split(make_records(n_patients=2, images_per_patient=1), seed=0)


def test_split_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no records"):
        patient_split([], seed=0)


def test_split_rejects_duplicate_image_ids() -> None:
    with pytest.raises(ValueError, match="duplicate image_id"):
        patient_split([record("a", "p1"), record("a", "p2")], seed=0)


def test_patient_split_stratified_by_site(mixed_records: list) -> None:
    s = patient_split(mixed_records, seed=0, stratify=lambda recs: recs[0].site_id)
    assert s.leakage.is_clean
    for name in ("train", "val", "test"):
        sites = Counter(r.site_id for r in s[name])
        assert set(sites) == {"site_a", "site_b", "site_c"}


def test_stratification_balances_a_rare_label() -> None:
    recs = make_records(n_patients=200, images_per_patient=(1, 4), caries_rate=0.15, seed=5)

    def has_initial_lesion(patient: list[RadiographRecord]) -> bool:
        return any(
            a.finding is Finding.CARIES and a.stage is not None and a.stage.is_initial
            for r in patient for a in r.annotations
        )

    s = patient_split(recs, seed=0, stratify=has_initial_lesion)
    positive = {r.group_key for r in recs if has_initial_lesion([r])}
    overall = len(positive) / len({r.group_key for r in recs})
    for name in ("train", "val", "test"):
        groups = groups_of(s[name])
        assert len(groups & positive) / len(groups) == pytest.approx(overall, abs=0.08)


# --- patient_kfold --------------------------------------------------------------


@pytest.mark.parametrize("n_splits", [2, 5, 10])
def test_kfold_test_sets_partition_the_data(n_splits: int, mixed_records: list) -> None:
    folds = patient_kfold(mixed_records, seed=0, n_splits=n_splits)
    assert len(folds) == n_splits
    tested = [i for f in folds for i in f.image_ids("test")]
    assert sorted(tested) == sorted(r.image_id for r in mixed_records)
    for f in folds:
        assert f.leakage.is_clean
        assert groups_of(f["train"]).isdisjoint(groups_of(f["test"]))
        assert len(f["train"]) + len(f["test"]) == len(mixed_records)


def test_kfold_fold_sizes_are_balanced(mixed_records: list) -> None:
    sizes = [len(f["test"]) for f in patient_kfold(mixed_records, seed=0, n_splits=5)]
    assert max(sizes) - min(sizes) <= 4  # at most one patient's worth of images


def test_kfold_stratified(mixed_records: list) -> None:
    folds = patient_kfold(mixed_records, seed=0, n_splits=3,
                          stratify=lambda recs: recs[0].site_id)
    for f in folds:
        assert {r.site_id for r in f["test"]} == {"site_a", "site_b", "site_c"}


def test_kfold_deterministic(mixed_records: list) -> None:
    a = patient_kfold(mixed_records, seed=3)
    b = patient_kfold(list(reversed(mixed_records)), seed=3)
    assert [f.image_ids("test") for f in a] == [f.image_ids("test") for f in b]


def test_kfold_detects_incomplete_coverage(
    mixed_records: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guards against a CV splitter (sklearn change, or a future swap) that
    # silently drops records from every test fold.
    import dcai.data.splits as splits_mod

    class DropsLastRecord:
        def __init__(self, **_: object) -> None: ...

        def split(self, X, y, groups):
            kept = sorted(set(groups) - {groups[-1]})
            a = set(kept[: len(kept) // 2])
            first = [i for i, g in enumerate(groups) if g in a]
            second = [i for i, g in enumerate(groups) if g in set(kept) - a]
            yield second, first
            yield first, second

    monkeypatch.setattr(splits_mod, "StratifiedGroupKFold", DropsLastRecord)
    with pytest.raises(PatientLeakageError, match="cover"):
        patient_kfold(mixed_records, seed=0, n_splits=2)


@pytest.mark.parametrize("n_splits", [1, 11])
def test_kfold_validates_n_splits(n_splits: int) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        patient_kfold(make_records(n_patients=10), seed=0, n_splits=n_splits)


# --- leave_one_site_out ---------------------------------------------------------


def test_leave_one_site_out(mixed_records: list) -> None:
    folds = leave_one_site_out(mixed_records)
    assert [f.name for f in folds] == [f"holdout={s}" for s in ("site_a", "site_b", "site_c")]
    for f, site in zip(folds, ("site_a", "site_b", "site_c")):
        assert f.kind is SplitKind.SITE_HOLDOUT
        assert {r.site_id for r in f["test"]} == {site}
        assert site not in {r.site_id for r in f["train"]}


def test_patient_seen_at_two_sites_is_leakage() -> None:
    recs = [record("a", "p1", site_id="A"), record("b", "p1", site_id="B"),
            record("c", "p2", site_id="A")]
    with pytest.raises(PatientLeakageError, match="ds/p1"):
        leave_one_site_out(recs)


def test_site_holdout_requires_site_ids() -> None:
    with pytest.raises(ValueError, match="no site_id"):
        leave_one_site_out([record("a", "p1", site_id="A"), record("b", "p2")])


def test_site_holdout_requires_two_sites() -> None:
    with pytest.raises(ValueError, match="two sites"):
        leave_one_site_out([record("a", "p1", site_id="A"), record("b", "p2", site_id="A")])


# --- naive_image_split ----------------------------------------------------------


def test_naive_split_leaks_and_measures_it(bitewing_sets: list) -> None:
    s = naive_image_split(bitewing_sets, seed=0)
    assert s.kind is SplitKind.IMAGE_NAIVE
    # With four images per patient across a 70/15/15 split, most patients leak.
    assert s.leakage.leaked_fraction > 0.5
    ids = [i for name in s.partitions for i in s.image_ids(name)]
    assert sorted(ids) == sorted(r.image_id for r in bitewing_sets)


def test_naive_split_proportions(bitewing_sets: list) -> None:
    s = naive_image_split(bitewing_sets, seed=0)
    assert len(s["test"]) == pytest.approx(0.15 * len(bitewing_sets), abs=1)


def test_naive_split_warns_when_there_is_nothing_to_leak() -> None:
    # One image per patient (or assumed-unique ids): E0 and E1 splits coincide.
    recs = make_records(n_patients=40, images_per_patient=1,
                        patient_id_source=PatientIdSource.ASSUMED_UNIQUE)
    with pytest.warns(NoLeakageToMeasureWarning):
        s = naive_image_split(recs, seed=0)
    assert s.leakage.is_clean
    assert s.leakage.unverifiable_groups == 40


def test_naive_split_no_warning_when_leaking(bitewing_sets: list) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoLeakageToMeasureWarning)
        naive_image_split(bitewing_sets, seed=0)


# --- synthetic fixture sanity -----------------------------------------------------


def test_synthetic_records_have_the_structure_tests_rely_on() -> None:
    recs = make_records(n_patients=30, images_per_patient=(1, 4), readers=("r1", "r2"),
                        caries_rate=0.6, seed=9)
    per_patient = Counter(r.group_key for r in recs)
    assert max(per_patient.values()) > 1
    # Each patient is imaged at exactly one site.
    sites_per_patient = {g: {r.site_id for r in recs if r.group_key == g} for g in per_patient}
    assert all(len(s) == 1 for s in sites_per_patient.values())
    stages = {a.stage for r in recs for a in r.annotations}
    assert None in stages and stages - {None} <= set(CariesStage) - {CariesStage.NONE}
    assert all(r.is_multi_annotator for r in recs)
