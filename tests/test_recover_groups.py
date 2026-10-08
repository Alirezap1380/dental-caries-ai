from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import adjusted_rand_score

from dcai.data.schema import Modality, PatientIdSource, RadiographRecord
from dcai.data.synthetic_images import SITES, SyntheticImage, make_image_study
from dcai.probes.recover_groups import (
    cluster_sites,
    fingerprint,
    fingerprint_features,
    intensity_thumbnail,
    load_gray,
    phash,
    pixel_embedding,
    prevalence_by_cluster,
    recover_patients,
    recovery_markdown,
    run_recovery,
    split_sensitivity,
    with_recovered_patients,
)
from dcai.probes.source_probe import site_probe

N_BOOT = 100


def as_shipped(images: list[SyntheticImage]) -> list[RadiographRecord]:
    """Records the way a release without patient ids arrives: one 'patient' per image."""
    return [
        RadiographRecord(
            image_id=i.image_id, path=str(i.path), patient_id=i.image_id,
            patient_id_source=PatientIdSource.ASSUMED_UNIQUE, dataset="dentex_like",
            modality=Modality.PANORAMIC, readers=frozenset({"consensus"}),
        )
        for i in images
    ]


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> list[SyntheticImage]:
    return make_image_study(tmp_path_factory.mktemp("study"), n_patients=90, seed=3,
                            repeat_fraction=0.35)


@pytest.fixture(scope="module")
def arrays(study: list[SyntheticImage]):
    imgs = [load_gray(i.path) for i in study]
    return (np.array([phash(x) for x in imgs]), np.array([pixel_embedding(x) for x in imgs]),
            np.array([intensity_thumbnail(x) for x in imgs]))


def true_pairs(study: list[SyntheticImage]) -> set[frozenset[str]]:
    return {frozenset((a.image_id, b.image_id)) for a in study for b in study
            if a.image_id < b.image_id and a.true_patient == b.true_patient}


# --- I/O and hashing ----------------------------------------------------------------


def test_load_gray_handles_bit_depths(tmp_path: Path) -> None:
    Image.fromarray(np.full((4, 4), 255, np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.full((4, 4), 65535, np.uint16)).save(tmp_path / "b.png")
    Image.fromarray(np.full((4, 4, 3), 51, np.uint8)).save(tmp_path / "c.png")
    assert load_gray(tmp_path / "a.png").max() == 1.0
    assert load_gray(tmp_path / "b.png").max() == pytest.approx(1.0)
    assert load_gray(tmp_path / "c.png").max() == pytest.approx(0.2)


def test_phash_separates_same_from_different(study: list[SyntheticImage], arrays) -> None:
    hashes = arrays[0]
    idx = {i.image_id: k for k, i in enumerate(study)}
    same = [int((hashes[idx[a]] != hashes[idx[b]]).sum()) for a, b in map(tuple, true_pairs(study))]
    rng = np.random.default_rng(0)
    diff = []
    while len(diff) < 200:
        a, b = rng.choice(len(study), 2, replace=False)
        if study[a].true_patient != study[b].true_patient:
            diff.append(int((hashes[a] != hashes[b]).sum()))
    assert np.median(same) < np.median(diff)


def test_pixel_embedding_is_exposure_invariant_but_thumbnail_is_not() -> None:
    img = np.random.default_rng(0).random((64, 128))
    a, b = pixel_embedding(img), pixel_embedding(img * 0.6)
    assert np.allclose(a, b, atol=1e-5)  # float32 resize
    assert not np.allclose(intensity_thumbnail(img), intensity_thumbnail(img * 0.6))
    assert np.all(pixel_embedding(np.zeros((64, 128))) == 0)


# --- patients -----------------------------------------------------------------------


def test_recovers_repeat_visits(study: list[SyntheticImage], arrays) -> None:
    hashes, emb, _ = arrays
    rec = recover_patients([i.image_id for i in study], hashes, emb, seed=0)
    assert rec.mixture.separated and rec.threshold is not None
    found = {frozenset((p.a, p.b)) for p in rec.pairs}
    truth = true_pairs(study)
    precision = len(found & truth) / len(found)
    recall = len(found & truth) / len(truth)
    assert precision > 0.85 and recall > 0.85
    assert sum(size * n for size, n in rec.group_sizes.items()) == len(study)
    assert rec.n_merged_images >= 2 * len(found & truth) - 2


def test_no_repeats_means_no_pairs_declared(tmp_path: Path) -> None:
    # Without repeat visits there is no high-similarity mode, and the probe must
    # say so rather than invent a cutoff.
    study = make_image_study(tmp_path, n_patients=60, seed=4, repeat_fraction=0.0)
    imgs = [load_gray(i.path) for i in study]
    rec = recover_patients([i.image_id for i in study], np.array([phash(x) for x in imgs]),
                           np.array([pixel_embedding(x) for x in imgs]), seed=0)
    assert rec.threshold is None
    assert not [p for p in rec.pairs if p not in rec.duplicates]


def test_exact_duplicate_found_by_hash(study: list[SyntheticImage], tmp_path: Path) -> None:
    src = study[0].path
    dup = tmp_path / f"copy{src.suffix}"
    shutil.copy(src, dup)
    others = [i.path for i in study[1:10]]
    paths = [src, dup, *others]
    imgs = [load_gray(p) for p in paths]
    rec = recover_patients([p.name for p in paths], np.array([phash(x) for x in imgs]),
                           np.array([pixel_embedding(x) for x in imgs]), seed=0)
    assert any({d.a, d.b} == {src.name, dup.name} and d.hamming == 0 for d in rec.duplicates)


def test_recover_patients_validation() -> None:
    with pytest.raises(ValueError, match=">= 3 images"):
        recover_patients(["a", "b"], np.zeros((2, 64), bool), np.zeros((2, 4)), seed=0)


def test_recovered_ids_and_split_sensitivity(study: list[SyntheticImage], arrays) -> None:
    hashes, emb, _ = arrays
    rec = recover_patients([i.image_id for i in study], hashes, emb, seed=0)
    shipped = as_shipped(study)
    recovered = with_recovered_patients(shipped, rec)
    merged = [r for r in recovered if r.patient_id_source is PatientIdSource.DERIVED]
    assert merged and all(r.patient_id.startswith("recovered-") for r in merged)
    assert len(merged) == rec.n_merged_images
    singles = [r for r in recovered if r.patient_id_source is PatientIdSource.ASSUMED_UNIQUE]
    assert all(r.patient_id == r.image_id for r in singles)

    sens = split_sensitivity(shipped, rec, seed=0)
    # The shipped-id split leaks recovered patients; its own check could not see it.
    assert sens.groups_straddling > 0
    assert 0 < sens.test_images_with_partner_in_train <= sens.n_test_images
    assert sens.n_recovered_pairs == len(rec.pairs)
    # Splitting on the recovered ids closes that leak.
    clean = split_sensitivity(recovered, rec, seed=0)
    assert clean.groups_straddling == 0 and clean.test_images_with_partner_in_train == 0


# --- sites --------------------------------------------------------------------------


def test_fingerprint_reads_acquisition(study: list[SyntheticImage]) -> None:
    by_site = {i.true_site: fingerprint(i.image_id, i.path) for i in study}
    assert by_site["site_c"].bit_depth == 16 and by_site["site_a"].bit_depth == 8
    assert by_site["site_b"].file_format == "JPEG" and by_site["site_b"].quant_tables
    assert by_site["site_a"].quant_tables == ""
    assert (by_site["site_a"].height, by_site["site_a"].width) == SITES[0].size
    for f in by_site.values():
        assert f.histogram.sum() == pytest.approx(1.0)
        assert f.noise_spectrum.shape == (16,)


def test_sites_recovered_cleanly(study: list[SyntheticImage]) -> None:
    fps = [fingerprint(i.image_id, i.path) for i in study]
    c = cluster_sites(fingerprint_features(fps), seed=0)
    assert c.clean and c.best_k == 3
    assert adjusted_rand_score([i.true_site for i in study], c.labels) == pytest.approx(1.0)
    assert sum(c.sizes) == len(study)


def test_single_site_is_not_clean(tmp_path: Path) -> None:
    study = make_image_study(tmp_path, n_patients=24, seed=5, sites=SITES[:1])
    feats = fingerprint_features([fingerprint(i.image_id, i.path) for i in study])
    assert not cluster_sites(feats, seed=0).clean


def test_near_constant_columns_are_dropped_not_amplified(study: list[SyntheticImage]) -> None:
    # Regression: identical values have a float std of ~1e-17, which once
    # passed `std > 0` and was blown up into unit-variance noise.
    fps = [fingerprint(i.image_id, i.path) for i in study if i.true_site == "site_a"][:5]
    feats = fingerprint_features(fps)
    assert feats.shape[1] < 3 + 1 + 32 + 16  # size, bit depth, format, qtable blocks dropped
    assert np.isfinite(feats).all()


def test_constant_fingerprints_are_refused(study: list[SyntheticImage]) -> None:
    fp = fingerprint(study[0].image_id, study[0].path)
    with pytest.raises(ValueError, match="constant"):
        fingerprint_features([fp, fp, fp])
    with pytest.raises(ValueError, match="not enough"):
        cluster_sites(np.zeros((2, 3)), seed=0)


def test_prevalence_difference_detected(study: list[SyntheticImage]) -> None:
    sites = np.array([s.name for s in SITES])
    clusters = np.searchsorted(sites, [i.true_site for i in study])
    res = prevalence_by_cluster([i.has_lesion for i in study], clusters,
                                [i.true_patient for i in study], seed=0, n_boot=N_BOOT)
    assert res.p_value < 0.05
    assert res.prevalence["site_0"].value < res.prevalence["site_2"].value


# --- the probe ----------------------------------------------------------------------


def test_probe_answer_depends_on_the_feature_space(study: list[SyntheticImage], arrays) -> None:
    _, standardised, raw = arrays
    sites = [i.true_site for i in study]
    groups = [i.true_patient for i in study]
    sees_exposure = site_probe(raw, sites, groups, seed=0, n_boot=N_BOOT)
    blind = site_probe(standardised, sites, groups, seed=0, n_boot=N_BOOT)
    assert sees_exposure.auc.lo > 0.8
    assert blind.auc.value < sees_exposure.auc.value
    assert sees_exposure.classes == ("site_a", "site_b", "site_c")
    assert set(sees_exposure.per_class_auc) == set(sees_exposure.classes)


def test_probe_refuses_circular_features(study: list[SyntheticImage]) -> None:
    feats = fingerprint_features([fingerprint(i.image_id, i.path) for i in study])
    with pytest.raises(ValueError, match="circular"):
        site_probe(feats, [i.true_site for i in study], [i.true_patient for i in study],
                   seed=0, forbidden_features=feats)


def test_probe_binary_and_validation() -> None:
    rng = np.random.default_rng(0)
    y = np.array(["a", "b"] * 30)
    x = rng.normal(size=(60, 3)) + (y == "b")[:, None] * 2.0
    groups = [f"p{i}" for i in range(60)]
    r = site_probe(x, y, groups, seed=0, n_boot=N_BOOT)
    assert r.auc.value > 0.9 and r.classes == ("a", "b")
    with pytest.raises(ValueError, match="two classes"):
        site_probe(x, ["a"] * 60, groups, seed=0)
    with pytest.raises(ValueError, match="one label and group"):
        site_probe(x, y[:10], groups, seed=0)


# --- end to end ---------------------------------------------------------------------


def test_run_recovery_and_report(study: list[SyntheticImage]) -> None:
    shipped = as_shipped(study)
    r = run_recovery(shipped, {i.image_id: i.path for i in study},
                     {i.image_id: i.has_lesion for i in study}, seed=0, n_boot=N_BOOT)
    assert r.shipped_ids_per_patient == 1.0
    assert r.sites.clustering.clean and r.sites.shortcut_risk
    md = recovery_markdown(r, synthetic=True)
    for text in ("SYNTHETIC IMAGES", "no usable patient ids", "straddle partitions",
                 "Clean", "Shortcut risk", "raw-intensity thumbnails"):
        assert text in md


def test_macro_auc_undefined_on_degenerate_resamples() -> None:
    from dcai.probes.source_probe import _macro_auc

    proba = np.full((4, 3), 1 / 3)
    assert np.isnan(_macro_auc(np.array([0, 0, 0, 0]), proba, 3))  # one class present
    assert np.isnan(_macro_auc(np.array([0, 1, 0, 1]), proba, 3))  # a class missing
    assert _macro_auc(np.array([0, 1, 2, 0]), proba, 3) == 0.5


def test_p_value_formatting() -> None:
    from dcai.probes.recover_groups import _p

    assert _p(1e-9) == "< 0.0001" and _p(0.0123) == "= 0.0123"
