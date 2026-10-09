"""Shared DENTEX experiment setup: load, recover patients, run the integrity gates.

Both experiment scripts (stage 2 alone, and stage 1) start here, so they see the
same records, the same recovered patients and therefore the same split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu

from dcai.data.dentex import DentexRelease, extract_images, load_dentex
from dcai.data.near_duplicates import Signature, near_duplicates, signatures_for
from dcai.data.schema import Finding, RadiographRecord
from dcai.eval.bootstrap import difference, grouped_bootstrap
from dcai.probes.recover_groups import (
    PatientRecovery,
    cluster_sites,
    fingerprint,
    fingerprint_features,
    recover_patients,
    with_recovered_patients,
)

DENTEX_FINDINGS = (Finding.CARIES, Finding.DEEP_CARIES, Finding.PERIAPICAL_LESION,
                   Finding.IMPACTED_TOOTH)


def duplicate_check(records: Sequence[RadiographRecord], sigs: dict[str, Signature],
                    recovery: PatientRecovery) -> list[str]:
    """Every near-duplicate pair inside the evaluation set must share a recovered group.

    A pair split across groups could land on both sides of our own patient split
    and leak straight into stage 2, so it stops the run.
    """
    image_of = {r.meta["sha256"]: r.image_id for r in records}
    pairs = near_duplicates([sigs[h] for h in image_of])
    crossing = [p for p in pairs
                if recovery.group_of[image_of[p.a]] != recovery.group_of[image_of[p.b]]]
    if crossing:
        raise PatientRecoveryError(
            f"patient recovery missed {len(crossing)} near-duplicate pair(s) inside the "
            "evaluation set: " + "; ".join(
                f"{image_of[p.a]} ~ {image_of[p.b]} (Hamming {p.hamming}, r = {p.corr:.3f})"
                for p in crossing[:5]))
    return [f"{image_of[p.a]} ~ {image_of[p.b]} (Hamming {p.hamming}, r = {p.corr:.3f})"
            for p in pairs]


def selection_check(records: Sequence[RadiographRecord], inventoried: set[str], *,
                    seed: int, n_boot: int) -> tuple[str, bool]:
    """Are the inventoried images a random subset of the evaluation set?"""
    inv = [r for r in records if r.image_id in inventoried]
    rest = [r for r in records if r.image_id not in inventoried]
    rows, differs = [], False

    for name in ("width", "height"):
        a, b = [getattr(r, name) for r in inv], [getattr(r, name) for r in rest]
        p = float(mannwhitneyu(a, b).pvalue)
        differs |= p < 0.05
        rows.append(f"| image {name}, median px | {np.median(a):.0f} | {np.median(b):.0f} | "
                    f"Mann-Whitney p = {p:.3g} |")

    clusters = cluster_sites(
        fingerprint_features([fingerprint(r.image_id, Path(r.path)) for r in records]), seed=seed)
    in_inv = np.array([r.image_id in inventoried for r in records])
    tab = np.array([[np.sum((clusters.labels == k) & in_inv), np.sum((clusters.labels == k) & ~in_inv)]
                    for k in range(clusters.best_k)])
    p = float(chi2_contingency(tab)[1])
    differs |= p < 0.05
    rows.append(f"| recovered site cluster (k = {clusters.best_k}, silhouette "
                f"{clusters.silhouette[clusters.best_k]:.2f}), images per cluster | "
                f"{tab[:, 0].tolist()} | {tab[:, 1].tolist()} | χ² p = {p:.3g} |")

    for k, finding in enumerate(DENTEX_FINDINGS):
        ests = []
        for j, group in enumerate((inv, rest)):
            y = np.array([any(a.finding is finding for a in r.annotations) for r in group], float)
            ests.append(grouped_bootstrap(lambda i, y=y: float(y[i].mean()),
                                          [r.group_key for r in group],
                                          seed=seed + 2 * k + j, n_boot=n_boot))
        d = difference(ests[0], ests[1], paired=False)
        differs |= d.lo > 0 or d.hi < 0
        rows.append(f"| share of images with {finding.value} | {ests[0]} | {ests[1]} | "
                    f"difference {d} |")
    header = (f"| | inventoried ({len(inv)}) | not inventoried ({len(rest)}) | test |\n"
              "|---|---|---|---|")
    return header + "\n" + "\n".join(rows), differs



class PatientRecoveryError(RuntimeError):
    """Patient recovery left a near-duplicate pair in different groups."""


def resnet_image_embeddings(records: Sequence[RadiographRecord], cache: Path) -> np.ndarray:
    """ImageNet ResNet-50 global features of each whole panoramic (cached by image id)."""
    from dcai.models.encoders import FrozenResNet50
    from dcai.probes.recover_groups import _resize, load_gray

    ids = [r.image_id for r in records]
    if cache.exists():
        data = np.load(cache)
        if list(data["ids"]) == ids:
            return data["emb"]
    encoder = FrozenResNet50()
    emb = np.array([encoder(_resize(load_gray(Path(r.path)), (448, 960))[None].astype(np.float32))[0]
                    for r in records])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), emb=emb)
    return emb


@dataclass(frozen=True)
class Prepared:
    release: DentexRelease
    records: list[RadiographRecord]  # all 755, patient ids from recovery
    recovery: PatientRecovery
    # Second opinion from a learned representation. If it finds pairs the pixel
    # recovery missed, `prepare` stops: the split would be wrong.
    resnet_recovery: PatientRecovery
    near_pairs: list[str]
    selection_table: str
    selection_effect: bool


def prepare(raw: Path, work: Path, *, seed: int, n_boot: int,
            extract: tuple[str, ...] = ("diagnosis", "validation")) -> Prepared:
    cache, store = work / "cache", work / "images"
    rel = load_dentex(raw, store=store, index_cache=cache / "sha256_index.json")
    extract_images(rel, raw, store, subsets=extract)
    evals = {r.meta["sha256"] for r in rel.records}
    sigs = signatures_for([(str(raw / e.zip_name), e.member, e.sha256)
                           for e in rel.index if e.sha256 in evals],
                          cache=cache / "pixel_signatures.json")
    shas = [r.meta["sha256"] for r in rel.records]
    recovery = recover_patients([r.image_id for r in rel.records],
                                np.array([sigs[h].phash for h in shas]),
                                np.array([sigs[h].thumb for h in shas]), seed=seed)
    resnet = recover_patients([r.image_id for r in rel.records],
                              np.zeros((len(shas), 64), dtype=bool),  # embedding evidence only
                              resnet_image_embeddings(rel.records,
                                                      cache / "resnet50_image_embeddings.npz"),
                              seed=seed, max_hamming=-1)
    missed = [p for p in resnet.pairs if recovery.group_of[p.a] != recovery.group_of[p.b]]
    if missed:
        raise PatientRecoveryError(
            f"ResNet-50 embeddings declare {len(missed)} same-patient pair(s) the pixel "
            "recovery did not; merge them before splitting")
    records = with_recovered_patients(list(rel.records), recovery)
    near_pairs = duplicate_check(records, sigs, recovery)
    table, selected = selection_check(
        records, {r.image_id for r in records if r.teeth_present is not None},
        seed=seed, n_boot=n_boot)
    return Prepared(rel, records, recovery, resnet, near_pairs, table, selected)
