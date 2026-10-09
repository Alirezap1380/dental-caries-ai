"""Stage 2 on human tooth boxes, through the full evaluation pipeline.

    python scripts/run_dentex.py --config configs/dentex_stage2_frozen.yaml \
        --data data/dentex/raw --out results/dentex_stage2_frozen

1. Load the release by content; recover patient groups from pixel signatures.
2. Integrity gates: every near-duplicate pair inside the evaluation set must
   share a recovered patient group (else stop); test whether the inventoried
   images are a random subset of the evaluation set.
3. Keep the images with a human tooth inventory; split them by recovered patient.
4. Frozen ImageNet ResNet-50 features for every inventoried tooth.
5. Fit the head on train-partition teeth only; predict every tooth.
6. Evaluate with the existing pipeline and write the report.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu

from dcai.data.dentex import extract_images, load_dentex
from dcai.data.near_duplicates import Signature, near_duplicates, signatures_for
from dcai.data.schema import CONSENSUS, Finding, RadiographRecord
from dcai.eval.bootstrap import difference, grouped_bootstrap
from dcai.eval.detection import Detection
from dcai.eval.ratings import tooth_grades
from dcai.eval.units import ToothPrediction
from dcai.figures import write_figures
from dcai.models.encoders import FrozenResNet50, tooth_features
from dcai.models.tooth_head import fit_head
from dcai.pipeline import DatasetInputs, StudyConfig, evaluate, make_split
from dcai.probes.recover_groups import (
    PatientRecovery,
    cluster_sites,
    fingerprint,
    fingerprint_features,
    load_gray,
    recover_patients,
    with_recovered_patients,
)
from dcai.report import render_markdown, small_cell_flags, tripwire

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
        raise SystemExit(
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("data/dentex"))
    args = ap.parse_args(argv)
    cfg = StudyConfig.from_yaml(args.config)
    cache, store = args.work / "cache", args.work / "images"
    t0 = time.time()

    rel = load_dentex(args.data, store=store, index_cache=cache / "sha256_index.json")
    extract_images(rel, args.data, store, subsets=("diagnosis", "validation"))

    # Patient groups from the same pixel signatures the audit uses (cached).
    evals = {r.meta["sha256"] for r in rel.records}
    sigs = signatures_for([(str(args.data / e.zip_name), e.member, e.sha256)
                           for e in rel.index if e.sha256 in evals],
                          cache=cache / "pixel_signatures.json")
    shas = [r.meta["sha256"] for r in rel.records]
    recovery = recover_patients([r.image_id for r in rel.records],
                                np.array([sigs[h].phash for h in shas]),
                                np.array([sigs[h].thumb for h in shas]), seed=cfg.seed)
    records = with_recovered_patients(list(rel.records), recovery)
    near_pairs = duplicate_check(records, sigs, recovery)
    inventoried = [r for r in records if r.teeth_present is not None]
    selection, selected = selection_check(records, {r.image_id for r in inventoried},
                                          seed=cfg.seed, n_boot=cfg.n_boot)
    print(f"integrity checks passed ({time.time() - t0:.0f}s); selection effect: {selected}")

    split = make_split(inventoried, cfg)
    train_ids = {r.image_id for r in split["train"]}
    group_of = {r.image_id: r.group_key for r in inventoried}
    keys, x = tooth_features(
        [(r.image_id, (lambda p=r.path: load_gray(Path(p))), rel.tooth_boxes[r.image_id])
         for r in inventoried],
        FrozenResNet50(),
    )
    print(f"features {x.shape} ({time.time() - t0:.0f}s)")
    grades = {r.image_id: tooth_grades(r, CONSENSUS, cfg.scale) for r in inventoried}
    y = np.array([grades[k.image_id][k.tooth_fdi] for k in keys])
    tr = np.array([k.image_id in train_ids for k in keys])
    head = fit_head(x[tr], y[tr], [group_of[k.image_id] for k in keys if k.image_id in train_ids],
                    n_classes=cfg.scale.n_categories, seed=cfg.seed)
    proba = head.predict_proba(x)

    preds = [ToothPrediction(k.image_id, k.tooth_fdi, tuple(map(float, p)))
             for k, p in zip(keys, proba, strict=True)]
    dets = [Detection(k.image_id, rel.tooth_boxes[k.image_id][k.tooth_fdi], float(1 - p[0]),
                      Finding.DEEP_CARIES if p[2] > p[1] else Finding.CARIES)
            for k, p in zip(keys, proba, strict=True)]
    result = evaluate(DatasetInputs("dentex", inventoried, preds, dets), None, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(result, synthetic=False, figures=write_figures(result, args.out))
    title, rest = md.split("\n\n", 1)
    header = "\n\n".join([
        "## Model and data, for this run",
        (f"- Data: DENTEX (HuggingFace `ibrahimhamamci/DENTEX`, CC-BY-NC-SA 4.0). Evaluation "
         f"set: {len(records)} depth-labelled images (diagnosis train + validation), of which "
         f"**{len(inventoried)}** have a human tooth inventory from an enumeration-subset copy. "
         "Every number below uses those images only."),
        (f"- Patients: recovered from pixels ({len(recovery.pairs)} probable same-patient "
         f"pairs; mixture separated: {recovery.mixture.separated}; Ashman's D "
         f"{recovery.mixture.ashman_d:.2f}). The split is over recovered patients."),
        (f"- Stage 2 only: frozen ImageNet ResNet-50 features on human tooth crops, plus a "
         f"multinomial logistic head (C = {head.c:g}, chosen by patient-grouped CV log-loss) "
         f"trained on {int(tr.sum())} train-partition teeth. No stage-1 enumerator yet: the "
         "tooth boxes are human."),
        "## Integrity checks, for this run",
        (f"- Exact duplicates inside the evaluation set: 0 (the loader refuses them). "
         f"Near-duplicate pairs by pixel evidence: {len(near_pairs)}, every one inside a single "
         "recovered patient group, so none can cross the split"
         + (": " + "; ".join(near_pairs) if near_pairs else "") + "."),
        ("- Are the inventoried images a random subset of the evaluation set? Patient-grouped "
         "95% CIs; tests uncorrected and exploratory:"),
        selection,
        ("**The inventoried images differ from the rest on at least one of these, so every "
         "tooth-level number below carries a selection effect: it describes the "
         "enumeration-copy subset, not DENTEX as a whole.**" if selected else
         "No difference detected on these checks. That does not prove randomness, but no "
         "selection effect is visible."),
    ])
    (args.out / "report.md").write_text(f"{title}\n\n{header}\n\n{rest}")
    print(f"wrote {args.out / 'report.md'} in {time.time() - t0:.0f}s")
    for f in tripwire(result):
        print(f"TRIPWIRE  {f}")
    for f in small_cell_flags(result):
        print(f"small cell  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
