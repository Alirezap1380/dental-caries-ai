"""Stage 2 on human tooth boxes, through the full evaluation pipeline.

    python scripts/run_dentex.py --config configs/dentex_stage2_frozen.yaml \
        --data data/dentex/raw --out results/dentex_stage2_frozen

1. Load the release by content; recover patient groups from pixel signatures.
2. Keep the images with a human tooth inventory; split them by recovered patient.
3. Frozen ImageNet ResNet-50 features for every inventoried tooth.
4. Fit the head on train-partition teeth only; predict every tooth.
5. Evaluate with the existing pipeline and write the report.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dcai.data.dentex import extract_images, load_dentex
from dcai.data.near_duplicates import signatures_for
from dcai.data.schema import CONSENSUS, Finding
from dcai.eval.detection import Detection
from dcai.eval.ratings import tooth_grades
from dcai.eval.units import ToothPrediction
from dcai.figures import write_figures
from dcai.models.encoders import FrozenResNet50, tooth_features
from dcai.models.tooth_head import fit_head
from dcai.pipeline import DatasetInputs, StudyConfig, evaluate, make_split
from dcai.probes.recover_groups import load_gray, recover_patients, with_recovered_patients
from dcai.report import render_markdown, small_cell_flags, tripwire


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
    ids = [r.image_id for r in rel.records]
    shas = [r.meta["sha256"] for r in rel.records]
    recovery = recover_patients(ids, np.array([sigs[h].phash for h in shas]),
                                np.array([sigs[h].thumb for h in shas]), seed=cfg.seed)
    records = with_recovered_patients(list(rel.records), recovery)
    inventoried = [r for r in records if r.teeth_present is not None]

    split = make_split(inventoried, cfg)
    train_ids = {r.image_id for r in split["train"]}
    by_id = {r.image_id: r for r in inventoried}
    keys, x = tooth_features(
        [(r.image_id, (lambda p=r.path: load_gray(Path(p))), rel.tooth_boxes[r.image_id])
         for r in inventoried],
        FrozenResNet50(),
    )
    grades = {r.image_id: tooth_grades(r, CONSENSUS, cfg.scale) for r in inventoried}
    y = np.array([grades[k.image_id][k.tooth_fdi] for k in keys])
    tr = np.array([k.image_id in train_ids for k in keys])
    head = fit_head(x[tr], y[tr], [by_id[k.image_id].group_key for k in keys if k.image_id in train_ids],
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
    model = "\n\n".join([
        "## Model and data, for this run",
        (f"- Data: DENTEX (HuggingFace `ibrahimhamamci/DENTEX`, CC-BY-NC-SA 4.0). Evaluation "
        f"set: {len(records)} depth-labelled images (diagnosis train + validation), of which "
        f"**{len(inventoried)}** have a human tooth inventory from an enumeration-subset copy. "
        "Every number below uses those images only."),
        (f"- Patients: recovered from pixels ({len(recovery.pairs)} probable same-patient pairs; "
        f"mixture separated: {recovery.mixture.separated}; Ashman's D "
        f"{recovery.mixture.ashman_d:.2f}). The split is over recovered patients."),
        (f"- Stage 2 only: frozen ImageNet ResNet-50 features on human tooth crops, plus a "
        f"multinomial logistic head (C = {head.c:g}, chosen by patient-grouped CV log-loss) "
        f"trained on {int(tr.sum())} train-partition teeth. No stage-1 enumerator yet: the "
        "tooth boxes are human."),
    ])
    (args.out / "report.md").write_text(f"{title}\n\n{model}\n\n{rest}")
    print(f"wrote {args.out / 'report.md'} in {time.time() - t0:.0f}s")
    for f in tripwire(result):
        print(f"TRIPWIRE  {f}")
    for f in small_cell_flags(result):
        print(f"small cell  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
