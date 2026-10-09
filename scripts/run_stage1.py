"""Stage 1: tooth enumerator, trained honestly and naively. Figure 1 measures the gap.

    python scripts/run_stage1.py --config configs/dentex_stage2_frozen.yaml \
        --data data/dentex/raw --out results/dentex_stage1 --epochs 12

The evaluation set (755 images) is split by recovered patient. Two enumerators
are trained with identical settings:

- honest: the enumeration subset minus every image whose content or recovered
  patient is in the held-out validation/test partitions;
- naive: the whole enumeration subset, as the benchmark intends. It contains
  byte-identical copies of held-out images.

Both are scored on the SAME held-out test images that have a human enumeration:
teeth found AND numbered correctly. Because the images are shared, the
naive-minus-honest difference is paired, and it is the effect of training on
copies of the test images (figure 1).
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from dcai.data.dentex import COCO_FILES, TRAIN_ZIP, parse_enumeration_coco
from dcai.dentex_setup import prepare
from dcai.eval.bootstrap import Estimate, difference, grouped_bootstrap
from dcai.models.enumerator import (
    EnumeratorConfig,
    eligible_training_images,
    predict_teeth,
    score_enumeration,
    train_enumerator,
)
from dcai.pipeline import StudyConfig, make_split
from dcai.probes.recover_groups import load_gray


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("data/dentex"))
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args(argv)
    cfg = StudyConfig.from_yaml(args.config)
    t0 = time.time()
    store, models = args.work / "images", args.work / "models"

    prep = prepare(args.data, args.work, seed=cfg.seed, n_boot=cfg.n_boot,
                   extract=("diagnosis", "validation", "enumeration"))
    rel, records = prep.release, prep.records
    split = make_split(records, cfg)
    held_out = [r for name in ("val", "test") for r in split[name]]
    group_of_sha = {r.meta["sha256"]: r.group_key for r in records}

    with zipfile.ZipFile(args.data / TRAIN_ZIP) as zf:
        enumeration = parse_enumeration_coco(json.loads(zf.read(COCO_FILES["enumeration"])))
    teeth_of: dict[str, list] = {}
    for e in rel.index:
        if e.subset == "enumeration" and e.sha256 not in teeth_of:
            teeth_of[e.sha256] = enumeration[e.file_name]
    candidates = [(sha, group_of_sha.get(sha, f"enumeration-only/{sha}")) for sha in teeth_of]
    honest = eligible_training_images(
        candidates, [(r.meta["sha256"], r.group_key) for r in held_out])
    naive = sorted(teeth_of)
    print(f"enumeration images: {len(naive)} distinct; honest training set {len(honest)} "
          f"({len(naive) - len(honest)} excluded as held-out content or patients)")

    ecfg = EnumeratorConfig(epochs=args.epochs, seed=cfg.seed)
    models.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for name, shas in (("honest", honest), ("naive", naive)):
        samples = [((lambda h=h: load_gray(store / f"{h}.png")), teeth_of[h]) for h in shas]
        print(f"training {name} enumerator on {len(samples)} images")
        model = train_enumerator(samples, ecfg, log=lambda m, n=name: print(f"  [{n}] {m}"))
        import torch
        torch.save(model.state_dict(), models / f"enumerator_{name}.pt")
        fitted[name] = model
        print(f"  [{name}] done ({time.time() - t0:.0f}s)")

    # Score on held-out TEST images with a human enumeration.
    test_imgs = [r for r in split["test"] if r.image_id in rel.tooth_boxes]
    counts = {name: [] for name in fitted}
    for r in test_imgs:
        img = load_gray(Path(r.path))
        truth = rel.tooth_boxes[r.image_id]
        for name, model in fitted.items():
            pred = {f: b for f, (b, _) in predict_teeth(model, img).items()}
            s = score_enumeration(pred, truth)
            counts[name].append((s.correct, s.n_pred, s.n_true))
    groups = [r.group_key for r in test_imgs]
    copies = sum(r.meta["sha256"] in set(naive) for r in test_imgs)

    def est(name: str, which: str) -> Estimate:
        c = np.array(counts[name], dtype=float)
        col = 1 if which == "precision" else 2
        return grouped_bootstrap(lambda i: c[i, 0].sum() / c[i, col].sum(), groups,
                                 seed=cfg.seed, n_boot=cfg.n_boot)

    rows, fig = [], {}
    for which in ("recall", "precision"):
        h, n = est("honest", which), est("naive", which)
        d = difference(n, h, paired=True)
        fig[which] = (h, n, d)
        rows.append(f"| {which} | {h} | {n} | **{d}** |")

    args.out.mkdir(parents=True, exist_ok=True)
    plot(fig, args.out / "figure1_enumeration.svg")
    md = "\n\n".join([
        "# Stage 1: tooth enumeration, honest vs naive training",
        (f"DENTEX, CC-BY-NC-SA 4.0. torchvision Faster R-CNN (MobileNetV3-FPN, COCO "
        f"pretrained), {args.epochs} epochs, identical settings for both models. The split "
        f"is over all {len(records)} evaluation images by recovered patient "
        f"({len(held_out)} images held out as validation + test)."),
        (f"- **honest** trained on {len(honest)} enumeration images: none shares content "
        "or recovered patient with a held-out image.\n"
        f"- **naive** trained on all {len(naive)} distinct enumeration images, as the "
        "benchmark intends."),
        (f"Scored on the {len(test_imgs)} held-out test images that have a human "
        f"enumeration ({copies} of them have a byte-identical copy in the naive training "
        "set). A tooth counts as correct when its FDI number is right AND its box overlaps "
        "the human box at IoU ≥ 0.5. Patient-grouped 95% bootstrap CIs; the difference is "
        "paired (same images, same resamples)."),
        "| | honest | naive | naive − honest (paired) |\n|---|---|---|---|\n" + "\n".join(rows),
        "![Figure 1](figure1_enumeration.svg)",
        ("Note on the patient split: patient recovery found no repeat visits in this data "
        "(a weak null, see README), so held-out *patients* are effectively held-out images."),
    ])
    (args.out / "report.md").write_text(md + "\n")
    print(f"wrote {args.out / 'report.md'} in {time.time() - t0:.0f}s")
    print("\n".join(rows))
    return 0


def plot(fig: dict, path: Path) -> None:
    from dcai.figures import INK, INK_2, SERIES, _axes, plt

    f, ax = _axes("Figure 1: training on copies of the test images (stage 1)",
                  "teeth found and correctly numbered")
    for k, which in enumerate(("recall", "precision")):
        h, n, _ = fig[which]
        for off, (label, e, color) in zip((-0.1, 0.1), (("honest", h, SERIES["internal"]),
                                                        ("naive", n, SERIES["external"])),
                                          strict=True):
            ax.errorbar([k + off], [e.value], yerr=[[e.value - e.lo], [e.hi - e.value]],
                        fmt="o", color=color, elinewidth=2, markersize=7, capsize=0,
                        label=label if k == 0 else None)
    ax.set_xticks([0, 1], ["recall", "precision"], color=INK)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    f.tight_layout()
    f.savefig(path, format="svg", facecolor="#fcfcfb")
    plt.close(f)


if __name__ == "__main__":
    raise SystemExit(main())
