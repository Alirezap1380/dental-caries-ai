"""Stage 1: tooth enumerator, trained with and without copies of the test images.

    python scripts/run_stage1.py --config configs/dentex_stage2_frozen.yaml \
        --data data/dentex/raw --out results/dentex_stage1 --epochs 12

The evaluation set (755 images) is split by recovered patient. Three enumerators
are trained with identical settings:

- honest: the enumeration subset minus every image whose content or recovered
  patient is in the held-out validation/test partitions;
- swap: the honest set with as many randomly chosen images replaced by the
  excluded ones (which include copies of the test images). Same SIZE as
  honest, so swap minus honest isolates the effect of training on copies
  (figure 1);
- naive: the whole enumeration subset, as the benchmark intends. Larger than
  honest, so naive minus honest mixes the copy effect with more training data.

All three are scored on the SAME held-out test images that have a human
enumeration (teeth found AND numbered correctly), so every difference is paired.
Each saved model has a manifest of the exact images it trained on.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    excluded = sorted(set(naive) - set(honest))
    rng = np.random.default_rng(cfg.seed)
    kept = sorted(rng.choice(honest, size=len(honest) - len(excluded), replace=False).tolist())
    swap = sorted(kept + excluded)
    print(f"enumeration images: {len(naive)} distinct; honest training set {len(honest)} "
          f"({len(naive) - len(honest)} excluded as held-out content or patients)")

    # Guard: the experiment must be the one the report describes. An earlier run
    # silently trained two models instead of three after an edit failed; check
    # the configuration itself, not the code that is supposed to produce it.
    plan = {"honest": honest, "swap": swap, "naive": naive}
    held_shas = {r.meta["sha256"] for r in held_out}
    problems = [msg for ok, msg in (
        (len(plan) == 3, "expected three models"),
        (len(swap) == len(honest), f"swap has {len(swap)} images, honest {len(honest)}"),
        (set(excluded) <= set(swap), "swap does not contain every excluded image"),
        (not (set(honest) & held_shas), "honest shares content with held-out images"),
        (set(honest) | set(swap) <= set(naive), "naive is not a superset of honest and swap"),
        (len(excluded) > 0, "no excluded images: swap would equal honest"),
    ) if not ok]
    if problems:
        raise SystemExit("stage-1 configuration is not the experiment described: "
                         + "; ".join(problems))

    ecfg = EnumeratorConfig(epochs=args.epochs, seed=cfg.seed)
    models.mkdir(parents=True, exist_ok=True)
    fitted = {}
    import torch

    for name, shas in plan.items():
        samples = [((lambda h=h: load_gray(store / f"{h}.png")), teeth_of[h]) for h in shas]
        print(f"training {name} enumerator on {len(samples)} images")
        model = train_enumerator(samples, ecfg, log=lambda m, n=name: print(f"  [{n}] {m}"))
        torch.save(model.state_dict(), models / f"enumerator_{name}.pt")
        (models / f"enumerator_{name}.json").write_text(json.dumps(
            {"config": dataclasses.asdict(ecfg), "trained_on_sha256": shas}, indent=1))
        fitted[name] = model
        print(f"  [{name}] done ({time.time() - t0:.0f}s)")

    if set(fitted) != set(plan):
        raise SystemExit(f"refusing to write a report: trained {sorted(fitted)}, "
                         f"planned {sorted(plan)}")

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
    in_train = {name: sum(r.meta["sha256"] in set(shas) for r in test_imgs)
                for name, shas in (("honest", honest), ("swap", swap), ("naive", naive))}

    def est(name: str, which: str) -> Estimate:
        c = np.array(counts[name], dtype=float)
        col = 1 if which == "precision" else 2
        return grouped_bootstrap(lambda i: c[i, 0].sum() / c[i, col].sum(), groups,
                                 seed=cfg.seed, n_boot=cfg.n_boot)

    rows, fig, trips = [], {}, []
    for which in ("recall", "precision"):
        for name in plan:
            e = est(name, which)
            if e.value > 0.95:
                trips.append(
                    f"- ⚠ {name} {which}: {e} — **expected**: {in_train[name]} of "
                    f"{len(test_imgs)} scored test images are byte-identical copies inside "
                    "its training set. This is the leak the experiment measures, not a "
                    "result to report." if in_train[name] else
                    f"- ⚠ {name} {which}: {e} — **unexpected**: none of the scored test "
                    "images is in its training set. Investigate before believing it.")
        h, sw, n = est("honest", which), est("swap", which), est("naive", which)
        d_copy, d_naive = difference(sw, h, paired=True), difference(n, h, paired=True)
        fig[which] = (h, sw)
        rows.append(f"| {which} | {h} | {sw} | {n} | **{d_copy}** | {d_naive} |")

    args.out.mkdir(parents=True, exist_ok=True)
    plot(fig, args.out / "figure1_enumeration.svg")
    md = "\n\n".join([
        "# Stage 1: tooth enumeration, with and without copies of the test images",
        (f"DENTEX, CC-BY-NC-SA 4.0. torchvision Faster R-CNN (MobileNetV3-FPN, COCO "
         f"pretrained), {args.epochs} epochs, identical settings for all three models. The "
         f"split is over all {len(records)} evaluation images by recovered patient "
         f"({len(held_out)} images held out as validation + test)."),
        (f"- **honest** ({len(honest)} images): no image shares content or recovered patient "
         f"with a held-out image. Scored test images in its training set: {in_train['honest']}.\n"
         f"- **swap** ({len(swap)} images, same size): honest with {len(excluded)} random "
         "images replaced by the excluded ones, which include copies of the test images. "
         f"Scored test images in its training set: {in_train['swap']}. **swap − honest is the "
         "effect of training on copies at fixed training-set size: figure 1.**\n"
         f"- **naive** ({len(naive)} images): the whole enumeration subset, as the benchmark "
         f"intends. Scored test images in its training set: {in_train['naive']}. naive − "
         f"honest also includes {len(naive) - len(honest)} more training images, so it is "
         "confounded."),
        (f"Scored on the {len(test_imgs)} held-out test images that have a human enumeration. "
         "A tooth counts as correct when its FDI number is right AND its box overlaps the "
         "human box at IoU ≥ 0.5. Patient-grouped 95% bootstrap CIs; differences are paired "
         "(same images, same resamples)."),
        "### Leakage tripwire (> 0.95)\n\n" + ("\n".join(trips) if trips else
                                                "No estimate exceeds 0.95."),
        ("| | honest | swap | naive | swap − honest (copies, size-matched) | naive − honest "
         "(confounded) |\n|---|---|---|---|---|---|\n" + "\n".join(rows)),
        "![Figure 1](figure1_enumeration.svg)",
        ("*Figure 1. Teeth found and correctly numbered, enumerators trained with (swap) and "
         "without (honest) byte-identical copies of the scored test images, at equal "
         "training-set size.*"),
        ("Note on the patient split: patient recovery finds no repeat visits under either "
         "pixel thumbnails or ResNet-50 features (treated as a real null; see README), so "
         "held-out patients are held-out images."),
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
        h, sw = fig[which]
        for off, (label, e, color) in zip(
                (-0.1, 0.1),
                (("honest: no copies", h, SERIES["internal"]),
                 ("swap: copies, same size", sw, SERIES["external"])),
                strict=True):
            ax.errorbar([k + off], [e.value], yerr=[[e.value - e.lo], [e.hi - e.value]],
                        fmt="o", color=color, elinewidth=2, markersize=7, capsize=0,
                        label=label if k == 0 else None)
    ax.set_xticks([0, 1], ["recall", "precision"], color=INK)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.8, 1.0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    f.tight_layout()
    f.savefig(path, format="svg", facecolor="#fcfcfb")
    plt.close(f)


if __name__ == "__main__":
    raise SystemExit(main())
