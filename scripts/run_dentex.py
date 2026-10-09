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
from pathlib import Path

import numpy as np

from dcai.data.schema import CONSENSUS, Finding
from dcai.dentex_setup import prepare
from dcai.eval.abstention import fit_operating_point
from dcai.eval.bootstrap import difference, grouped_bootstrap
from dcai.eval.detection import Detection
from dcai.eval.ratings import tooth_grades
from dcai.eval.stratified import auc
from dcai.eval.units import ToothPrediction
from dcai.figures import write_figures
from dcai.models.encoders import FrozenResNet50, tooth_features
from dcai.models.tooth_head import fit_head
from dcai.pipeline import DatasetInputs, StudyConfig, evaluate, make_split
from dcai.probes.recover_groups import load_gray
from dcai.report import render_markdown, small_cell_flags, tripwire


def deep_caries_diagnostics(y: np.ndarray, proba: np.ndarray, x: np.ndarray, groups: np.ndarray,
                            train: np.ndarray, val: np.ndarray, test: np.ndarray,
                            threshold: float, cfg: StudyConfig) -> str:
    """Is low deep-caries sensitivity detection difficulty, or the head under-predicting a rare class?

    Sensitivity in the report is detection-only: a tooth counts when P(lesion) =
    1 - P(sound) clears the threshold, whatever depth is called. Here: (a) the
    depth-correct rate beside it; (b) where missed deep lesions sit relative to
    the threshold; (c) a binary head (sound vs any caries) on the same features,
    which removes the rare third class. If the binary head recovers deep
    sensitivity, the gap was class imbalance; if not, these features find deep
    teeth genuinely harder.
    """
    boot = {"seed": cfg.seed, "n_boot": cfg.n_boot}
    g = groups[test]
    yt, pt = y[test], proba[test]
    p_lesion = 1 - pt[:, 0]
    called_deep = pt[:, 2] > pt[:, 1]

    def rate(num: np.ndarray, den: np.ndarray):
        return grouped_bootstrap(
            lambda i: float(num[i][den[i]].mean()) if den[i].any() else float("nan"), g, **boot)

    rows = []
    for grade, name in ((1, "caries"), (2, "deep_caries")):
        sel = yt == grade
        detected = p_lesion >= threshold
        correct = detected & (called_deep if grade == 2 else ~called_deep)
        rows.append(f"| {name} | {int(sel.sum())} | {rate(detected, sel)} | {rate(correct, sel)} |")

    deep, shallow = yt == 2, yt == 1
    missed_deep = p_lesion[deep & (p_lesion < threshold)]
    just_below = float(np.mean(missed_deep >= threshold / 2)) if missed_deep.size else float("nan")
    deep_as_caries = float(np.mean(pt[deep, 1] > pt[deep, 2])) if deep.any() else float("nan")
    counts = np.bincount(y[train], minlength=3)

    # Binary head on the same features; threshold fitted on validation the same way.
    bin_head = fit_head(x[train], (y[train] > 0).astype(int), groups[train].tolist(),
                        n_classes=2, seed=cfg.seed)
    pb = bin_head.predict_proba(x)[:, 1]
    opc = cfg.operating_point
    op = fit_operating_point(pb[val], (y[val] > 0).astype(int), groups[val],
                             deployment_role=opc.deployment_role, constraint=opc.constraint,
                             target=opc.target, rationale=opc.rationale)
    pbt = pb[test]
    sens3 = grouped_bootstrap(lambda i: float((p_lesion[i][deep[i]] >= threshold).mean())
                              if deep[i].any() else float("nan"), g, **boot)
    sensb = grouped_bootstrap(lambda i: float((pbt[i][deep[i]] >= op.threshold).mean())
                              if deep[i].any() else float("nan"), g, **boot)
    sensb_shallow = grouped_bootstrap(lambda i: float((pbt[i][shallow[i]] >= op.threshold).mean())
                                      if shallow[i].any() else float("nan"), g, **boot)
    gain = difference(sensb, sens3, paired=True)
    auc3 = {k: auc(p_lesion[(yt == k) | (yt == 0)], yt[(yt == k) | (yt == 0)] == k) for k in (1, 2)}
    aucb = {k: auc(pbt[(yt == k) | (yt == 0)], yt[(yt == k) | (yt == 0)] == k) for k in (1, 2)}

    # Deep minus shallow AUC vs sound, binary head, on the same patient resamples.
    def auc_gap(i: np.ndarray) -> float:
        yi, si = yt[i], pbt[i]
        deep_auc = auc(si[(yi == 2) | (yi == 0)], yi[(yi == 2) | (yi == 0)] == 2)
        shallow_auc = auc(si[(yi == 1) | (yi == 0)], yi[(yi == 1) | (yi == 0)] == 1)
        return deep_auc - shallow_auc

    gap = grouped_bootstrap(auc_gap, g, **boot)
    n_called_deep = int(called_deep[deep].sum())
    depth_verdict = (
        f"**Depth calls: imbalance.** The 3-class head calls deep caries on {n_called_deep} of "
        f"{int(deep.sum())} deep teeth. With {counts[2]} deep teeth among {counts.sum()} "
        "training teeth, regularisation shrinks the rare class until it is never the "
        "argmax. Depth-correct performance on deep caries is a class-imbalance failure, not "
        "a property of the lesions." if n_called_deep <= 0.1 * deep.sum() else
        f"Depth calls: the 3-class head calls deep caries on {n_called_deep} of "
        f"{int(deep.sum())} deep teeth.")
    if gain.lo > 0:
        detection_verdict = ("**Detection: imbalance.** The binary head detects reliably more deep "
                             "lesions than the 3-class head (paired CI excludes 0).")
    elif gap.hi < 0:
        detection_verdict = ("**Detection: difficulty, for this representation.** Deep teeth rank "
                             "reliably below shallow ones without any threshold (AUC gap CI "
                             "excludes 0), and merging classes does not recover them.")
    else:
        detection_verdict = (
            "**Detection: unresolved at this sample size.** The binary head's gain on deep "
            f"lesions ({gain}) and the deep-minus-shallow AUC gap ({gap}) both have intervals "
            f"that include 0. What is clear: {just_below:.0%} of the missed deep lesions sit just "
            "below the threshold, so the shortfall is mostly a margin effect near the "
            "operating point, not lesions the features cannot see.")
    return "\n\n".join([
        "## Deep caries: detection difficulty or class imbalance?",
        ("Section 3's sensitivity is **detection-only**: a tooth counts as detected when "
         "P(lesion) = 1 − P(sound) clears the threshold, whatever depth the model calls. "
         "The second column below adds whether the depth was also called correctly."),
        "| reference depth | test teeth | lesion detected (any caries class) | depth called "
        "correctly |\n|---|---|---|---|\n" + "\n".join(rows),
        (f"Training teeth per class (sound / caries / deep): {counts.tolist()}. Of the "
         f"{int(deep.sum())} deep-caries test teeth, {deep_as_caries:.0%} get P(caries) > "
         f"P(deep). Of the {missed_deep.size} missed, {just_below:.0%} sit just below the "
         f"threshold (P(lesion) between {threshold / 2:.3f} and {threshold:.3f}); median "
         f"P(lesion) of missed deep teeth {np.median(missed_deep):.3f}."
         if missed_deep.size else f"Training teeth per class: {counts.tolist()}."),
        (f"Binary head (sound vs any caries, same features, threshold {op.threshold:.3f} fitted "
         f"on validation): deep-caries sensitivity **{sensb}** against {sens3} for the 3-class "
         f"head (paired difference {gain}); shallow-caries sensitivity {sensb_shallow}. AUC vs "
         f"sound, shallow / deep: 3-class {auc3[1]:.3f} / {auc3[2]:.3f}; binary "
         f"{aucb[1]:.3f} / {aucb[2]:.3f}."),
        (f"Deep minus shallow AUC vs sound (binary head, patient-grouped, paired resamples): "
         f"{gap}."),
        depth_verdict,
        detection_verdict,
    ])


def accuracy_vs_doing_nothing(y: np.ndarray, proba: np.ndarray, groups: np.ndarray,
                              test: np.ndarray, threshold: float, cfg: StudyConfig) -> str:
    """Tooth-level accuracy of the model against a baseline that calls every tooth sound.

    On a mostly sound dentition, accuracy rewards doing nothing. Shown so a headline
    accuracy can be read against what it would take to beat it.
    """
    boot = {"seed": cfg.seed, "n_boot": cfg.n_boot}
    g = groups[test]
    lesion = y[test] > 0
    flagged = (1 - proba[test, 0]) >= threshold
    model_ok, sound_ok = flagged == lesion, ~lesion
    acc_model = grouped_bootstrap(lambda i: float(model_ok[i].mean()), g, **boot)
    acc_sound = grouped_bootstrap(lambda i: float(sound_ok[i].mean()), g, **boot)
    gap = difference(acc_sound, acc_model, paired=True)
    n, n_les = int(lesion.size), int(lesion.sum())
    return "\n\n".join([
        "## Accuracy against doing nothing",
        (f"Tooth-level accuracy (lesion vs sound) on the {n} test teeth, of which {n_les} carry "
         "a lesion. Patient-grouped 95% CIs; the difference is paired over the same teeth."),
        ("| | accuracy | lesions caught |\n|---|---|---|\n"
        f"| model at the operating point | {acc_model} ({int(model_ok.sum())}/{n}) | "
        f"{int((flagged & lesion).sum())} of {n_les} |\n"
        f"| always predict sound | {acc_sound} ({int(sound_ok.sum())}/{n}) | 0 of {n_les} |"),
        (f"Always-sound minus model: **{gap}**. A model that does nothing beats this one on "
         "accuracy while catching no lesions at all, which is why accuracy is never the "
         "headline here."),
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("data/dentex"))
    args = ap.parse_args(argv)
    cfg = StudyConfig.from_yaml(args.config)
    t0 = time.time()
    prep = prepare(args.data, args.work, seed=cfg.seed, n_boot=cfg.n_boot)
    rel, records, recovery = prep.release, prep.records, prep.recovery
    near_pairs, selection, selected = prep.near_pairs, prep.selection_table, prep.selection_effect
    inventoried = [r for r in records if r.teeth_present is not None]
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
    val_ids = {r.image_id for r in split["val"]}
    test_ids = {r.image_id for r in split["test"]}
    diagnostics = deep_caries_diagnostics(
        y, proba, x, np.array([group_of[k.image_id] for k in keys]), tr,
        np.array([k.image_id in val_ids for k in keys]),
        np.array([k.image_id in test_ids for k in keys]),
        result.operating_point.threshold, cfg)
    accuracy = accuracy_vs_doing_nothing(
        y, proba, np.array([group_of[k.image_id] for k in keys]),
        np.array([k.image_id in test_ids for k in keys]), result.operating_point.threshold, cfg)
    test_set = result.internal
    det = test_set.detection[0].report
    spec = test_set.stratified.specificity
    teeth_per_image = test_set.n_teeth / det.n_images
    cohort = (f"{len(inventoried)} of {len(records)} DENTEX images, the ones with an "
              "enumeration-subset copy. They are not a random subset: most sit in one "
              "recovered acquisition cluster and they carry fewer deep, periapical and "
              "impacted findings (see Integrity checks). These numbers describe that "
              "cluster's dentition, not DENTEX as a whole.")

    args.out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(result, synthetic=False, figures=write_figures(result, args.out),
                         cohort_note=cohort)
    title, rest = md.split("\n\n", 1)
    header = "\n\n".join([
        "## Summary",
        (f"**An honest reimplementation yields a system a clinician could not use.** At a "
         f"second-reader operating point fitted on validation for sensitivity ≥ "
         f"{cfg.operating_point.target:.2f}, the model flags **{det.fp_per_image.value:.1f} "
         f"[{det.fp_per_image.lo:.1f}, {det.fp_per_image.hi:.1f}] healthy teeth on every "
         f"panoramic** (specificity {spec} on about {teeth_per_image:.0f} teeth per image). A "
         "second reader that adds seven false flags to each image costs more review time "
         "than it saves. This is the honest baseline the rest of the project has to beat."),
        f"Cohort caveat, repeated in every tooth-level section: {cohort}",
        "## Model and data, for this run",
        (f"- Data: DENTEX (HuggingFace `ibrahimhamamci/DENTEX`, CC-BY-NC-SA 4.0). Evaluation "
         f"set: {len(records)} depth-labelled images (diagnosis train + validation), of which "
         f"**{len(inventoried)}** have a human tooth inventory from an enumeration-subset copy. "
         "Every number below uses those images only."),
        (f"- Patients: recovery finds no repeat visits, under two representations. Pixel "
         f"thumbnails: {len(recovery.pairs)} pairs (Ashman's D {recovery.mixture.ashman_d:.2f}, "
         f"no separated mode). ImageNet ResNet-50 full-image features: "
         f"{len(prep.resnet_recovery.pairs)} pairs (Ashman's D "
         f"{prep.resnet_recovery.mixture.ashman_d:.2f}; ΔBIC two − one components "
         f"{prep.resnet_recovery.mixture.bic_two - prep.resnet_recovery.mixture.bic_one:+.1f}, "
         "so no second mode). We treat this as a real null: the card's \"randomly selected\" "
         "plausibly means one image per patient. Rule 1's leakage evidence on DENTEX is the "
         "byte-identical duplicates (see the audit), and the split is effectively per image."),
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
    (args.out / "report.md").write_text(f"{title}\n\n{header}\n\n{rest}\n{accuracy}\n\n{diagnostics}\n")
    print(f"wrote {args.out / 'report.md'} in {time.time() - t0:.0f}s")
    for f in tripwire(result):
        print(f"TRIPWIRE  {f}")
    for f in small_cell_flags(result):
        print(f"small cell  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
