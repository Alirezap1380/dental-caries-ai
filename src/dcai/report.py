"""Render an `EvaluationResult` as a Markdown report.

Two rules are built in rather than left to the author:

- **Multiplicity.** The report states up front that every interval is an
  uncorrected, exploratory 95% patient-level bootstrap interval, except the
  pre-specified headline claims, which are Bonferroni-corrected across their family.
- **The 0.95 tripwire.** Any performance estimate above 0.95 is marked and listed
  at the top of the report as something to explain before it is believed. At
  these sample sizes, numbers that high usually mean leakage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from dcai import __version__
from dcai.eval.bootstrap import Estimate
from dcai.eval.stratified import StratifiedReport
from dcai.pipeline import EvaluationResult

LEAK_THRESHOLD = 0.95
# Below this many units an estimate above 0.95 is most likely small-sample noise
# (3 of 3 lesions found is 1.000). Such flags are still listed, separately, so the
# tripwire neither hides anything nor cries wolf on every tiny subgroup cell.
MIN_CELL = 20
FLAG = "⚠"


class _Formatter:
    def __init__(self) -> None:
        self.flags: list[str] = []
        self.small_cell_flags: list[str] = []

    def metric(self, est: Estimate | None, where: str, n: int | None = None) -> str:
        """A performance estimate (sensitivity, AUC, AP, kappa...): subject to the tripwire.

        `n` is the number of units the estimate rests on (e.g. lesions in the stratum).
        """
        if est is None:
            return "—"
        text = str(est)
        if est.value > LEAK_THRESHOLD:
            if n is not None and n < MIN_CELL:
                self.small_cell_flags.append(f"{where} (n={n}): {text}")
            else:
                self.flags.append(f"{where}: {text}")
            text += f" {FLAG}"
        return text

    @staticmethod
    def plain(est: Estimate | None) -> str:
        return "—" if est is None else str(est)


def _sets(r: EvaluationResult) -> list:
    """The evaluated sets present: internal always, external when there is one."""
    return [s for s in (r.internal, r.external) if s is not None]


def _table(header: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    header = list(header)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _stratified_rows(
    f: _Formatter, sets: list[tuple[str, StratifiedReport]]
) -> list[list[object]]:
    first = sets[0][1]
    rows: list[list[object]] = []
    for i, stratum in enumerate(first.strata):
        row: list[object] = [f"**{stratum.name}**"]
        for name, rep in sets:
            s = rep.strata[i]
            row += [s.n_units, f.metric(s.sensitivity, f"{name} sensitivity ({s.name})", s.n_units),
                    f.metric(s.auc_vs_sound, f"{name} AUC vs sound ({s.name})", s.n_units)]
        rows.append(row)
    spec: list[object] = ["sound teeth: specificity"]
    pooled: list[object] = ["*pooled sensitivity (shown only to expose what pooling hides)*"]
    for name, rep in sets:
        spec += [rep.n_sound, f.metric(rep.specificity, f"{name} specificity", rep.n_sound), "—"]
        n_lesions = sum(s.n_units for s in rep.strata)
        pooled += [n_lesions, f.plain(rep.pooled_sensitivity), "—"]
    return [*rows, spec, pooled]


def _section_data(r: EvaluationResult) -> str:
    rows = []
    for s in _sets(r):
        d = s.summary
        rows.append([
            s.name, d.n_images, d.n_patients, s.n_teeth, d.max_images_per_patient,
            ", ".join(d.sites) or "—", d.multi_reader_images,
            ", ".join(f"{k} {v}" for k, v in d.patient_id_sources.items()),
            ", ".join(f"{k} {v}" for k, v in d.inventory_sources.items()),
        ])
    sp = r.splits
    split_rows = [[k, sp.images[k], sp.patients[k]] for k in sp.images]
    used = sp.patient_leakage
    if sp.kind.value == "image_naive":
        split_line = (
            f"Internal set, **naive image-level split (E0)**. **THIS SPLIT LEAKS: "
            f"{len(used.leaked_groups)} of {used.n_groups} patients ({_pct(used.leaked_fraction)}) "
            f"appear in more than one partition, and {_pct(sp.test_images_with_seen_patient)} of "
            "test images belong to a patient the model saw in train or validation.** Every "
            "internal-test number below is inflated by that contamination."
        )
    else:
        split_line = (
            "Internal set, patient-level split. The leakage check ran at construction and "
            f"passed: {used.n_groups} patients, none in more than one partition; "
            f"{_pct(sp.test_images_with_seen_patient)} of test images belong to a seen patient."
        )
    naive = sp.naive_leakage
    naive_line = (
        f"A naive image-level split of the same data would put **{len(naive.leaked_groups)} "
        f"of {naive.n_groups} patients ({_pct(naive.leaked_fraction)})** in more than one "
        "partition. That is the contamination an image-level split carries. E0 reproduces "
        "it deliberately."
    )
    if sp.naive_note:
        naive_line += f"\n\n> {sp.naive_note}"
    return "\n\n".join([
        "## 1. Data and splits",
        _table(["set", "images", "patients", "teeth", "max images / patient", "sites",
                "multi-reader images", "patient ids", "tooth inventory"], rows),
        split_line,
        _table(["partition", "images", "patients"], split_rows),
        naive_line,
    ])


def _section_agreement(f: _Formatter, r: EvaluationResult, figures: Mapping[str, str]) -> str:
    a = r.agreement
    parts = [
        "## 2. Reader agreement (internal test, tooth level)",
        (f"{a.n_teeth} teeth in {a.n_images} images, readers {', '.join(a.readers)}. "
        "The unit is the tooth (FDI), so readers are paired with no box-matching step. "
        f"Cohen's kappa uses {a.weights} weights on the ordinal depth scale."),
        (f"- Krippendorff's alpha (ordinal, handles unread images): "
        f"**{f.metric(a.krippendorff, 'Krippendorff alpha')}**" if a.krippendorff is not None
         else f"- Agreement: {a.fleiss_reason}"),
        f"- Fleiss' kappa: {f.metric(a.fleiss, 'Fleiss kappa') if a.fleiss else a.fleiss_reason}",
        "Individual reader pairs (how much humans disagree; this is *not* the ceiling):",
        _table(["reader A", "reader B", "teeth", "kappa"],
               [[p.rater_a, p.rater_b, p.n_items, f.metric(p.kappa, f"kappa {p.rater_a}-{p.rater_b}")]
                for p in a.pairwise]),
    ]
    sw = a.sweep
    if sw is None:
        parts.append(f"Ceiling not computable: {a.ceiling_reason}")
        return "\n\n".join(parts)

    parts += [
        ("**Model against the human ceiling, across thresholds.** The ceiling for reader X is "
        "the agreement between X and the leave-one-out consensus of the other readers, so a "
        "consensus-trained model is compared with a consensus (matched variance). The band "
        "is the mean ceiling over readers. Kappa at one threshold would mix \"worse than "
        "readers\" with \"pinned to a threshold readers don't use\", so the model is swept."),
        f"- Human ceiling (mean LOO-consensus κ): **{f.metric(sw.ceiling, 'ceiling kappa')}**",
        (f"- Model at the operating point (threshold {sw.operating_threshold:.3f}): "
        f"{f.metric(sw.at_operating, 'model kappa at operating point')}"),
        (f"- Model at its κ-optimal threshold ({sw.best_threshold:.3f}): "
        f"{f.metric(sw.at_best, 'model kappa at best threshold')}. This threshold was chosen on "
        "these data, so the value is optimistic: a diagnostic, not a deployable number."),
        "- Does any threshold reach the band? "
        + ("**yes**: the ranking can match readers, so any shortfall at the operating point is "
           "a threshold choice." if sw.reaches_ceiling else
           "**no**: at no threshold does the model agree with readers as well as their own "
           "consensus does."),
    ]
    if sw.operating_point_costs_agreement:
        parts.append(
            f"**The operating point and the κ-optimal threshold differ a lot.** Moving the "
            f"threshold from {sw.operating_threshold:.3f} to {sw.best_threshold:.3f} raises κ by "
            f"{f.plain(sw.best_minus_operating)} (paired). At the deployed threshold the model "
            "calls lesions far more readily than the readers do, and that part of the shortfall "
            "is the price of the binding constraint. "
            + ("It is not evidence that the model ranks teeth worse than readers: at its best "
               "threshold it reaches the band." if sw.reaches_ceiling else
               "It is not the whole story: even at its best threshold the model stays below the "
               "band, so the rest of the gap is the model.")
        )
    else:
        parts.append(
            "The operating point sits close to the κ-optimal threshold "
            f"(paired difference {f.plain(sw.best_minus_operating)})."
        )
    if "sweep" in figures:
        parts.append(f"![Model kappa across thresholds]({figures['sweep']})")
    parts += [
        (f"Per reader, at the κ-optimal threshold ({sw.best_threshold:.3f}), the model's most "
        "charitable setting. If it does not exceed a reader's ceiling here, it does not "
        "exceed it anywhere."),
        _table(
            ["reader", "teeth", "model κ", "ceiling κ (LOO consensus)",
             "mean individual κ", "model − ceiling", "verdict"],
            [[c.reader, c.n_items, f.metric(c.model_kappa, f"model kappa vs {c.reader}"),
              f.metric(c.ceiling_kappa, f"ceiling kappa {c.reader}"),
              f.plain(c.individual_kappa), f.plain(c.difference),
              "**exceeds ceiling: explain before claiming**" if c.exceeds_ceiling
              else "below ceiling" if c.below_ceiling else "within ceiling"]
             for c in a.ceiling],
        ),
        ("*Exceeding the ceiling is not a success.* It means reader mimicry, leakage, or a "
        "model that averages label noise better than an (N−1)-reader panel. It needs "
        "explaining."),
    ]
    return "\n\n".join(parts)


def _section_stratified(f: _Formatter, r: EvaluationResult, figures: Mapping[str, str]) -> str:
    op = r.operating_point
    sets = [("internal", r.internal.stratified)]
    if r.external is not None:
        sets.append(("external", r.external.stratified))
    parts = [
        "## 3. Depth-stratified tooth-level performance",
        (f"Operating point fitted on internal *validation* patients only. Deployment role: "
        f"**{op.deployment_role.value.replace('_', ' ')}**; binding constraint: "
        f"**{op.constraint} ≥ {op.target:.2f}** (achieved {op.fitted_value:.3f} on validation) "
        f"at threshold **{op.threshold:.3f}** on P(lesion). It is frozen for internal test "
        "and external."),
        f"> Rationale on record: {op.rationale}",
        "Reference standard: strict-majority consensus of each image's readers, per tooth.",
        _table(
            ["depth", *(f"{h} ({name})" for name, _ in sets
                        for h in ("n", "sensitivity", "AUC vs sound"))],
            _stratified_rows(f, sets),
        ),
    ]
    if "depth" in figures:
        parts.append(f"![Sensitivity by lesion depth]({figures['depth']})")
    return "\n\n".join(parts)


def _section_detection(f: _Formatter, r: EvaluationResult) -> str:
    parts = [
        "## 4. Lesion-level detection, per reader",
        (f"Boxes matched to each reader's own lesions (IoU ≥ {r.config.iou_threshold}); "
        "sensitivity and FP/image at box score ≥ "
        + (f"{r.operating_point.threshold:.3f} (the operating threshold)"
           if r.config.box_score_threshold is None else f"{r.config.box_score_threshold}")
        + ". AP per "
        "depth ignores lesions of the other depth, and background false positives count "
        "against every depth."),
    ]
    names = r.config.scale.categories[1:]
    for s in _sets(r):
        rows = []
        for d in s.detection:
            sens = [f.metric(x.sensitivity, f"{s.name} {d.reader} lesion sensitivity ({x.name})",
                             x.n_units) for x in d.report.sensitivity.strata]
            aps = [f.metric(x.ap, f"{s.name} {d.reader} AP ({x.name})", x.n_lesions)
                   for x in d.ap.strata]
            rows.append([d.reader, d.n_images, *sens, f.plain(d.report.fp_per_image), *aps,
                         f.plain(d.ap.pooled)])
        parts += [
            f"**{s.name}**",
            _table(["reader", "images", *(f"sens {n}" for n in names), "FP / image",
                    *(f"AP {n}" for n in names), "*pooled AP*"], rows),
        ]
    return "\n\n".join(parts)


def _section_calibration(f: _Formatter, r: EvaluationResult, figures: Mapping[str, str]) -> str:
    rows = []
    for s in _sets(r):
        c = s.calibration
        rows.append([
            s.name, c.n, f"{c.prevalence:.3f}", f"{c.mean_predicted:.3f}", f.plain(c.ece),
            f"{c.ece_noise_floor:.3f}", "yes" if c.ece_exceeds_noise_floor else "no",
            f.plain(c.brier), f.plain(c.slope), f.plain(c.intercept),
        ])
    parts = [
        "## 5. Calibration (tooth level)",
        ("Per-tooth P(lesion) against the consensus reference. Box scores are deliberately "
        "not calibrated, because a detector chooses how many boxes to emit. The noise floor "
        "is the 95th percentile of ECE for a perfectly calibrated model at this n. "
        "Slope < 1 means overconfident; intercept < 0 means it over-predicts."),
        _table(["set", "teeth", "prevalence", "mean P", (f"ECE ({r.internal.calibration.strategy}"
                f" bins)"), "ECE noise floor", "exceeds floor", "Brier", "slope", "intercept"],
               rows),
    ]
    if "reliability" in figures:
        parts.append(f"![Reliability diagram]({figures['reliability']})")
    return "\n\n".join(parts)


def _section_abstention(f: _Formatter, r: EvaluationResult) -> str:
    op = r.operating_point
    rows = []
    for s in _sets(r):
        x = s.selective
        rows.append([
            s.name, f.plain(x.coverage),
            f.metric(x.sensitivity_accepted, f"{s.name} sensitivity among accepted"),
            f.metric(x.specificity_accepted, f"{s.name} specificity among accepted"),
            f.plain(x.missed_positive_rate),
            f.metric(x.sensitivity_no_abstention, f"{s.name} sensitivity, no abstention"),
            f.metric(x.specificity_no_abstention, f"{s.name} specificity, no abstention"),
            f"**{x.patient_overlap}**" if x.patient_overlap else "0",
        ])
    return "\n\n".join([
        "## 6. Abstention at the operating point",
        (f"Teeth within ±{op.margin:.3f} of the threshold are referred to a clinician; the band "
        f"was sized on validation for {_pct(op.coverage_target)} coverage. *Missed positive "
        "rate* is lesions the system cleared without referral, as a share of all lesions. "
        "That is the number that matters clinically."),
        _table(["set", "coverage", "sens (accepted)", "spec (accepted)", "missed positive rate",
                "sens (no abstention)", "spec (no abstention)",
                "patients shared with fitting set"], rows),
    ])


def _section_subgroups(f: _Formatter, r: EvaluationResult) -> str:
    parts = [
        "## 7. Subgroups",
        ("Depth-stratified within every subgroup: age is confounded with lesion depth, so a "
        "pooled per-age sensitivity would invent an age effect."),
    ]
    names = r.config.scale.categories[1:]
    for s in _sets(r):
        for sg in s.subgroups:
            title = f"**{s.name}, by {sg.attribute}**"
            if sg.not_computable:
                parts.append(f"{title}: {sg.not_computable}.")
                continue
            rows = []
            for c in sg.cells:
                if c.report is None:
                    rows.append([c.value, c.n_units, c.n_groups, *(["—"] * len(names)), c.reason])
                    continue
                rows.append([c.value, c.n_units, c.n_groups, *(
                    f.metric(x.sensitivity, f"{s.name} {sg.attribute}={c.value} sens ({x.name})",
                             x.n_units)
                    for x in c.report.strata), ""])
            parts += [
                f"{title} (metadata coverage {_pct(sg.coverage)})",
                _table([sg.attribute, "teeth", "patients", *(f"sens {n}" for n in names), "note"],
                       rows),
            ]
    return "\n\n".join(parts)


def _section_headline(r: EvaluationResult) -> str:
    rows = []
    for h in r.headline:
        if h.estimate is None:
            rows.append([h.claim, "not computable", h.reason])
            continue
        e = h.estimate
        verdict = "excludes 0" if e.lo > 0 or e.hi < 0 else "**includes 0: not supported**"
        rows.append([h.claim, str(e), verdict])
    level = r.headline[0].estimate.level if r.headline and r.headline[0].estimate else None
    lvl = f"{100 * level:.2f}%" if level else "corrected"
    return "\n\n".join([
        "## Headline claims (pre-specified)",
        (f"The three claims fixed in advance, with **Bonferroni-corrected {lvl} intervals** "
        f"(family-wise α = {r.config.headline_alpha}). Sensitivities are tooth-level at the "
        "frozen operating point. The external gaps compare independent samples."),
        _table(["claim", f"estimate [{lvl} CI]", "verdict"], rows),
    ])


def render_markdown(
    r: EvaluationResult, *, synthetic: bool, figures: Mapping[str, str] | None = None
) -> str:
    figures = figures or {}
    f = _Formatter()
    body = [
        _section_headline(r),
        _section_data(r),
        _section_agreement(f, r, figures),
        _section_stratified(f, r, figures),
        _section_detection(f, r),
        _section_calibration(f, r, figures),
        _section_abstention(f, r),
        _section_subgroups(f, r),
        "## 8. Not computable on this data\n\n"
        + ("\n".join(f"- {n}" for n in r.not_computable) or "Nothing."),
    ]

    head = [f"# Evaluation report: {r.config.name}"]
    if r.splits.kind.value == "image_naive":
        head.append(
            "> **E0: NAIVE IMAGE-LEVEL SPLIT.** This run deliberately reproduces the leaky "
            "protocol. Patients cross partitions (section 1), so internal-test numbers are "
            "inflated by construction. The tripwire below is expected to fire, and the "
            "external numbers are the honest ones."
        )
    if synthetic:
        head.append(
            "> **SYNTHETIC DATA.** Every number below is a property of the generator "
            "(`dcai/data/synthetic.py`, `dcai/simulate.py`), not of any trained model. This "
            "run exists to show that the harness composes end to end."
        )
    head += [
        (f"dcai {__version__} · seed {r.config.seed} · {r.config.n_boot} bootstrap resamples · "
        f"scale `{r.config.scale.name}`"),
        ("### How to read this report\n\n"
        "- **Every interval is a 95% percentile bootstrap, resampled by patient, uncorrected "
        "for multiple comparisons, and exploratory.** The only exceptions are the headline "
        "claims, which carry Bonferroni-corrected intervals.\n"
        f"- {FLAG} marks any performance estimate above {LEAK_THRESHOLD}. At these sample "
        "sizes, explain it (look for leakage first) before believing it.\n"
        "- Tooth-level metrics use the strict-majority reader consensus as reference. "
        "Lesion-level detection is scored against each reader separately."),
        "### Leakage tripwire\n\n" + (
            "\n".join(f"- {FLAG} {x}" for x in f.flags) if f.flags
            else f"No performance estimate resting on ≥ {MIN_CELL} units exceeds {LEAK_THRESHOLD}."
        ) + (
            f"\n\nAbove {LEAK_THRESHOLD} but resting on fewer than {MIN_CELL} units, so most "
            "likely small-sample noise (listed so nothing is hidden):\n\n"
            + "\n".join(f"- {x}" for x in f.small_cell_flags) if f.small_cell_flags else ""
        ),
    ]
    return "\n\n".join(head + body) + "\n"


def _collect(r: EvaluationResult) -> _Formatter:
    f = _Formatter()
    _section_agreement(f, r, {})
    _section_stratified(f, r, {})
    _section_detection(f, r)
    _section_abstention(f, r)
    _section_subgroups(f, r)
    return f


def tripwire(r: EvaluationResult) -> list[str]:
    """Estimates on >= MIN_CELL units that trip the 0.95 check: the real alarm."""
    return _collect(r).flags


def small_cell_flags(r: EvaluationResult) -> list[str]:
    """Estimates above 0.95 on fewer than MIN_CELL units: reported, most likely noise."""
    return _collect(r).small_cell_flags

