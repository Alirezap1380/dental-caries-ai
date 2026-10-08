"""End-to-end evaluation: records + model outputs in, one `EvaluationResult` out.

The interface is deliberately the one real data will use. Loaders produce
`RadiographRecord`s, a model produces per-tooth `ToothPrediction`s and box
`Detection`s, and this module composes the harness over them. The synthetic run
feeds it simulated inputs. Nothing here knows the difference.

Order of operations, and why:
1. Patient-level split of the internal set (`split: patient`, the default). The
   naive split is built too, only to *measure* how much it would have leaked.
   `split: image_naive` runs experiment E0 instead: the naive split is the one
   actually used. Its leakage is then reported as part of the result, and the
   0.95 tripwire is expected to fire.
2. Operating point fitted on internal *validation* patients only.
3. That frozen operating point is evaluated on internal test and on the external
   set. Nothing is re-tuned on either (rule 4: the gap is a result).
4. Agreement and the ceiling are computed on internal test teeth.
5. Headline claims (pre-specified in config) get Bonferroni-corrected intervals.
   Every other interval in the report is uncorrected and exploratory.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dcai.data.schema import DatasetSummary, RadiographRecord, summarise
from dcai.data.splits import (
    LeakageReport,
    NoLeakageToMeasureWarning,
    Split,
    SplitKind,
    naive_image_split,
    patient_split,
)
from dcai.eval.abstention import (
    OperatingPoint,
    SelectiveResult,
    evaluate_operating_point,
    fit_operating_point,
)
from dcai.eval.agreement import (
    CeilingComparison,
    KappaSweep,
    PairAgreement,
    fleiss_agreement,
    kappa_sweep,
    krippendorff_agreement,
    model_vs_readers,
    pairwise_agreement,
)
from dcai.eval.bootstrap import Estimate, difference
from dcai.eval.calibration import CalibrationReport, calibration_report
from dcai.eval.detection import (
    APReport,
    Detection,
    DetectionReport,
    detection_report,
    match_lesions,
    stratified_average_precision,
)
from dcai.eval.ratings import tooth_level_ratings
from dcai.eval.scales import DENTEX_DEPTH, ICCMS_STAGE, OrdinalScale
from dcai.eval.stratified import StratifiedReport, stratified_report
from dcai.eval.subgroups import DEFAULT_AGE_EDGES, SubgroupReport, age_bands, subgroup_report
from dcai.eval.units import ToothPrediction, ToothTable, tooth_table

SCALES = {s.name: s for s in (DENTEX_DEPTH, ICCMS_STAGE)}
MODEL = "model"


# --- config -------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPointConfig:
    deployment_role: str  # second_reader | autonomous_triage
    constraint: str  # sensitivity | specificity: the error rate held to `target`
    target: float
    coverage_target: float
    rationale: str


@dataclass(frozen=True)
class SyntheticConfig:
    internal_patients: int
    external_patients: int
    external_demographics: bool
    # E0 only: the simulated model memorises patients it saw in training, as an
    # overfit network does. Under a naive split their other images reach test.
    memorize_train: bool = False
    images_per_patient: tuple[int, int] = (1, 2)


@dataclass(frozen=True)
class StudyConfig:
    name: str
    seed: int
    n_boot: int
    scale: OrdinalScale
    splits: Mapping[str, float]
    operating_point: OperatingPointConfig
    iou_threshold: float
    box_score_threshold: float
    headline_alpha: float
    age_edges: tuple[int, ...] = DEFAULT_AGE_EDGES
    synthetic: SyntheticConfig | None = None
    split_kind: SplitKind = SplitKind.PATIENT

    def __post_init__(self) -> None:
        if self.split_kind not in (SplitKind.PATIENT, SplitKind.IMAGE_NAIVE):
            raise ValueError(f"split must be patient or image_naive, got {self.split_kind}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        raw = yaml.safe_load(Path(path).read_text())
        syn = raw.get("synthetic")
        return cls(
            name=raw["name"],
            seed=int(raw["seed"]),
            n_boot=int(raw["n_boot"]),
            scale=SCALES[raw["scale"]],
            splits={k: float(v) for k, v in raw["splits"].items()},
            operating_point=OperatingPointConfig(**raw["operating_point"]),
            iou_threshold=float(raw["detection"]["iou_threshold"]),
            box_score_threshold=float(raw["detection"]["score_threshold"]),
            headline_alpha=float(raw["headline_alpha"]),
            age_edges=tuple(raw.get("age_edges", DEFAULT_AGE_EDGES)),
            synthetic=None if syn is None else SyntheticConfig(**{
                **syn, "images_per_patient": tuple(syn.get("images_per_patient", (1, 2))),
            }),
            split_kind=SplitKind(raw.get("split", "patient")),
        )


# --- results ------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetInputs:
    name: str
    records: Sequence[RadiographRecord]
    predictions: Sequence[ToothPrediction]
    detections: Sequence[Detection]


@dataclass(frozen=True)
class SplitResult:
    kind: SplitKind
    images: dict[str, int]
    patients: dict[str, int]
    patient_leakage: LeakageReport
    naive_leakage: LeakageReport
    naive_note: str | None
    # Of the split actually used: test images whose patient also has an image in
    # train or validation. 0 for a patient split by construction.
    test_images_with_seen_patient: float


@dataclass(frozen=True)
class AgreementResult:
    n_teeth: int
    n_images: int
    readers: tuple[str, ...]
    weights: str
    krippendorff: Estimate | None  # None with fewer than two readers
    fleiss: Estimate | None
    fleiss_reason: str | None
    pairwise: tuple[PairAgreement, ...]
    ceiling: tuple[CeilingComparison, ...]  # per reader, at the kappa-optimal threshold
    ceiling_reason: str | None
    sweep: KappaSweep | None


@dataclass(frozen=True)
class ReaderDetection:
    reader: str
    n_images: int
    report: DetectionReport
    ap: APReport


@dataclass(frozen=True)
class SetResult:
    name: str
    summary: DatasetSummary
    n_teeth: int
    stratified: StratifiedReport
    calibration: CalibrationReport
    selective: SelectiveResult
    subgroups: tuple[SubgroupReport, ...]
    detection: tuple[ReaderDetection, ...]


@dataclass(frozen=True)
class HeadlineClaim:
    claim: str
    estimate: Estimate | None  # at the Bonferroni-corrected level
    reason: str | None = None  # why it could not be computed


@dataclass(frozen=True)
class EvaluationResult:
    config: StudyConfig
    splits: SplitResult
    operating_point: OperatingPoint
    agreement: AgreementResult
    internal: SetResult
    external: SetResult | None  # None: no external dataset (rule 4 then has nothing behind it)
    headline: tuple[HeadlineClaim, ...]
    not_computable: tuple[str, ...] = field(default_factory=tuple)


# --- composition --------------------------------------------------------------------


def _restrict(inputs: DatasetInputs, records: Sequence[RadiographRecord], name: str):
    ids = {r.image_id for r in records}
    return DatasetInputs(
        name,
        list(records),
        [p for p in inputs.predictions if p.image_id in ids],
        [d for d in inputs.detections if d.image_id in ids],
    )


def _table(inputs: DatasetInputs, scale: OrdinalScale) -> ToothTable:
    return tooth_table(inputs.records, inputs.predictions, scale=scale)


def _inventoried(inputs: DatasetInputs, notes: list[str]) -> DatasetInputs:
    """Restrict to images with a tooth inventory; tooth-level metrics need one.

    Lesion-level detection does not, so it keeps every image. The exclusion is
    recorded rather than silent.
    """
    keep = [r for r in inputs.records if r.teeth_present is not None]
    if not keep:
        raise ValueError(f"{inputs.name}: no image has a tooth inventory; "
                         "tooth-level metrics cannot be computed")
    if len(keep) < len(inputs.records):
        notes.append(
            f"{inputs.name}: {len(inputs.records) - len(keep)} of {len(inputs.records)} images "
            "have no human tooth inventory, so tooth-level metrics use the other "
            f"{len(keep)} (lesion-level detection uses all)")
    return _restrict(inputs, keep, inputs.name)


def _evaluate_set(
    inputs: DatasetInputs,
    op: OperatingPoint,
    cfg: StudyConfig,
    seed: int,
    notes: list[str],
    *,
    allow_patient_overlap: bool = False,
) -> SetResult:
    t = _table(_inventoried(inputs, notes), cfg.scale)
    boot = {"seed": seed, "n_boot": cfg.n_boot}
    stratified = stratified_report(t.reference, t.p_lesion, t.groups, scale=cfg.scale,
                                   threshold=op.threshold, **boot)
    for s in stratified.strata:
        if s.sensitivity is None:
            notes.append(f"{inputs.name}: no '{s.name}' lesions in the reference standard")

    subgroups = []
    for attribute, values in (("age", age_bands(t.ages, cfg.age_edges)), ("sex", t.sexes)):
        sg = subgroup_report(values, t.reference, t.p_lesion, t.groups, attribute=attribute,
                             scale=cfg.scale, threshold=op.threshold, **boot)
        if sg.not_computable:
            notes.append(f"{inputs.name}, subgroups by {attribute}: {sg.not_computable}")
        subgroups.append(sg)

    detection = []
    for reader in sorted(set().union(*(r.readers for r in inputs.records))):
        read = [r for r in inputs.records if reader in r.readers]
        ids = {r.image_id for r in read}
        m = match_lesions(read, [d for d in inputs.detections if d.image_id in ids],
                          reader=reader, scale=cfg.scale, iou_threshold=cfg.iou_threshold)
        detection.append(ReaderDetection(
            reader, len(read),
            detection_report(m, threshold=cfg.box_score_threshold, **boot),
            stratified_average_precision(m, **boot),
        ))

    return SetResult(
        name=inputs.name,
        summary=summarise(inputs.records),
        n_teeth=t.n,
        stratified=stratified,
        calibration=calibration_report(t.p_lesion, t.has_lesion, t.groups, **boot),
        selective=evaluate_operating_point(op, t.p_lesion, t.has_lesion, t.groups,
                                           allow_patient_overlap=allow_patient_overlap, **boot),
        subgroups=tuple(subgroups),
        detection=tuple(detection),
    )


def _agreement(
    inputs: DatasetInputs, op: OperatingPoint, cfg: StudyConfig, notes: list[str]
) -> AgreementResult:
    inputs = _inventoried(inputs, [])  # the exclusion is already noted by _evaluate_set
    humans = tooth_level_ratings(inputs.records, cfg.scale)
    table = _table(inputs, cfg.scale)
    # Both are built in inventory order; a mismatch would silently mis-pair readers and model.
    if humans.item_ids != table.unit_ids:
        raise AssertionError("tooth units out of alignment")
    boot = {"seed": cfg.seed, "n_boot": cfg.n_boot}
    weights = "linear"
    if len(humans.raters) < 2:
        reason = (f"one reader only ({', '.join(humans.raters)}): inter-observer agreement and "
                  "the human ceiling are not computable on this data")
        notes.append(f"{inputs.name}: {reason}")
        return AgreementResult(
            n_teeth=len(humans.item_ids), n_images=len(inputs.records), readers=humans.raters,
            weights=weights, krippendorff=None, fleiss=None, fleiss_reason=reason,
            pairwise=(), ceiling=(), ceiling_reason=reason, sweep=None,
        )

    fleiss = fleiss_reason = None
    try:
        fleiss = fleiss_agreement(humans, **boot)
    except ValueError as e:
        fleiss_reason = f"not computable: {e}"
        notes.append(f"{inputs.name}: Fleiss' kappa {fleiss_reason}")

    ceiling: tuple[CeilingComparison, ...] = ()
    ceiling_reason = sweep = None
    try:
        sweep = kappa_sweep(humans, table.predicted_grade, operating_threshold=op.threshold,
                            weights=weights, **boot)
        # Per-reader mimicry check at the model's most charitable threshold: if it
        # does not exceed the ceiling there, it does not exceed it anywhere.
        ceiling = tuple(model_vs_readers(
            humans.with_rater(MODEL, table.model_grades(sweep.best_threshold)), MODEL,
            weights=weights, **boot,
        ))
    except ValueError as e:
        ceiling_reason = str(e)
        notes.append(f"{inputs.name}: agreement ceiling not computable: {e}")

    return AgreementResult(
        n_teeth=len(humans.item_ids),
        n_images=len(inputs.records),
        readers=humans.raters,
        weights=weights,
        krippendorff=krippendorff_agreement(humans, metric="ordinal", **boot),
        fleiss=fleiss,
        fleiss_reason=fleiss_reason,
        pairwise=tuple(pairwise_agreement(humans, weights=weights, **boot)),
        ceiling=ceiling,
        ceiling_reason=ceiling_reason,
        sweep=sweep,
    )


def _headline(internal: SetResult, external: SetResult | None, cfg: StudyConfig):
    """Pre-specified claims, Bonferroni-corrected across the family."""
    level = 1.0 - cfg.headline_alpha / 3
    names = cfg.scale.categories
    shallow, deepest = names[1], names[-1]

    def gap(name: str) -> Estimate | None:
        if external is None:
            return None
        a = internal.stratified.stratum(name).sensitivity
        b = external.stratified.stratum(name).sensitivity
        return None if a is None or b is None else difference(a, b, paired=False)

    claims = [
        (f"Depth gap, internal test: sensitivity({deepest}) - sensitivity({shallow})",
         internal.stratified.depth_gap),
        (f"External drop in {shallow} sensitivity: internal - external", gap(shallow)),
        (f"External drop in {deepest} sensitivity: internal - external", gap(deepest)),
    ]
    return tuple(
        HeadlineClaim(text, est.at_level(level)) if est is not None
        else HeadlineClaim(text, None, "no external dataset" if external is None and "External"
                           in text else "a stratum is empty on one side")
        for text, est in claims
    )


def make_split(records: Sequence[RadiographRecord], cfg: StudyConfig) -> Split:
    """The internal split `cfg` asks for. Deterministic in (records, seed)."""
    if cfg.split_kind is SplitKind.IMAGE_NAIVE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoLeakageToMeasureWarning)  # measured below
            return naive_image_split(records, seed=cfg.seed, fractions=cfg.splits)
    return patient_split(records, seed=cfg.seed, fractions=cfg.splits)


def evaluate(
    internal: DatasetInputs, external: DatasetInputs | None, cfg: StudyConfig
) -> EvaluationResult:
    notes: list[str] = []
    split = make_split(internal.records, cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", NoLeakageToMeasureWarning)
        naive = naive_image_split(internal.records, seed=cfg.seed, fractions=cfg.splits)
    naive_note = str(caught[0].message) if caught else None
    e0 = cfg.split_kind is SplitKind.IMAGE_NAIVE
    if e0:
        notes.append(
            "internal: E0 naive image-level split. Patient leakage is deliberate here, so every "
            "internal-test number is inflated by construction"
        )

    val = _restrict(internal, split["val"], "internal validation")
    test = _restrict(internal, split["test"], "internal test")
    seen = {r.group_key for name in ("train", "val") if name in split.partitions
            for r in split[name]}
    test_seen = sum(r.group_key in seen for r in split["test"]) / len(split["test"])

    val_table = _table(_inventoried(val, notes), cfg.scale)
    opc = cfg.operating_point
    op = fit_operating_point(
        val_table.p_lesion, val_table.has_lesion, val_table.groups,
        deployment_role=opc.deployment_role, constraint=opc.constraint, target=opc.target,
        coverage_target=opc.coverage_target, rationale=opc.rationale,
    )

    internal_result = _evaluate_set(test, op, cfg, cfg.seed, notes, allow_patient_overlap=e0)
    # A different seed: internal and external are independent samples, and the
    # independent-difference CI for the headline gaps requires it.
    external_result = None
    if external is None:
        notes.append("no external dataset: rule 4 (external validation is the real number) "
                     "has nothing behind it yet, so every number here is internal")
    else:
        external_result = _evaluate_set(external, op, cfg, cfg.seed + 1, notes)
    return EvaluationResult(
        config=cfg,
        splits=SplitResult(
            kind=cfg.split_kind,
            images={k: len(v) for k, v in split.partitions.items()},
            patients={k: len({r.group_key for r in v}) for k, v in split.partitions.items()},
            patient_leakage=split.leakage,
            naive_leakage=naive.leakage,
            naive_note=naive_note,
            test_images_with_seen_patient=test_seen,
        ),
        operating_point=op,
        agreement=_agreement(test, op, cfg, notes),
        internal=internal_result,
        external=external_result,
        headline=_headline(internal_result, external_result, cfg),
        not_computable=tuple(notes),
    )


def synthetic_inputs(cfg: StudyConfig) -> tuple[DatasetInputs, DatasetInputs]:
    """Generate the synthetic internal and external sets described by `cfg.synthetic`."""
    from dcai.data.synthetic import make_reader_study
    from dcai.simulate import EXTERNAL, INTERNAL, simulate_model

    if cfg.synthetic is None:
        raise ValueError("config has no synthetic section")
    s = cfg.synthetic
    internal = make_reader_study(n_patients=s.internal_patients, seed=cfg.seed,
                                 images_per_patient=s.images_per_patient)
    external = make_reader_study(
        n_patients=s.external_patients, seed=cfg.seed + 1, dataset="synthetic_external",
        sites=("site_ext",), demographics=s.external_demographics,
    )
    memorized: frozenset[str] = frozenset()
    if s.memorize_train:
        memorized = frozenset(r.group_key for r in make_split(internal.records, cfg)["train"])
    ip, idet = simulate_model(internal, INTERNAL, seed=cfg.seed, memorized=memorized)
    ep, edet = simulate_model(external, EXTERNAL, seed=cfg.seed + 1)
    return (
        DatasetInputs("internal", internal.records, ip, idet),
        DatasetInputs("external", external.records, ep, edet),
    )

