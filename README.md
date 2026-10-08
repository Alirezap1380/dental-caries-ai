# dental-caries-ai

Caries detection on dental radiographs, with an honest uncertainty budget.

The goal is not a high accuracy number. Most published dental-imaging classifiers
report inflated metrics because of (a) image-level rather than patient-level
splits, (b) single-source or source-confounded data, and (c) pooled metrics that
hide failure on the clinically important subgroup. This project reproduces the
naive result, breaks it, and rebuilds it honestly.

## Methodology rules (enforced in code)

1. **Patient-level splits only.** Every split runs a leakage check at construction
   and raises `PatientLeakageError` (explicit raise, so it survives `python -O`).
   The naive image-level split exists only for E0, and it measures its own leak.
2. **Never pool metrics across lesion stage.** Unstaged lesions are `stage=None`,
   distinct from `CariesStage.NONE` (sound).
3. **Inter-observer agreement is the ceiling.** Annotations stay per-reader;
   records list every reader, including those who found nothing.
4. **External validation is the real number.**
5. **No metric without a patient-grouped bootstrap CI.**

## Operating point: deployment role first

Caries inverts the usual screening asymmetry. A false negative on an early lesion
means it is caught later and restored instead of arrested: a bounded, partly
recoverable harm. A false positive that is acted on means drilling a sound tooth:
permanent loss of structure, and the start of the restorative cycle (each
replacement larger than the last, ending in a crown or extraction).
Over-treatment of early lesions is a documented harm, so maximising sensitivity is
not automatically correct here. The operating point is therefore specified in
this order, and `OperatingPoint` enforces the first two steps in code:

1. **Deployment role.**
   - `second_reader`: a dentist adjudicates every flag.
   - `autonomous_triage`: a flag drives treatment without review.
2. **The asymmetry that follows.**
   - As a second reader, a false positive costs review time. A residual risk
     remains: a flag can anchor the reviewer towards treatment. Higher
     sensitivity is defensible.
   - Under autonomous triage, a false positive costs tooth structure, so
     specificity must bind. The code rejects any other choice.
3. **Chosen target.** For the current runs, `second_reader` with sensitivity ≥ 0.80
   against the consensus reference. This is provisional: it will be re-anchored to
   the readers' own sensitivity once agreement runs on real data, because a second
   reader far more sensitive than the readers mostly adds flags they will overrule.
4. **If the role changes** to autonomous triage, specificity becomes the binding
   constraint and is set first. Sensitivity becomes the cost the report states.

The rationale lives in `configs/*.yaml` and is printed verbatim in every report.
The kappa sweep (report section 2) shows what the chosen threshold costs in
agreement with readers, separately from how well the model ranks teeth.

## Known risk: provenance of the tooth inventory

Every tooth-level number (agreement, calibration, stratified sensitivity and
specificity) has the tooth inventory `teeth_present` as its denominator. Two ways
that can silently go wrong:

- **An inventory rebuilt from diagnosis boxes.** The union of readers' boxes
  contains only diseased teeth, so sound teeth vanish and specificity and kappa
  become meaningless. Guarded: `ratings.check_inventory` refuses an inventory in
  which almost every tooth carries a finding. Records must also declare
  `teeth_present_source`.
- **DENTEX's subsets may not line up.** DENTEX is released as separately
  annotated subsets: quadrant only, quadrant + enumeration, and the fully
  annotated quadrant + enumeration + diagnosis set. Our understanding is that
  diagnosis boxes carry the FDI number of diseased teeth only, and that a full
  per-tooth enumeration exists only in the enumeration subset. If that subset does
  not overlap the diagnosis subset, the images we evaluate on have no human tooth
  inventory. Tooth-level metrics would then need an enumeration model first
  (`InventorySource.PREDICTED`), which puts model error in every denominator.

**Before trusting any tooth-level number on real data, check against the files:**

1. Do image ids in the enumeration and diagnosis subsets intersect, and by how much?
2. On overlapping images, do enumeration labels cover sound teeth, i.e. is the
   inventory a full dentition?
3. Does `check_inventory` pass, and what fraction of teeth are unflagged?
4. If the inventory is predicted, report enumeration accuracy alongside every
   tooth-level metric.

## Status

- [x] `data/`: unified schema (incl. tooth inventory), patient-level splits with enforced leakage checks
- [x] `eval/`: tooth-level agreement (Cohen, Fleiss, Krippendorff; LOO-consensus ceiling),
  depth-stratified sensitivity/AUC/AP, tooth-level calibration (with ECE noise floor),
  abstention (rationale-required operating point), subgroups, patient-grouped bootstrap
- [x] end-to-end pipeline + Markdown report with headline Bonferroni, a kappa threshold sweep
  and a 0.95 leakage tripwire, verified on synthetic data in two configurations:
  - clean: [`results/synthetic/report.md`](results/synthetic/report.md), where the tripwire stays quiet
  - deliberately leaky E0: [`results/synthetic_e0_leaky/report.md`](results/synthetic_e0_leaky/report.md),
    where the tripwire fires (also a permanent test)
- [ ] `probes/recover_groups.py`: recover patient and site structure DENTEX may not ship
- [ ] `configs/label_mapping_tufts_dentex.yaml`
- [ ] loaders (blocked on data access)
- [ ] baselines

## Run

```bash
.venv/bin/python scripts/run_experiment.py --config configs/synthetic.yaml --out results/synthetic
.venv/bin/python scripts/run_experiment.py --config configs/synthetic_e0_leaky.yaml --out results/synthetic_e0_leaky
```

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The evaluation harness needs no GPU stack. Torch etc. live behind the `[train]` extra.
