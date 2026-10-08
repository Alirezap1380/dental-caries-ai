# Evaluation report: synthetic_e0_leaky

> **E0: NAIVE IMAGE-LEVEL SPLIT.** This run deliberately reproduces the leaky protocol. Patients cross partitions (section 1), so internal-test numbers are inflated by construction. The tripwire below is expected to fire, and the external numbers are the honest ones.

> **SYNTHETIC DATA.** Every number below is a property of the generator (`dcai/data/synthetic.py`, `dcai/simulate.py`), not of any trained model. This run exists to show that the harness composes end to end.

dcai 0.1.0 · seed 20261009 · 2000 bootstrap resamples · scale `dentex_depth`

### How to read this report

- **Every interval is a 95% percentile bootstrap, resampled by patient, uncorrected for multiple comparisons, and exploratory.** The only exceptions are the headline claims, which carry Bonferroni-corrected intervals.
- ⚠ marks any performance estimate above 0.95. At these sample sizes, explain it (look for leakage first) before believing it.
- Tooth-level metrics use the strict-majority reader consensus as reference. Lesion-level detection is scored against each reader separately.

### Leakage tripwire

- ⚠ internal AUC vs sound (caries): 0.984 [0.980, 0.988]
- ⚠ internal AUC vs sound (deep_caries): 0.992 [0.989, 0.994]
- ⚠ internal specificity: 0.979 [0.976, 0.982]
- ⚠ external specificity: 1.000 [0.999, 1.000]
- ⚠ internal test r1 lesion sensitivity (deep_caries): 0.955 [0.930, 0.976]
- ⚠ internal test r2 lesion sensitivity (deep_caries): 0.968 [0.947, 0.986]
- ⚠ internal test r3 lesion sensitivity (deep_caries): 0.974 [0.955, 0.991]
- ⚠ internal test r4 lesion sensitivity (deep_caries): 0.953 [0.929, 0.974]
- ⚠ internal test specificity among accepted: 1.000 [1.000, 1.000]
- ⚠ internal test specificity, no abstention: 0.979 [0.976, 0.982]
- ⚠ external specificity among accepted: 1.000 [1.000, 1.000]
- ⚠ external specificity, no abstention: 1.000 [0.999, 1.000]

## Headline claims (pre-specified)

The three claims fixed in advance, with **Bonferroni-corrected 98.33% intervals** (family-wise α = 0.05). Sensitivities are tooth-level at the frozen operating point. The external gaps compare independent samples.

| claim | estimate [98.33% CI] | verdict |
|---|---|---|
| Depth gap, internal test: sensitivity(deep_caries) - sensitivity(caries) | 0.097 [0.040, 0.153] | excludes 0 |
| External drop in caries sensitivity: internal - external | 0.737 [0.673, 0.798] | excludes 0 |
| External drop in deep_caries sensitivity: internal - external | 0.815 [0.745, 0.881] | excludes 0 |

## 1. Data and splits

| set | images | patients | teeth | max images / patient | sites | multi-reader images | patient ids | tooth inventory |
|---|---|---|---|---|---|---|---|---|
| internal test | 420 | 349 | 11827 | 3 | site_a, site_b, site_c | 420 | provided 420 | annotated 420 |
| external | 379 | 250 | 10665 | 2 | site_ext | 379 | provided 379 | annotated 379 |

Internal set, **naive image-level split (E0)**. **THIS SPLIT LEAKS: 525 of 700 patients (75.0%) appear in more than one partition, and 97.4% of test images belong to a patient the model saw in train or validation.** Every internal-test number below is inflated by that contamination.

| partition | images | patients |
|---|---|---|
| train | 1260 | 648 |
| val | 420 | 343 |
| test | 420 | 349 |

A naive image-level split of the same data would put **525 of 700 patients (75.0%)** in more than one partition. That is the contamination an image-level split carries. E0 reproduces it deliberately.

## 2. Reader agreement (internal test, tooth level)

11827 teeth in 420 images, readers r1, r2, r3, r4. The unit is the tooth (FDI), so readers are paired with no box-matching step. Cohen's kappa uses linear weights on the ordinal depth scale.

- Krippendorff's alpha (ordinal, handles unread images): **0.667 [0.650, 0.682]**

- Fleiss' kappa: not computable: needs a complete design, but some readers did not read every image; use Krippendorff's alpha

Individual reader pairs (how much humans disagree; this is *not* the ceiling):

| reader A | reader B | teeth | kappa |
|---|---|---|---|
| r1 | r2 | 10546 | 0.633 [0.608, 0.656] |
| r1 | r3 | 10453 | 0.652 [0.629, 0.672] |
| r1 | r4 | 10730 | 0.620 [0.599, 0.641] |
| r2 | r3 | 10692 | 0.618 [0.596, 0.640] |
| r2 | r4 | 10969 | 0.610 [0.587, 0.632] |
| r3 | r4 | 10876 | 0.597 [0.575, 0.619] |

**Model against the human ceiling, across thresholds.** The ceiling for reader X is the agreement between X and the leave-one-out consensus of the other readers, so a consensus-trained model is compared with a consensus (matched variance). The band is the mean ceiling over readers. Kappa at one threshold would mix "worse than readers" with "pinned to a threshold readers don't use", so the model is swept.

- Human ceiling (mean LOO-consensus κ): **0.686 [0.669, 0.701]**

- Model at the operating point (threshold 0.956): 0.626 [0.605, 0.645]

- Model at its κ-optimal threshold (0.450): 0.649 [0.634, 0.665]. This threshold was chosen on these data, so the value is optimistic: a diagnostic, not a deployable number.

- Does any threshold reach the band? **no**: at no threshold does the model agree with readers as well as their own consensus does.

**The operating point and the κ-optimal threshold differ a lot.** Moving the threshold from 0.956 to 0.450 raises κ by 0.023 [0.015, 0.033] (paired). At the deployed threshold the model calls lesions far more readily than the readers do, and that part of the shortfall is the price of the binding constraint. It is not the whole story: even at its best threshold the model stays below the band, so the rest of the gap is the model.

![Model kappa across thresholds](figures/kappa_sweep.svg)

Per reader, at the κ-optimal threshold (0.450), the model's most charitable setting. If it does not exceed a reader's ceiling here, it does not exceed it anywhere.

| reader | teeth | model κ | ceiling κ (LOO consensus) | mean individual κ | model − ceiling | verdict |
|---|---|---|---|---|---|---|
| r1 | 11067 | 0.665 [0.644, 0.687] | 0.703 [0.682, 0.724] | 0.635 [0.617, 0.651] | -0.038 [-0.062, -0.014] | below ceiling |
| r2 | 11306 | 0.647 [0.626, 0.669] | 0.687 [0.665, 0.708] | 0.620 [0.602, 0.637] | -0.039 [-0.060, -0.020] | below ceiling |
| r3 | 11213 | 0.664 [0.645, 0.683] | 0.679 [0.657, 0.699] | 0.622 [0.605, 0.639] | -0.014 [-0.036, 0.009] | within ceiling |
| r4 | 11490 | 0.621 [0.599, 0.643] | 0.675 [0.655, 0.696] | 0.609 [0.592, 0.626] | -0.054 [-0.077, -0.032] | below ceiling |

*Exceeding the ceiling is not a success.* It means reader mimicry, leakage, or a model that averages label noise better than an (N−1)-reader panel. It needs explaining.

## 3. Depth-stratified tooth-level performance

Operating point fitted on internal *validation* patients only. Deployment role: **second reader**; binding constraint: **sensitivity ≥ 0.80** (achieved 0.801 on validation) at threshold **0.956** on P(lesion). It is frozen for internal test and external.

> Rationale on record: Deployment role: second reader. A dentist adjudicates every flag before anything is done to the tooth. Asymmetry: caries inverts the usual screening asymmetry. A missed early lesion is found later and restored instead of arrested, a bounded and partly recoverable harm. A false positive that is acted on drills sound tooth structure and starts the restorative cycle. In this role a false positive costs review time rather than tooth structure (residual risk: a flag can anchor the reviewer towards treatment), so sensitivity may bind. Target: sensitivity >= 0.80 against the consensus reference. This is provisional, to be re-anchored to the readers' own sensitivity once the agreement analysis runs on real data: a second reader far more sensitive than the readers mostly adds flags they will overrule. If the role changes to autonomous triage (a flag drives treatment), specificity becomes the binding constraint, its target is set first, and sensitivity becomes the reported cost.

Reference standard: strict-majority consensus of each image's readers, per tooth.

| depth | n (internal) | sensitivity (internal) | AUC vs sound (internal) | n (external) | sensitivity (external) | AUC vs sound (external) |
|---|---|---|---|---|---|---|
| **caries** | 591 | 0.760 [0.709, 0.809] | 0.984 [0.980, 0.988] ⚠ | 483 | 0.023 [0.011, 0.036] | 0.739 [0.715, 0.765] |
| **deep_caries** | 364 | 0.857 [0.804, 0.908] | 0.992 [0.989, 0.994] ⚠ | 331 | 0.042 [0.023, 0.063] | 0.874 [0.854, 0.894] |
| sound teeth: specificity | 10872 | 0.979 [0.976, 0.982] ⚠ | — | 9851 | 1.000 [0.999, 1.000] ⚠ | — |
| *pooled sensitivity (shown only to expose what pooling hides)* | 955 | 0.797 [0.749, 0.842] | — | 814 | 0.031 [0.020, 0.043] | — |

![Sensitivity by lesion depth](figures/depth_sensitivity.svg)

## 4. Lesion-level detection, per reader

Boxes matched to each reader's own lesions (IoU ≥ 0.5); sensitivity and FP/image at box score ≥ 0.3. AP per depth ignores lesions of the other depth, and background false positives count against every depth.

**internal test**

| reader | images | sens caries | sens deep_caries | FP / image | AP caries | AP deep_caries | *pooled AP* |
|---|---|---|---|---|---|---|---|
| r1 | 393 | 0.754 [0.714, 0.790] | 0.955 [0.930, 0.976] ⚠ | 0.814 [0.705, 0.929] | 0.550 [0.507, 0.602] | 0.795 [0.754, 0.835] | 0.740 [0.708, 0.772] |
| r2 | 401 | 0.759 [0.718, 0.798] | 0.968 [0.947, 0.986] ⚠ | 1.002 [0.889, 1.122] | 0.501 [0.456, 0.555] | 0.774 [0.733, 0.818] | 0.724 [0.694, 0.758] |
| r3 | 398 | 0.645 [0.606, 0.682] | 0.974 [0.955, 0.991] ⚠ | 0.729 [0.626, 0.844] | 0.500 [0.463, 0.546] | 0.841 [0.805, 0.878] | 0.700 [0.673, 0.730] |
| r4 | 408 | 0.798 [0.761, 0.832] | 0.953 [0.929, 0.974] ⚠ | 1.152 [1.024, 1.279] | 0.478 [0.432, 0.535] | 0.726 [0.686, 0.774] | 0.711 [0.680, 0.746] |

**external**

| reader | images | sens caries | sens deep_caries | FP / image | AP caries | AP deep_caries | *pooled AP* |
|---|---|---|---|---|---|---|---|
| r1 | 368 | 0.348 [0.307, 0.391] | 0.680 [0.633, 0.723] | 4.117 [3.915, 4.315] | 0.092 [0.070, 0.121] | 0.292 [0.246, 0.343] | 0.260 [0.228, 0.294] |
| r2 | 365 | 0.361 [0.315, 0.406] | 0.658 [0.605, 0.706] | 4.140 [3.930, 4.348] | 0.097 [0.073, 0.129] | 0.265 [0.219, 0.314] | 0.253 [0.220, 0.288] |
| r3 | 357 | 0.300 [0.269, 0.336] | 0.684 [0.638, 0.729] | 4.025 [3.798, 4.242] | 0.089 [0.072, 0.110] | 0.297 [0.250, 0.349] | 0.243 [0.214, 0.273] |
| r4 | 366 | 0.419 [0.374, 0.470] | 0.665 [0.619, 0.711] | 4.161 [3.956, 4.361] | 0.127 [0.100, 0.160] | 0.229 [0.190, 0.274] | 0.259 [0.227, 0.294] |

## 5. Calibration (tooth level)

Per-tooth P(lesion) against the consensus reference. Box scores are deliberately not calibrated, because a detector chooses how many boxes to emit. The noise floor is the 95th percentile of ECE for a perfectly calibrated model at this n. Slope < 1 means overconfident; intercept < 0 means it over-predicts.

| set | teeth | prevalence | mean P | ECE (quantile bins) | ECE noise floor | exceeds floor | Brier | slope | intercept |
|---|---|---|---|---|---|---|---|---|---|
| internal test | 11827 | 0.081 | 0.112 | 0.031 [0.028, 0.035] | 0.004 | yes | 0.029 [0.026, 0.032] | 0.746 [0.715, 0.782] | -1.768 [-1.960, -1.609] |
| external | 10665 | 0.076 | 0.168 | 0.094 [0.089, 0.099] | 0.010 | yes | 0.082 [0.078, 0.086] | 0.683 [0.631, 0.738] | -1.445 [-1.544, -1.348] |

![Reliability diagram](figures/reliability.svg)

## 6. Abstention at the operating point

Teeth within ±0.864 of the threshold are referred to a clinician; the band was sized on validation for 85.0% coverage. *Missed positive rate* is lesions the system cleared without referral, as a share of all lesions. That is the number that matters clinically.

| set | coverage | sens (accepted) | spec (accepted) | missed positive rate | sens (no abstention) | spec (no abstention) | patients shared with fitting set |
|---|---|---|---|---|---|---|---|
| internal test | 0.853 [0.839, 0.867] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] ⚠ | 0.013 [0.005, 0.021] | 0.797 [0.749, 0.842] | 0.979 [0.976, 0.982] ⚠ | **153** |
| external | 0.540 [0.530, 0.550] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] ⚠ | 0.170 [0.142, 0.197] | 0.031 [0.020, 0.043] | 1.000 [0.999, 1.000] ⚠ | 0 |

## 7. Subgroups

Depth-stratified within every subgroup: age is confounded with lesion depth, so a pooled per-age sensitivity would invent an age effect.

**internal test, by age** (metadata coverage 100.0%)

| age | teeth | patients | sens caries | sens deep_caries | note |
|---|---|---|---|---|---|
| 18-34 | 3463 | 94 | 0.830 [0.735, 0.913] | 0.828 [0.700, 0.943] |  |
| 35-49 | 3161 | 90 | 0.763 [0.655, 0.858] | 0.846 [0.720, 0.945] |  |
| 50-64 | 2866 | 92 | 0.751 [0.643, 0.851] | 0.904 [0.824, 0.975] |  |
| 65+ | 2337 | 73 | 0.664 [0.533, 0.783] | 0.848 [0.743, 0.931] |  |

**internal test, by sex** (metadata coverage 100.0%)

| sex | teeth | patients | sens caries | sens deep_caries | note |
|---|---|---|---|---|---|
| female | 5637 | 168 | 0.745 [0.663, 0.817] | 0.868 [0.792, 0.933] |  |
| male | 6190 | 181 | 0.774 [0.704, 0.842] | 0.849 [0.773, 0.915] |  |

**external, by age**: no age metadata on any of 10665 units: not computable on this data.

**external, by sex**: no sex metadata on any of 10665 units: not computable on this data.

## 8. Not computable on this data

- internal: E0 naive image-level split. Patient leakage is deliberate here, so every internal-test number is inflated by construction
- external, subgroups by age: no age metadata on any of 10665 units: not computable on this data
- external, subgroups by sex: no sex metadata on any of 10665 units: not computable on this data
- internal test: Fleiss' kappa not computable: needs a complete design, but some readers did not read every image; use Krippendorff's alpha
