# Group recovery report

> **SYNTHETIC IMAGES.** Every number below is a property of `dcai/data/synthetic_images.py`, which makes repeat visits near-identical and sites cleanly distinct: an easy case, meant to show the probes compose.

Intervals are 95% patient-grouped bootstrap intervals (grouped by *recovered* patient), uncorrected and exploratory.

## Patient ids as shipped

409 images; 1.00 images per shipped patient id. **The release carries no usable patient ids, so no result published on it can have used a patient-level split.** That is a finding about the benchmark.

## Recovered patients

Embedding for matching: pixel thumbnail (torch-free fallback). Nearest-neighbour cosine similarity (dataset-mean-centred): 10th/50th/90th percentile 0.504 / 0.713 / 0.969. All pairs: median -0.044, 99th 0.497, 99.9th 0.810.

Two-component fit to nearest-neighbour similarity: modes at 0.573 and 0.888 (high-mode weight 0.48), Ashman's D 4.27, ΔBIC (two − one) -155.7. **A separated high-similarity mode exists**, so pairs above 0.736 are declared probable same-patient.

- near-identical by perceptual hash: 37 pairs

- probable same-patient pairs (including those): 99

- recovered group sizes: 212 × 1, 97 × 2, 1 × 3. Groups of 3 or more can be chains of spurious links: inspect them.

## What the shipped-id split could not see

A split on the shipped ids passes its own leakage check vacuously. Judged by the recovered groups, **45 of 98 multi-image groups straddle partitions**, and **19 of 61 test images** have a probable same-patient image in train. Experiments should run on both the shipped-id and the recovered-id split (`with_recovered_patients`) and report the difference.

## Recovered sites

Silhouette by k: k=2: 0.662, k=3: 0.766, k=4: 0.604, k=5: 0.413, k=6: 0.204. Best k = 3 (expected 3); cluster sizes (137, 137, 135). **Clean**: the expected k wins with silhouette ≥ 0.5.

Caries prevalence by recovered site: site_0 0.409 [0.326, 0.493]; site_1 0.226 [0.161, 0.303]; site_2 0.570 [0.472, 0.661]. χ² p < 0.0001 (treats images as independent).

Site probe on raw-intensity thumbnails (torch-free fallback), never on the fingerprints the sites were clustered from: macro AUC **1.000 [1.000, 1.000]** (0.5 = chance), patient-grouped 5-fold CV.

A positive probe is the informative direction: the images carry site, and this representation exposes it. (A null would only have cleared this representation, not the images.)

**Shortcut risk: site is reliably predictable from what the model sees, AND prevalence differs by site.** Diagnostic performance on this data is suspect until it is shown to survive site-stratified evaluation.

## Validation against the generator's ground truth

Only possible on synthetic data. True same-patient pairs: 109; declared: 99; correct: 98 (precision 0.990, recall 0.899). These are the easy case by construction: repeat visits here differ only by a small shift and fresh noise.
