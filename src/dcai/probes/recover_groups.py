"""Recover patient and site structure that a dataset does not ship.

If a benchmark releases no patient ids, no published result on it could have used
a patient-level split. That is a finding about the benchmark. If it releases no
per-image site labels, its shortcut risk cannot be checked directly. This module
tries to recover both from the images themselves. It is pure numpy/scipy/sklearn
plus PIL: embeddings from a learned encoder are passed in, not computed here.

Patients (near-duplicate detection)
    Panoramics carry a near-unique signature in restorations, missing teeth and
    implants. Two stages:
    1. Perceptual hash (DCT pHash). Small Hamming distance means the same image,
       re-exported or re-cropped.
    2. Embedding cosine similarity, for repeat visits: same anatomy, different
       acquisition. Each image's nearest-neighbour similarity gets a 1- vs
       2-component Gaussian mixture. Pairs are declared probable same-patient only
       if BIC prefers two components *and* the high mode is cleanly separated
       (Ashman's D > 2). Otherwise the distribution is reported and no pairs are
       declared. A cutoff picked by eye would be a researcher degree of freedom.

Sites (acquisition fingerprints)
    Image dimensions, aspect ratio, bit depth, file format, JPEG quantisation
    tables, intensity histogram shape, and the noise power spectrum of the
    high-pass residual. Each block is standardised and weighted equally, so the
    32-bin histogram cannot outvote bit depth. K-means over k = 2..6 with
    silhouette. "Clean" means the expected k wins and silhouette >= 0.5
    (Kaufman & Rousseeuw: reasonable structure).

The recovered assignments are then used where the real ones would have been:
patient-level splits with and without the recovered pairs separated, and the
shortcut probe with a prevalence test across recovered sites.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from dcai.data.schema import PatientIdSource, RadiographRecord
from dcai.data.splits import patient_split
from dcai.eval.bootstrap import Estimate, grouped_bootstrap
from dcai.probes.source_probe import ProbeResult, site_probe

# --- image I/O --------------------------------------------------------------------


def load_gray(path: Path) -> np.ndarray:
    """Image as float in [0, 1], whatever its bit depth."""
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    if arr.dtype == np.uint8:
        return arr.astype(float) / 255.0
    arr = arr.astype(float)
    top = 65535.0 if arr.max() > 255 else 255.0
    return np.clip(arr / top, 0.0, 1.0)


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to (height, width) with antialiasing."""
    h, w = size
    return np.asarray(
        Image.fromarray(img.astype(np.float32)).resize((w, h), Image.Resampling.LANCZOS),
        dtype=float,
    )


# --- patients ---------------------------------------------------------------------


def phash(img: np.ndarray, hash_size: int = 8, oversample: int = 4) -> np.ndarray:
    """DCT perceptual hash: low-frequency coefficients above their median."""
    n = hash_size * oversample
    low = dctn(_resize(img, (n, n)), norm="ortho")[:hash_size, :hash_size]
    return (low > np.median(low.ravel()[1:])).ravel()  # median excludes the DC term


def pixel_embedding(img: np.ndarray, size: tuple[int, int] = (24, 48)) -> np.ndarray:
    """Torch-free fallback embedding: a low-pass, standardised thumbnail.

    Enough to find re-exports and repeat visits on aligned panoramics. A learned
    encoder (e.g. a DINO-style model) should replace it on real data.
    """
    thumb = _resize(gaussian_filter(img, 2.0), size).ravel()
    thumb = thumb - thumb.mean()
    sd = thumb.std()
    return thumb / sd if sd > 0 else thumb


def _cosine(emb: np.ndarray) -> np.ndarray:
    # Centre on the dataset mean first: every panoramic shares the arch layout,
    # which would make all pairs look alike.
    x = emb - emb.mean(axis=0)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x @ x.T


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    hamming: int
    cosine: float


@dataclass(frozen=True)
class MixtureFit:
    bic_one: float
    bic_two: float
    means: tuple[float, float]  # (low, high)
    sds: tuple[float, float]
    high_weight: float
    ashman_d: float

    @property
    def separated(self) -> bool:
        return self.bic_two < self.bic_one and self.ashman_d > 2.0


@dataclass(frozen=True)
class PatientRecovery:
    image_ids: tuple[str, ...]
    nn_cosine: np.ndarray  # each image's similarity to its nearest neighbour
    all_pairs_quantiles: dict[float, float]
    mixture: MixtureFit
    threshold: float | None  # None: no separated high-similarity mode
    duplicates: tuple[Pair, ...]  # near-identical by pHash
    pairs: tuple[Pair, ...]  # probable same-patient (includes duplicates)
    group_of: dict[str, str]  # image id -> recovered group label
    group_sizes: dict[int, int]  # group size -> count

    @property
    def n_merged_images(self) -> int:
        return sum(size * n for size, n in self.group_sizes.items() if size > 1)


def _fit_mixture(values: np.ndarray, seed: int) -> MixtureFit:
    v = values.reshape(-1, 1)
    one = GaussianMixture(1, random_state=seed).fit(v)
    two = GaussianMixture(2, random_state=seed, n_init=5).fit(v)
    order = np.argsort(two.means_.ravel())
    means = two.means_.ravel()[order]
    sds = np.sqrt(two.covariances_.ravel()[order])
    d = float(np.sqrt(2) * abs(means[1] - means[0]) / np.sqrt(sds[0] ** 2 + sds[1] ** 2))
    return MixtureFit(
        bic_one=float(one.bic(v)), bic_two=float(two.bic(v)),
        means=(float(means[0]), float(means[1])), sds=(float(sds[0]), float(sds[1])),
        high_weight=float(two.weights_[order[1]]), ashman_d=d,
    )


def _threshold(fit: MixtureFit) -> float:
    """Where the two fitted components are equally dense, between the means."""
    grid = np.linspace(fit.means[0], fit.means[1], 2001)

    def dens(m: float, s: float, w: float) -> np.ndarray:
        return w * np.exp(-0.5 * ((grid - m) / s) ** 2) / s

    diff = (dens(fit.means[1], fit.sds[1], fit.high_weight)
            - dens(fit.means[0], fit.sds[0], 1 - fit.high_weight))
    return float(grid[int(np.argmax(diff > 0))])


def recover_patients(
    image_ids: Sequence[str],
    hashes: np.ndarray,
    embeddings: np.ndarray,
    *,
    seed: int,
    max_hamming: int = 4,
) -> PatientRecovery:
    ids = tuple(image_ids)
    n = len(ids)
    hashes = np.asarray(hashes, dtype=bool)
    if hashes.shape[0] != n or embeddings.shape[0] != n or n < 3:
        raise ValueError("need >= 3 images with one hash and one embedding each")

    cos = _cosine(np.asarray(embeddings, dtype=float))
    ham = (hashes[:, None, :] != hashes[None, :, :]).sum(axis=2)
    off = ~np.eye(n, dtype=bool)
    nn = np.where(off, cos, -np.inf).max(axis=1)
    upper = cos[np.triu_indices(n, 1)]

    fit = _fit_mixture(nn, seed)
    threshold = _threshold(fit) if fit.separated else None

    dup, linked = [], []
    for i, j in zip(*np.triu_indices(n, 1), strict=True):
        pair = Pair(ids[i], ids[j], int(ham[i, j]), float(cos[i, j]))
        if ham[i, j] <= max_hamming:
            dup.append(pair)
            linked.append(pair)
        elif threshold is not None and cos[i, j] >= threshold:
            linked.append(pair)

    # Union-find. Chaining can merge distinct patients through spurious links,
    # so group sizes are reported: an implausibly large group is a warning sign.
    parent = list(range(n))
    index = {k: i for i, k in enumerate(ids)}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pr in linked:
        parent[find(index[pr.a])] = find(index[pr.b])
    roots = [find(i) for i in range(n)]
    label = {r: f"recovered-{k:04d}" for k, r in enumerate(dict.fromkeys(roots))}
    group_of = {ids[i]: label[roots[i]] for i in range(n)}
    sizes = np.bincount(np.unique(roots, return_inverse=True)[1])

    return PatientRecovery(
        image_ids=ids,
        nn_cosine=nn,
        all_pairs_quantiles={q: float(np.quantile(upper, q)) for q in (0.5, 0.9, 0.99, 0.999)},
        mixture=fit,
        threshold=threshold,
        duplicates=tuple(dup),
        pairs=tuple(linked),
        group_of=group_of,
        group_sizes={int(s): int(c) for s, c in zip(*np.unique(sizes, return_counts=True),
                                                     strict=True)},
    )


def with_recovered_patients(
    records: Sequence[RadiographRecord], recovery: PatientRecovery
) -> list[RadiographRecord]:
    """Records whose patient ids follow the recovered groups (DERIVED where merged)."""
    members: dict[str, int] = {}
    for g in recovery.group_of.values():
        members[g] = members.get(g, 0) + 1
    out = []
    for r in records:
        g = recovery.group_of[r.image_id]
        if members[g] > 1:
            r = replace(r, patient_id=g, patient_id_source=PatientIdSource.DERIVED)
        out.append(r)
    return out


@dataclass(frozen=True)
class SplitSensitivity:
    n_recovered_pairs: int
    # Under the split a benchmark with no patient ids would use (each image its own
    # patient): recovered groups that land in more than one partition...
    groups_straddling: int
    n_multi_image_groups: int
    # ...and test images with a probable same-patient image in train.
    test_images_with_partner_in_train: int
    n_test_images: int


def split_sensitivity(
    records: Sequence[RadiographRecord],
    recovery: PatientRecovery,
    *,
    seed: int,
    fractions: Mapping[str, float] | None = None,
) -> SplitSensitivity:
    """How badly the ids-as-shipped split leaks, judged by the recovered groups.

    `records` carry the ids as shipped. The split built on them passes its own
    leakage check vacuously; this measures what that check could not see.
    """
    kwargs = {} if fractions is None else {"fractions": fractions}
    split = patient_split(records, seed=seed, **kwargs)
    homes: dict[str, set[str]] = {}
    for name, part in split.partitions.items():
        for r in part:
            homes.setdefault(recovery.group_of[r.image_id], set()).add(name)
    sizes: dict[str, int] = {}
    for g in recovery.group_of.values():
        sizes[g] = sizes.get(g, 0) + 1
    multi = {g for g, n in sizes.items() if n > 1}
    train_groups = {recovery.group_of[r.image_id] for r in split["train"]}
    test = split["test"]
    return SplitSensitivity(
        n_recovered_pairs=len(recovery.pairs),
        groups_straddling=sum(len(homes[g]) > 1 for g in multi),
        n_multi_image_groups=len(multi),
        test_images_with_partner_in_train=sum(
            recovery.group_of[r.image_id] in train_groups for r in test
        ),
        n_test_images=len(test),
    )


# --- sites ------------------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprint:
    image_id: str
    height: int
    width: int
    bit_depth: int
    file_format: str
    quant_tables: str  # hash of the JPEG quantisation tables; "" if not JPEG
    histogram: np.ndarray  # 32-bin intensity histogram, sums to 1
    noise_spectrum: np.ndarray  # radially averaged log power of the high-pass residual


def _noise_spectrum(img: np.ndarray, n_bins: int = 16, patch: int = 64) -> np.ndarray:
    residual = img - gaussian_filter(img, 2.0)
    h, w = residual.shape
    ph, pw = min(patch, h), min(patch, w)
    power = np.zeros((ph, pw))
    count = 0
    for y in range(0, h - ph + 1, ph):
        for x in range(0, w - pw + 1, pw):
            tile = residual[y:y + ph, x:x + pw]
            power += np.abs(np.fft.fft2(tile - tile.mean())) ** 2
            count += 1
    power = np.fft.fftshift(power / max(count, 1))
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - ph / 2, xx - pw / 2) / (min(ph, pw) / 2)
    idx = np.minimum((r * n_bins).astype(int), n_bins - 1)
    radial = np.bincount(idx.ravel(), weights=power.ravel(), minlength=n_bins)
    radial /= np.maximum(np.bincount(idx.ravel(), minlength=n_bins), 1)
    return np.log(radial + 1e-12)


def fingerprint(image_id: str, path: Path) -> Fingerprint:
    with Image.open(path) as im:
        fmt = im.format or "unknown"
        mode = im.mode
        tables = getattr(im, "quantization", None) or {}
        q = hashlib.sha1(repr(sorted((k, list(v)) for k, v in tables.items())).encode())
        quant = q.hexdigest()[:12] if tables else ""
    bit_depth = 16 if mode.startswith("I;16") or mode == "I" else 8
    img = load_gray(path)
    hist = np.histogram(img, bins=32, range=(0.0, 1.0))[0].astype(float)
    return Fingerprint(
        image_id=image_id, height=img.shape[0], width=img.shape[1], bit_depth=bit_depth,
        file_format=fmt, quant_tables=quant, histogram=hist / hist.sum(),
        noise_spectrum=_noise_spectrum(img),
    )


def fingerprint_features(fps: Sequence[Fingerprint]) -> np.ndarray:
    """Block-standardised feature matrix; each block carries equal total weight."""
    def onehot(values: list[str]) -> np.ndarray:
        cats = sorted(set(values))
        return np.array([[v == c for c in cats] for v in values], dtype=float)

    blocks = [
        np.array([[np.log(f.height), np.log(f.width), f.width / f.height] for f in fps]),
        np.array([[f.bit_depth] for f in fps], dtype=float),
        onehot([f.file_format for f in fps]),
        onehot([f.quant_tables for f in fps]),
        np.array([np.sqrt(f.histogram) for f in fps]),
        np.array([f.noise_spectrum for f in fps]),
    ]
    out = []
    for b in blocks:
        sd = b.std(axis=0)
        # Relative tolerance: a constant column's float std is ~1e-17, not 0, and
        # dividing by it would turn the constant into unit-variance noise.
        keep = sd > 1e-9 * np.maximum(1.0, np.abs(b.mean(axis=0)))
        if not keep.any():
            continue  # constant across the dataset: carries no site information
        z = (b[:, keep] - b[:, keep].mean(axis=0)) / sd[keep]
        out.append(z / np.sqrt(keep.sum()))
    if not out:
        raise ValueError("every fingerprint feature is constant: no acquisition structure")
    return np.hstack(out)


@dataclass(frozen=True)
class SiteClustering:
    silhouette: dict[int, float]  # k -> silhouette
    best_k: int
    expected_k: int
    labels: np.ndarray  # cluster per image, at best_k
    sizes: tuple[int, ...]

    @property
    def clean(self) -> bool:
        return self.best_k == self.expected_k and self.silhouette[self.best_k] >= 0.5


def cluster_sites(
    features: np.ndarray, *, seed: int, expected_k: int = 3, k_range: Sequence[int] = range(2, 7)
) -> SiteClustering:
    ks = [k for k in k_range if 2 <= k < features.shape[0]]
    if not ks:
        raise ValueError("not enough images to cluster")
    fits = {k: KMeans(k, n_init=10, random_state=seed).fit(features) for k in ks}
    sil = {k: float(silhouette_score(features, fits[k].labels_)) for k in ks}
    best = max(sil, key=sil.get)
    labels = fits[best].labels_
    return SiteClustering(sil, best, expected_k, labels,
                          tuple(int(c) for c in np.bincount(labels)))


@dataclass(frozen=True)
class PrevalenceByCluster:
    prevalence: dict[str, Estimate]  # per recovered site
    chi2: float
    p_value: float
    # The chi-square test treats images as independent. Same-patient images
    # inflate its effective n slightly; the grouped CIs above do not have that problem.


def prevalence_by_cluster(
    has_lesion: Sequence[bool] | np.ndarray,
    clusters: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
) -> PrevalenceByCluster:
    y = np.asarray(has_lesion, dtype=bool)
    c = np.asarray(clusters)
    g = np.asarray(groups)
    table = np.array([[np.sum((c == k) & y), np.sum((c == k) & ~y)] for k in np.unique(c)])
    chi2, p, _, _ = chi2_contingency(table)
    prev = {}
    for k in np.unique(c):
        sel = c == k
        yk, gk = y[sel], g[sel]
        prev[f"site_{k}"] = grouped_bootstrap(lambda idx, yk=yk: float(yk[idx].mean()), gk,
                                              seed=seed, n_boot=n_boot)
    return PrevalenceByCluster(prev, float(chi2), float(p))


@dataclass(frozen=True)
class SiteRecovery:
    clustering: SiteClustering
    prevalence: PrevalenceByCluster
    probe: ProbeResult

    @property
    def shortcut_risk(self) -> bool:
        """Site reliably predictable from model inputs AND prevalence differs by site.

        "Reliably" means the probe AUC's lower bound clears chance (0.5). Any
        magnitude is reported alongside; there is no principled "safe" AUC above chance.
        """
        return self.probe.auc.lo > 0.5 and self.prevalence.p_value < 0.05


def recover_sites(
    fingerprints: Sequence[Fingerprint],
    embeddings: np.ndarray,
    has_lesion: Sequence[bool] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    seed: int,
    expected_k: int = 3,
    n_boot: int = 2000,
) -> SiteRecovery:
    """Cluster on fingerprints; probe and test prevalence against the clusters.

    The probe uses `embeddings`, never the fingerprint features (see source_probe).
    """
    feats = fingerprint_features(fingerprints)
    clustering = cluster_sites(feats, seed=seed, expected_k=expected_k)
    site_labels = np.array([f"site_{k}" for k in clustering.labels])
    return SiteRecovery(
        clustering=clustering,
        prevalence=prevalence_by_cluster(has_lesion, clustering.labels, groups,
                                         seed=seed, n_boot=n_boot),
        probe=site_probe(embeddings, site_labels, groups, seed=seed, n_boot=n_boot,
                         forbidden_features=feats),
    )


# --- orchestration and report ------------------------------------------------------


def intensity_thumbnail(img: np.ndarray, size: tuple[int, int] = (24, 48)) -> np.ndarray:
    """Raw-intensity thumbnail: keeps exposure, as a network fed raw pixels would.

    Contrast with `pixel_embedding`, which standardises each image and so is blind
    to exposure by construction. A site probe's answer depends on the feature
    space: probe with what the diagnostic model actually sees.
    """
    return _resize(gaussian_filter(img, 1.0), size).ravel()


@dataclass(frozen=True)
class RecoveryReport:
    n_images: int
    shipped_ids_per_patient: float  # images per shipped patient id; 1.0 means no ids
    embedding_name: str  # features used to match patients
    probe_feature_name: str  # features the site probe used
    patients: PatientRecovery
    split: SplitSensitivity
    sites: SiteRecovery


def run_recovery(
    records: Sequence[RadiographRecord],
    paths: Mapping[str, Path],
    has_lesion: Mapping[str, bool],
    *,
    seed: int,
    embeddings: np.ndarray | None = None,
    embedding_name: str = "pixel thumbnail (torch-free fallback)",
    n_boot: int = 2000,
) -> RecoveryReport:
    """Run both recoveries over `records` (patient ids as shipped).

    `embeddings` (one row per record, in order) should come from the encoder the
    diagnostic model uses. Without it, the fallback serves for patient matching,
    and the site probe uses raw-intensity thumbnails.
    """
    ids = [r.image_id for r in records]
    imgs = [load_gray(paths[i]) for i in ids]
    hashes = np.array([phash(x) for x in imgs])
    match_emb = embeddings if embeddings is not None else np.array(
        [pixel_embedding(x) for x in imgs])
    probe_emb = embeddings if embeddings is not None else np.array(
        [intensity_thumbnail(x) for x in imgs])
    patients = recover_patients(ids, hashes, match_emb, seed=seed)
    groups = [patients.group_of[i] for i in ids]
    sites = recover_sites(
        [fingerprint(i, paths[i]) for i in ids], probe_emb, [has_lesion[i] for i in ids],
        groups, seed=seed, n_boot=n_boot,
    )
    return RecoveryReport(
        n_images=len(ids),
        shipped_ids_per_patient=len(ids) / len({r.group_key for r in records}),
        embedding_name=embedding_name,
        probe_feature_name=(embedding_name if embeddings is not None
                            else "raw-intensity thumbnails (torch-free fallback)"),
        patients=patients,
        split=split_sensitivity(records, patients, seed=seed),
        sites=sites,
    )


def _p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"= {p:.4f}"


def recovery_markdown(r: RecoveryReport, *, synthetic: bool) -> str:
    p, sp, st = r.patients, r.split, r.sites
    mix = p.mixture
    lines = ["# Group recovery report"]
    if synthetic:
        lines.append(
            "> **SYNTHETIC IMAGES.** Every number below is a property of "
            "`dcai/data/synthetic_images.py`, which makes repeat visits near-identical "
            "and sites cleanly distinct: an easy case, meant to show the probes compose.")
    lines += [
        ("Intervals are 95% patient-grouped bootstrap intervals (grouped by *recovered* "
        "patient), uncorrected and exploratory."),
        "## Patient ids as shipped",
        f"{r.n_images} images; {r.shipped_ids_per_patient:.2f} images per shipped patient id."
        + (" **The release carries no usable patient ids, so no result published on it "
           "can have used a patient-level split.** That is a finding about the benchmark."
           if r.shipped_ids_per_patient == 1.0 else ""),
        "## Recovered patients",
        f"Embedding for matching: {r.embedding_name}. Nearest-neighbour cosine similarity "
        f"(dataset-mean-centred): 10th/50th/90th percentile "
        + " / ".join(f"{q:.3f}" for q in np.quantile(p.nn_cosine, [0.1, 0.5, 0.9]))
        + f". All pairs: median {p.all_pairs_quantiles[0.5]:.3f}, 99th {p.all_pairs_quantiles[0.99]:.3f}, 99.9th {p.all_pairs_quantiles[0.999]:.3f}.",
        f"Two-component fit to nearest-neighbour similarity: modes at {mix.means[0]:.3f} "
        f"and {mix.means[1]:.3f} (high-mode weight {mix.high_weight:.2f}), Ashman's D "
        f"{mix.ashman_d:.2f}, ΔBIC (two − one) {mix.bic_two - mix.bic_one:.1f}. "
        + (f"**A separated high-similarity mode exists**, so pairs above {p.threshold:.3f} "
           "are declared probable same-patient." if p.threshold is not None else
           "No separated high-similarity mode, so no embedding pairs are declared."),
        f"- near-identical by perceptual hash: {len(p.duplicates)} pairs",
        f"- probable same-patient pairs (including those): {len(p.pairs)}",
        "- recovered group sizes: " + ", ".join(
            f"{n} × {size}" for size, n in sorted(p.group_sizes.items()))
        + ". Groups of 3 or more can be chains of spurious links: inspect them.",
        "## What the shipped-id split could not see",
        (f"A split on the shipped ids passes its own leakage check vacuously. Judged by the "
        f"recovered groups, **{sp.groups_straddling} of {sp.n_multi_image_groups} "
        f"multi-image groups straddle partitions**, and **{sp.test_images_with_partner_in_train} "
        f"of {sp.n_test_images} test images** have a probable same-patient image in train. "
        "Experiments should run on both the shipped-id and the recovered-id split "
        "(`with_recovered_patients`) and report the difference."),
        "## Recovered sites",
        "Silhouette by k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in st.clustering.silhouette.items())
        + f". Best k = {st.clustering.best_k} (expected {st.clustering.expected_k}); cluster "
        f"sizes {st.clustering.sizes}. "
        + ("**Clean**: the expected k wins with silhouette ≥ 0.5." if st.clustering.clean
           else "**Not clean**: treat the recovered sites with caution."),
        "Caries prevalence by recovered site: " + "; ".join(
            f"{k} {v}" for k, v in st.prevalence.prevalence.items())
        + f". χ² p {_p(st.prevalence.p_value)} (treats images as independent).",
        (f"Site probe on {r.probe_feature_name}, never on the fingerprints the sites were "
        f"clustered from: macro AUC **{st.probe.auc}** (0.5 = chance), patient-grouped "
        f"{st.probe.n_splits}-fold CV."),
        ("A positive probe is the informative direction: the images carry site, and this "
         "representation exposes it. (A null would only have cleared this representation, "
         "not the images.)\n\n"
         "**Shortcut risk: site is reliably predictable from what the model sees, AND "
         "prevalence differs by site.** Diagnostic performance on this data is suspect "
         "until it is shown to survive site-stratified evaluation.") if st.shortcut_risk else
        ("No shortcut risk detected, and this is a null, which is the weak direction. "
         "It says only that *this representation* does not carry site linearly, or that "
         "prevalence does not differ. It does not say the images are site-free: another "
         "encoder or a nonlinear probe may find what this one did not. Never read it as "
         "an all-clear."),
    ]
    return "\n\n".join(lines) + "\n"
