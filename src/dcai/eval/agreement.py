"""Inter-observer agreement, and the model measured against it.

Rule 3: human agreement is the ceiling, not the floor. Two readers looking at the
same radiograph disagree on caries often, and most of all on early lesions.

The unit is the tooth (see `ratings.py` for why not boxes or images).

Three kinds of number come out of this module:

1. Human disagreement, described honestly: pairwise Cohen's kappa between
   individual readers, Fleiss' kappa (complete designs only), and Krippendorff's
   alpha (which handles readers who did not read every image, plus ordinal weighting).

2. The ceiling for a model, per reader X: agreement between X and the
   *leave-one-out consensus* of the other readers. A model trained on consensus
   labels approximates a consensus, and a consensus is lower-variance than any
   individual reader. So comparing the model to individual readers would make it
   "beat the humans" by construction. Comparing consensus with X puts
   matched-variance on both sides of the comparison.

3. The model against that ceiling, as a paired difference over the same patients.

Residual caveat, which belongs in the write-up: the LOO consensus pools N-1
readers, while a model trained on many images can average away label noise
beyond what any finite panel does. So `exceeds_ceiling` means "needs explaining"
(reader mimicry, leakage, or a genuinely better denoiser), not proof of mimicry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

from dcai.eval.bootstrap import Estimate, difference, grouped_bootstrap
from dcai.eval.ratings import MISSING, RatingMatrix, consensus_grades

Weights = Literal["linear", "quadratic"] | None
AlphaMetric = Literal["nominal", "ordinal", "interval"]


def _weight_matrix(k: int, weights: Weights) -> np.ndarray:
    i, j = np.indices((k, k))
    if weights is None:
        return (i != j).astype(float)
    if weights == "linear":
        return np.abs(i - j).astype(float)
    if weights == "quadratic":
        return ((i - j) ** 2).astype(float)
    raise ValueError(f"unknown weights {weights!r}")


def cohen_kappa(a: np.ndarray, b: np.ndarray, n_categories: int, weights: Weights = None) -> float:
    """Cohen's kappa over units both raters graded. NaN where undefined.

    Weighted variants suit ordinal scales: calling a deep lesion "caries" is a
    smaller disagreement than calling it "sound".
    """
    a, b = np.asarray(a), np.asarray(b)
    both = (a != MISSING) & (b != MISSING)
    a, b = a[both], b[both]
    if a.size == 0:
        return float("nan")
    k = n_categories
    observed = np.bincount(a * k + b, minlength=k * k).reshape(k, k) / a.size
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    w = _weight_matrix(k, weights)
    disagreement_expected = float((w * expected).sum())
    # Both raters constant on the same category: chance agreement is total and
    # kappa is 0/0. That is "undefined", not "perfect agreement".
    if np.isclose(disagreement_expected, 0.0):
        return float("nan")
    return 1.0 - float((w * observed).sum()) / disagreement_expected


def fleiss_kappa(ratings: np.ndarray, n_categories: int) -> float:
    """Fleiss' kappa for a complete design (every rater grades every unit)."""
    ratings = np.asarray(ratings)
    if ratings.ndim != 2 or ratings.shape[1] < 2:
        raise ValueError("need a units x raters matrix with at least two raters")
    if (ratings == MISSING).any():
        raise ValueError(
            "Fleiss' kappa needs every rater on every unit; use Krippendorff's alpha "
            "for incomplete designs"
        )
    if ratings.shape[0] == 0:
        return float("nan")
    n = ratings.shape[1]
    counts = (ratings[:, :, None] == np.arange(n_categories)).sum(axis=1)
    p_item = (counts * (counts - 1)).sum(axis=1) / (n * (n - 1))
    p_cat = counts.sum(axis=0) / counts.sum()
    p_e = float((p_cat**2).sum())
    if np.isclose(p_e, 1.0):
        return float("nan")
    return (float(p_item.mean()) - p_e) / (1.0 - p_e)


def krippendorff_alpha(
    ratings: np.ndarray, n_categories: int, metric: AlphaMetric = "ordinal"
) -> float:
    """Krippendorff's alpha. MISSING entries are allowed; units with < 2 ratings drop out.

    Built from the coincidence matrix (Krippendorff 2011, "Computing
    Krippendorff's alpha-reliability"). The ordinal metric uses the observed
    marginals, so "caries vs deep caries" counts as a smaller disagreement when
    many values lie between them, and a larger one when few do.
    """
    ratings = np.asarray(ratings)
    k = n_categories
    counts = np.stack([(ratings == c).sum(axis=1) for c in range(k)], axis=1).astype(float)
    m = counts.sum(axis=1)
    counts, m = counts[m >= 2], m[m >= 2]
    if counts.shape[0] == 0:
        return float("nan")
    o = (counts.T / (m - 1)) @ counts - np.diag((counts / (m - 1)[:, None]).sum(axis=0))
    n_c = o.sum(axis=1)
    n = n_c.sum()

    i, j = np.indices((k, k))
    if metric == "nominal":
        delta = (i != j).astype(float)
    elif metric == "interval":
        delta = ((i - j) ** 2).astype(float)
    elif metric == "ordinal":
        cum = np.r_[0.0, np.cumsum(n_c)]
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        delta = (cum[hi + 1] - cum[lo] - (n_c[i] + n_c[j]) / 2) ** 2
    else:
        raise ValueError(f"unknown metric {metric!r}")

    d_o = float((o * delta).sum()) / n
    d_e = float((np.outer(n_c, n_c) * delta).sum()) / (n * (n - 1))
    if np.isclose(d_e, 0.0):
        return float("nan")
    return 1.0 - d_o / d_e


@dataclass(frozen=True)
class PairAgreement:
    rater_a: str
    rater_b: str
    n_items: int
    kappa: Estimate | None  # None if the pair shares fewer than two patients


def pairwise_agreement(
    m: RatingMatrix,
    *,
    seed: int,
    weights: Weights = None,
    raters: Sequence[str] | None = None,
    n_boot: int = 2000,
) -> list[PairAgreement]:
    """Individual-vs-individual kappa: how much humans disagree. Not the ceiling."""
    raters = list(raters or m.raters)
    groups = np.asarray(m.groups)
    out = []
    for ra, rb in combinations(raters, 2):
        a, b = m.column(ra), m.column(rb)
        shared = (a != MISSING) & (b != MISSING)
        kappa = None
        if len(set(groups[shared])) >= 2:
            a_s, b_s = a[shared], b[shared]
            kappa = grouped_bootstrap(
                lambda idx, a_s=a_s, b_s=b_s: cohen_kappa(
                    a_s[idx], b_s[idx], m.scale.n_categories, weights
                ),
                groups[shared], seed=seed, n_boot=n_boot,
            )
        out.append(PairAgreement(ra, rb, int(shared.sum()), kappa))
    return out


def fleiss_agreement(
    m: RatingMatrix,
    *,
    seed: int,
    raters: Sequence[str] | None = None,
    n_boot: int = 2000,
) -> Estimate:
    cols = np.stack([m.column(r) for r in (raters or m.raters)], axis=1)
    if (cols == MISSING).any():
        raise ValueError(
            "needs a complete design, but some readers did not read every image; "
            "use Krippendorff's alpha"
        )
    return grouped_bootstrap(
        lambda idx: fleiss_kappa(cols[idx], m.scale.n_categories),
        m.groups, seed=seed, n_boot=n_boot,
    )


def krippendorff_agreement(
    m: RatingMatrix,
    *,
    seed: int,
    metric: AlphaMetric = "ordinal",
    raters: Sequence[str] | None = None,
    n_boot: int = 2000,
) -> Estimate:
    cols = np.stack([m.column(r) for r in (raters or m.raters)], axis=1)
    return grouped_bootstrap(
        lambda idx: krippendorff_alpha(cols[idx], m.scale.n_categories, metric),
        m.groups, seed=seed, n_boot=n_boot,
    )


@dataclass(frozen=True)
class CeilingComparison:
    reader: str
    n_items: int
    model_kappa: Estimate  # kappa(model, reader)
    ceiling_kappa: Estimate  # kappa(LOO consensus of the other readers, reader)
    # Mean kappa(other individual, reader). Describes human disagreement; it is
    # NOT the ceiling (higher-variance than a consensus-trained model).
    individual_kappa: Estimate
    difference: Estimate  # model - ceiling, paired over the same patients

    @property
    def exceeds_ceiling(self) -> bool:
        """Model agrees with this reader reliably more than the others' consensus does.

        Needs explaining before anything else is claimed: reader mimicry,
        leakage, or a model that denoises labels better than an (N-1)-reader panel.
        """
        return self.difference.lo > 0

    @property
    def below_ceiling(self) -> bool:
        return self.difference.hi < 0


def model_vs_readers(
    m: RatingMatrix,
    model: str,
    *,
    seed: int,
    weights: Weights = None,
    min_consensus: int = 2,
    n_boot: int = 2000,
) -> list[CeilingComparison]:
    """Compare the model with each reader against that reader's LOO-consensus ceiling.

    Units enter a reader's comparison only where that reader, the model and at
    least `min_consensus` other readers all graded them, so the three kappas are
    computed on the same units.
    """
    if model not in m.raters:
        raise ValueError(f"{model!r} is not a rater in this matrix")
    humans = [r for r in m.raters if r != model]
    if len(humans) < min_consensus + 1:
        raise ValueError(
            f"a leave-one-out consensus ceiling needs at least {min_consensus + 1} human "
            f"readers; got {humans}. Report model-vs-reader kappa without a ceiling instead."
        )
    k = m.scale.n_categories
    model_col = m.column(model)
    out = []
    for reader in humans:
        others = np.stack([m.column(h) for h in humans if h != reader], axis=1)
        loo = consensus_grades(others, min_raters=min_consensus)
        rows = (m.column(reader) != MISSING) & (model_col != MISSING) & (loo != MISSING)
        ref, pred, cons = m.column(reader)[rows], model_col[rows], loo[rows]
        indiv = others[rows]
        groups = np.asarray(m.groups)[rows]

        def mean_individual(idx: np.ndarray, ref: np.ndarray = ref,
                            indiv: np.ndarray = indiv) -> float:
            vals = [cohen_kappa(indiv[idx, c], ref[idx], k, weights)
                    for c in range(indiv.shape[1])]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        # Same seed, same groups: identical resamples, so differences are paired.
        def boot(stat, groups: np.ndarray = groups) -> Estimate:
            return grouped_bootstrap(stat, groups, seed=seed, n_boot=n_boot)

        model_k = boot(lambda idx, ref=ref, pred=pred: cohen_kappa(pred[idx], ref[idx], k, weights))
        ceiling_k = boot(lambda idx, ref=ref, cons=cons: cohen_kappa(cons[idx], ref[idx], k, weights))
        out.append(
            CeilingComparison(
                reader=reader,
                n_items=int(rows.sum()),
                model_kappa=model_k,
                ceiling_kappa=ceiling_k,
                individual_kappa=boot(mean_individual),
                difference=difference(model_k, ceiling_k, paired=True),
            )
        )
    return out


@dataclass(frozen=True)
class KappaSweep:
    """Model agreement across the whole threshold range, against the human ceiling.

    Kappa at one operating point mixes two questions: "is the model worse than the
    readers?" and "was it pinned to a threshold the readers don't use?". A
    sensitivity-first threshold calls many more teeth carious than any reader
    does, and kappa punishes that regardless of how good the ranking is. The sweep
    separates the two questions.

    `best_threshold` is chosen on the same data it is scored on, so `at_best` is
    optimistic. It is a diagnostic of the gap, not a deployable number.
    """

    thresholds: np.ndarray
    model_kappa: np.ndarray  # mean over readers of kappa(model_t, reader); point estimates
    ceiling: Estimate  # mean over readers of kappa(LOO consensus, reader): the band
    operating_threshold: float
    best_threshold: float
    at_operating: Estimate
    at_best: Estimate
    best_minus_operating: Estimate  # paired over the same patients

    @property
    def reaches_ceiling(self) -> bool:
        """Some threshold brings the model into the human band (at or above its lower edge)."""
        return bool(self.model_kappa.max() >= self.ceiling.lo)

    @property
    def operating_point_costs_agreement(self) -> bool:
        return self.best_minus_operating.lo > 0


def kappa_sweep(
    m: RatingMatrix,
    grades_at: Callable[[float], np.ndarray],
    *,
    operating_threshold: float,
    seed: int,
    thresholds: Sequence[float] | np.ndarray | None = None,
    weights: Weights = None,
    min_consensus: int = 2,
    n_boot: int = 2000,
) -> KappaSweep:
    """Sweep the model's threshold; `m` holds human readers only.

    `grades_at(t)` returns the model's grade for every item of `m` at threshold t.
    Each reader is scored on the units where they and at least `min_consensus`
    other readers graded, the same units as `model_vs_readers`.
    """
    readers = list(m.raters)
    if len(readers) < min_consensus + 1:
        raise ValueError(
            f"a leave-one-out consensus ceiling needs at least {min_consensus + 1} human "
            f"readers; got {readers}"
        )
    k = m.scale.n_categories
    refs, loos = [], []
    for reader in readers:
        others = np.stack([m.column(h) for h in readers if h != reader], axis=1)
        loo = consensus_grades(others, min_raters=min_consensus)
        usable = (m.column(reader) != MISSING) & (loo != MISSING)
        refs.append(np.where(usable, m.column(reader), MISSING))
        loos.append(np.where(usable, loo, MISSING))

    def mean_kappa(pred: np.ndarray, idx: np.ndarray) -> float:
        vals = [cohen_kappa(pred[idx], ref[idx], k, weights) for ref in refs]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def ceiling_stat(idx: np.ndarray) -> float:
        vals = [cohen_kappa(lo[idx], ref[idx], k, weights) for lo, ref in zip(loos, refs, strict=True)]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.r_[np.asarray(thresholds, dtype=float), operating_threshold])
    everything = np.arange(len(m.item_ids))
    curve = np.array([mean_kappa(np.asarray(grades_at(t)), everything) for t in thresholds])
    best = float(thresholds[int(np.nanargmax(curve))])

    def boot(stat: Callable[[np.ndarray], float]) -> Estimate:
        return grouped_bootstrap(stat, m.groups, seed=seed, n_boot=n_boot)

    at_op_grades = np.asarray(grades_at(operating_threshold))
    best_grades = np.asarray(grades_at(best))
    at_op = boot(lambda idx: mean_kappa(at_op_grades, idx))
    at_best = boot(lambda idx: mean_kappa(best_grades, idx))
    return KappaSweep(
        thresholds=thresholds,
        model_kappa=curve,
        ceiling=boot(ceiling_stat),
        operating_threshold=operating_threshold,
        best_threshold=best,
        at_operating=at_op,
        at_best=at_best,
        best_minus_operating=difference(at_best, at_op, paired=True),
    )
