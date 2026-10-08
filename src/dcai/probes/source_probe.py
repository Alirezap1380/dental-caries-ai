"""Shortcut probe: can a simple classifier predict acquisition source from image features?

If site is predictable from what a diagnostic model sees, *and* disease prevalence
differs by site, then any apparent diagnostic skill is suspect: the model can
score well by recognising the scanner. A high AUC here is the finding, not a
success to celebrate.

The claim is asymmetric, and reports must say so. A positive result (site
predictable) is informative: the images carry site, and this representation
exposes it to the model. A null says only that *this representation* does not
carry site linearly. It does not say the images don't. A different encoder,
or a nonlinear probe, may find it. A null is never an all-clear.

Two guards:
- Cross-validation is grouped by patient, so the probe cannot score by
  recognising a patient it saw in training.
- The probe's features must not be the features its labels were derived from.
  Recovered sites are clustered from acquisition fingerprints; probing with those
  same fingerprints would give AUC near 1 by construction. Pass image
  embeddings (what a model sees), never the clustering features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dcai.eval.bootstrap import Estimate, grouped_bootstrap


@dataclass(frozen=True)
class ProbeResult:
    classes: tuple[str, ...]
    n: int
    n_splits: int
    # Macro one-vs-rest AUC of out-of-fold probabilities; 0.5 is chance.
    auc: Estimate
    per_class_auc: dict[str, float]


def _macro_auc(y: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    present = np.unique(y)
    if present.size < 2:
        return float("nan")
    if n_classes == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    if present.size < n_classes:
        return float("nan")
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def site_probe(
    features: np.ndarray,
    labels: Sequence[str] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    seed: int,
    n_splits: int = 5,
    n_boot: int = 2000,
    forbidden_features: np.ndarray | None = None,
) -> ProbeResult:
    """Patient-grouped CV logistic regression from `features` to `labels`.

    `forbidden_features`: the features the labels were derived from (e.g. the
    fingerprints sites were clustered on). Probing with them raises.
    """
    x = np.asarray(features, dtype=float)
    labels = np.asarray(labels).astype(str)
    groups = np.asarray(groups)
    if x.ndim != 2 or x.shape[0] != labels.size or labels.size != groups.size:
        raise ValueError("features must be (n, d) with one label and group per row")
    if forbidden_features is not None and np.array_equal(x, forbidden_features):
        raise ValueError(
            "probe features are the features the labels were derived from: the probe would "
            "be circular. Use image embeddings instead."
        )
    classes = tuple(sorted(set(labels)))
    if len(classes) < 2:
        raise ValueError("need at least two classes to probe")
    y = np.searchsorted(classes, labels)

    proba = np.zeros((y.size, len(classes)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in cv.split(x, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
        clf.fit(x[train], y[train])
        proba[np.ix_(test, clf.classes_)] = clf.predict_proba(x[test])

    k = len(classes)
    per_class = {
        c: float(roc_auc_score(y == i, proba[:, i])) for i, c in enumerate(classes)
    }
    return ProbeResult(
        classes=classes,
        n=y.size,
        n_splits=n_splits,
        auc=grouped_bootstrap(lambda idx: _macro_auc(y[idx], proba[idx], k), groups,
                              seed=seed, n_boot=n_boot),
        per_class_auc=per_class,
    )
