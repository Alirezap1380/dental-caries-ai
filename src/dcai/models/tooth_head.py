"""Stage-2 head: tooth features -> P(sound, caries, deep caries).

A multinomial logistic regression on standardised frozen features. The
regularisation strength is chosen by patient-grouped cross-validation on the
training partition only, scored by log-loss rather than accuracy: the
probabilities feed calibration and the operating point, so their quality is
what matters. Classes are not re-weighted, because re-weighting would buy
sensitivity at the cost of calibrated probabilities. The operating point,
fitted on validation, is where the sensitivity trade-off belongs.

Pure sklearn: importable without torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FittedHead:
    pipeline: Pipeline
    n_classes: int
    c: float
    cv_log_loss: dict[float, float]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probabilities over all `n_classes`, even ones absent from training."""
        p = self.pipeline.predict_proba(x)
        out = np.zeros((x.shape[0], self.n_classes))
        out[:, self.pipeline.classes_] = p
        return out


def _model(c: float) -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def fit_head(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    *,
    n_classes: int,
    seed: int,
    cs: Sequence[float] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    n_splits: int = 5,
) -> FittedHead:
    """Choose C by patient-grouped CV log-loss, then refit on all training teeth."""
    if len(set(groups)) < n_splits:
        raise ValueError(f"need at least {n_splits} patients for {n_splits}-fold CV")
    labels = list(range(n_classes))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(x, y, groups))
    scores = {}
    for c in cs:
        losses = []
        for tr, te in folds:
            m = _model(c).fit(x[tr], y[tr])
            p = np.zeros((len(te), n_classes))
            p[:, m.classes_] = m.predict_proba(x[te])
            losses.append(log_loss(y[te], np.clip(p, 1e-12, 1.0), labels=labels))
        scores[float(c)] = float(np.mean(losses))
    best = min(scores, key=scores.get)
    return FittedHead(_model(best).fit(x, y), n_classes, best, scores)
