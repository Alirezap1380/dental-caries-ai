"""Model-side code that runs without torch: crops and the stage-2 head."""

from __future__ import annotations

import numpy as np
import pytest

from dcai.data.schema import BBox
from dcai.models.encoders import CROP, crop_tooth, tooth_features
from dcai.models.tooth_head import fit_head


def test_crop_is_square_letterboxed_and_clipped() -> None:
    img = np.zeros((100, 200), dtype=np.float32)
    img[20:80, 50:70] = 1.0  # a tall, narrow "tooth"
    crop = crop_tooth(img, BBox(50, 20, 70, 80), margin=0.0, size=60)
    assert crop.shape == (60, 60)
    # Aspect kept: the bright column stays narrow, padded at the sides.
    assert crop[:, 30].mean() > 0.9 and crop[:, 2].max() == 0.0
    edge = crop_tooth(img, BBox(0, 0, 10, 10), margin=0.5)  # margin past the border
    assert edge.shape == (CROP, CROP)


def test_tooth_features_batches_every_tooth() -> None:
    img = np.random.default_rng(0).random((50, 80)).astype(np.float32)
    boxes = {36: BBox(0, 0, 10, 20), 11: BBox(20, 5, 30, 25), 46: BBox(40, 10, 50, 40)}
    calls = []

    def encoder(batch: np.ndarray) -> np.ndarray:
        calls.append(len(batch))
        return batch.reshape(len(batch), -1)[:, :4]

    keys, feats = tooth_features([("a", lambda: img, boxes), ("b", lambda: img, {11: boxes[11]})],
                                 encoder, batch=2)
    assert [(k.image_id, k.tooth_fdi) for k in keys] == [("a", 11), ("a", 36), ("a", 46), ("b", 11)]
    assert feats.shape == (4, 4) and calls == [2, 2]
    assert tooth_features([], encoder)[1].shape == (0, 0)


def separable(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.8, 0.15, 0.05])
    x = rng.normal(size=(n, 10)) + np.eye(3)[y] @ rng.normal(size=(3, 10)) * 1.5
    groups = [f"p{i // 4}" for i in range(n)]
    return x, y, groups


def test_head_picks_c_by_grouped_cv_and_returns_full_probabilities() -> None:
    x, y, groups = separable()
    head = fit_head(x, y, groups, n_classes=3, seed=0)
    assert head.c in head.cv_log_loss and head.cv_log_loss[head.c] == min(head.cv_log_loss.values())
    p = head.predict_proba(x[:5])
    assert p.shape == (5, 3) and np.allclose(p.sum(axis=1), 1.0)


def test_head_on_float32_features_raises_no_probability_warning() -> None:
    # Regression: float32 encoder features gave float32 probabilities that missed
    # sum-to-one by ~1e-7 and made log-loss warn (an error under this suite).
    x, y, groups = separable()
    head = fit_head(x.astype(np.float32), y, groups, n_classes=3, seed=0)
    assert np.allclose(head.predict_proba(x[:5].astype(np.float32)).sum(axis=1), 1.0, atol=1e-12)


def test_head_handles_a_class_missing_from_training() -> None:
    x, y, groups = separable()
    keep = y != 2
    head = fit_head(x[keep], y[keep], [g for g, k in zip(groups, keep, strict=True) if k],
                    n_classes=3, seed=0)
    p = head.predict_proba(x[:5])
    assert p.shape == (5, 3) and np.all(p[:, 2] == 0.0)


def test_head_needs_enough_patients() -> None:
    x, y, _ = separable(20)
    with pytest.raises(ValueError, match="patients"):
        fit_head(x, y, ["a"] * 10 + ["b"] * 10, n_classes=3, seed=0)


# --- stage 1, torch-free parts ----------------------------------------------------


def test_fdi_label_round_trip() -> None:
    from dcai.data.schema import VALID_FDI
    from dcai.models.enumerator import FDI_CLASSES, fdi_to_label, label_to_fdi

    assert len(FDI_CLASSES) == 32 and set(FDI_CLASSES) == VALID_FDI
    assert all(label_to_fdi(fdi_to_label(f)) == f for f in VALID_FDI)
    assert min(fdi_to_label(f) for f in VALID_FDI) == 1  # 0 is background


def test_flip_swaps_quadrants_and_mirrors_boxes() -> None:
    from dcai.models.enumerator import mirror_fdi, mirror_teeth

    assert (mirror_fdi(16), mirror_fdi(26), mirror_fdi(37), mirror_fdi(48)) == (26, 16, 47, 38)
    ((f, b),) = mirror_teeth([(36, BBox(10, 5, 30, 50))], width=100)
    assert f == 46 and (b.x1, b.y1, b.x2, b.y2) == (70, 5, 90, 50)
    assert mirror_teeth(mirror_teeth([(36, b)], 100), 100) == [(36, b)]


def test_enumerator_never_trains_on_held_out_content_or_patients() -> None:
    from dcai.models.enumerator import eligible_training_images

    candidates = [("s1", "g1"), ("s2", "g2"), ("s3", "g3"), ("s4", "g4")]
    held_out = [("s2", "gX"),  # same image as candidate s2
                ("s9", "g3")]  # different image, same recovered patient as s3
    assert eligible_training_images(candidates, held_out) == ["s1", "s4"]


def test_enumeration_score() -> None:
    from dcai.models.enumerator import score_enumeration

    truth = {11: BBox(0, 0, 10, 10), 21: BBox(20, 0, 30, 10), 36: BBox(0, 20, 10, 30)}
    pred = {11: BBox(0, 0, 10, 10), 22: BBox(20, 0, 30, 10), 36: BBox(50, 50, 60, 60)}
    s = score_enumeration(pred, truth)
    assert (s.correct, s.precision, s.recall) == (1, 1 / 3, 1 / 3)
    empty = score_enumeration({}, {})
    assert np.isnan(empty.precision) and np.isnan(empty.recall)
