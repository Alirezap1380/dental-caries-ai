"""Stage 1: tooth enumerator, which finds and numbers every tooth. Needs the `[train]` extra.

torchvision Faster R-CNN (BSD-3) rather than Ultralytics YOLO (AGPL-3.0): this
repository is MIT, and an AGPL dependency would reach it. The cost is slower
training on Apple-silicon MPS. The mobile backbone keeps that tolerable.

Leakage rule for training data: subset (b) contains copies of evaluation images.
The enumerator must not train on any image whose *content or recovered
patient* is in the evaluation validation or test partitions. Otherwise
end-to-end tooth-level metrics are measured on images the enumerator has seen.
`eligible_training_images` enforces it.

Augmentation detail that is easy to get wrong: a horizontal flip mirrors the
patient's right and left, so FDI quadrants swap (1<->2, 3<->4). Flipping boxes
without relabelling would teach the model to number every tooth on the wrong side.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from dcai.data.schema import VALID_FDI, BBox

FDI_CLASSES = tuple(sorted(VALID_FDI))  # model label k (1-based) is FDI_CLASSES[k - 1]
_MIRROR = {1: 2, 2: 1, 3: 4, 4: 3}


def fdi_to_label(fdi: int) -> int:
    return FDI_CLASSES.index(fdi) + 1


def label_to_fdi(label: int) -> int:
    return FDI_CLASSES[label - 1]


def mirror_fdi(fdi: int) -> int:
    """FDI number of the same tooth after a left-right flip of the radiograph."""
    q, t = divmod(fdi, 10)
    return _MIRROR[q] * 10 + t


def mirror_teeth(teeth: Sequence[tuple[int, BBox]], width: float) -> list[tuple[int, BBox]]:
    return [(mirror_fdi(f), BBox(width - b.x2, b.y1, width - b.x1, b.y2)) for f, b in teeth]


def eligible_training_images(
    candidates: Iterable[tuple[str, str]],
    held_out: Iterable[tuple[str, str]],
) -> list[str]:
    """Enumeration-subset images the enumerator may train on.

    `candidates`: (sha256, recovered patient group) for each enumeration image.
    `held_out`: the same, for every evaluation validation/test image. A candidate
    is dropped if it shares content OR recovered patient with any held-out image.
    """
    held = list(held_out)
    bad_sha = {s for s, _ in held}
    bad_group = {g for _, g in held}
    return [s for s, g in candidates if s not in bad_sha and g not in bad_group]


@dataclass(frozen=True)
class EnumeratorConfig:
    epochs: int = 12
    lr: float = 0.01
    batch: int = 2
    min_size: int = 640
    max_size: int = 1400
    flip_p: float = 0.5
    seed: int = 0


def _to_tensor(img: np.ndarray):
    import torch

    return torch.from_numpy(np.repeat(img[None].astype(np.float32), 3, axis=0))


def build_model(config: EnumeratorConfig):
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT", min_size=config.min_size,
                                              max_size=config.max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(FDI_CLASSES) + 1)
    return model


def train_enumerator(
    samples: Sequence[tuple[Callable[[], np.ndarray], list[tuple[int, BBox]]]],
    config: EnumeratorConfig,
    *,
    device: str | None = None,
    log: Callable[[str], None] = print,
):
    """Fine-tune a COCO-pretrained detector to box and number teeth."""
    import torch

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(config).to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=1e-4)
    steps = config.epochs * int(np.ceil(len(samples) / config.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.lr, total_steps=steps)
    for epoch in range(config.epochs):
        order = rng.permutation(len(samples))
        losses = []
        for start in range(0, len(order), config.batch):
            images, targets = [], []
            for i in order[start:start + config.batch]:
                load, teeth = samples[i]
                img = load()
                if rng.random() < config.flip_p:
                    img, teeth = img[:, ::-1].copy(), mirror_teeth(teeth, img.shape[1])
                images.append(_to_tensor(img).to(device))
                targets.append({
                    "boxes": torch.from_numpy(
                        np.array([b.as_array() for _, b in teeth], dtype=np.float32).reshape(-1, 4)
                    ).to(device),
                    "labels": torch.tensor([fdi_to_label(f) for f, _ in teeth],
                                           dtype=torch.int64, device=device),
                })
            loss = sum(model(images, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        log(f"epoch {epoch + 1}/{config.epochs}: mean loss {np.mean(losses):.3f}")
    return model.eval()


def predict_teeth(model, img: np.ndarray, *, min_score: float = 0.5,
                  device: str | None = None) -> dict[int, tuple[BBox, float]]:
    """Best box per FDI number above `min_score`: the predicted tooth inventory."""
    import torch

    device = device or next(model.parameters()).device
    with torch.no_grad():
        out = model([_to_tensor(img).to(device)])[0]
    best: dict[int, tuple[BBox, float]] = {}
    for box, label, score in zip(out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(),
                                 out["scores"].cpu().numpy(), strict=True):
        fdi = label_to_fdi(int(label))
        if score >= min_score and (fdi not in best or score > best[fdi][1]):
            best[fdi] = (BBox(*map(float, box)), float(score))
    return best


@dataclass(frozen=True)
class EnumerationScore:
    n_true: int
    n_pred: int
    correct: int  # predicted tooth with the right FDI number and IoU >= threshold

    @property
    def precision(self) -> float:
        return self.correct / self.n_pred if self.n_pred else float("nan")

    @property
    def recall(self) -> float:
        return self.correct / self.n_true if self.n_true else float("nan")


def score_enumeration(pred: dict[int, BBox], truth: dict[int, BBox], *,
                      iou: float = 0.5) -> EnumerationScore:
    """Teeth found AND numbered correctly: the error stage 1 puts in every denominator."""
    correct = sum(1 for f, b in pred.items() if f in truth and b.iou(truth[f]) >= iou)
    return EnumerationScore(len(truth), len(pred), correct)
