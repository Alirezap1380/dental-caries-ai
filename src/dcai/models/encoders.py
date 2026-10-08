"""Tooth crops and frozen ImageNet encoders. Requires the `[train]` extra (torch).

The first stage-2 model is deliberately the weakest reasonable one: a frozen
ImageNet ResNet-50 as a feature extractor and a linear head on top. About 700
images would overfit a full fine-tune, and a frozen encoder makes the honest
baseline cheap to reproduce.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from dcai.data.schema import BBox

CROP = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_tooth(img: np.ndarray, box: BBox, *, margin: float = 0.15, size: int = CROP) -> np.ndarray:
    """Square crop around a tooth with some context, letterboxed, resized to `size`.

    Letterboxing (not stretching) keeps the tooth's aspect ratio. Panoramic teeth
    are tall and narrow, and stretching them would distort exactly the crown
    and root shapes the classifier reads.
    """
    h, w = img.shape
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    x1 = int(max(0, np.floor(box.x1 - margin * bw)))
    y1 = int(max(0, np.floor(box.y1 - margin * bh)))
    x2 = int(min(w, np.ceil(box.x2 + margin * bw)))
    y2 = int(min(h, np.ceil(box.y2 + margin * bh)))
    patch = img[y1:y2, x1:x2]
    side = max(patch.shape)
    canvas = np.zeros((side, side), dtype=np.float32)
    oy, ox = (side - patch.shape[0]) // 2, (side - patch.shape[1]) // 2
    canvas[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = patch
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.Resampling.BILINEAR),
                      dtype=np.float32)


@dataclass(frozen=True)
class ToothKey:
    image_id: str
    tooth_fdi: int


class FrozenResNet50:
    """ImageNet ResNet-50 with the classifier removed: 2048-d pooled features."""

    def __init__(self, device: str | None = None) -> None:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        """(n, H, W) grey crops in [0, 1] -> (n, 2048) features."""
        x = np.repeat(crops[:, None, :, :], 3, axis=1)
        x = (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
        with self.torch.no_grad():
            out = self.model(self.torch.from_numpy(x.astype(np.float32)).to(self.device))
        return out.cpu().numpy()


def tooth_features(
    jobs: Sequence[tuple[str, Callable[[], np.ndarray], dict[int, BBox]]],
    encoder: Callable[[np.ndarray], np.ndarray],
    *,
    batch: int = 64,
) -> tuple[list[ToothKey], np.ndarray]:
    """Features for every tooth of every image.

    `jobs`: (image_id, loader returning the grey image in [0, 1], FDI -> box).
    Images are loaded one at a time, so memory holds one panoramic plus a batch.
    """
    keys: list[ToothKey] = []
    feats: list[np.ndarray] = []
    pending: list[np.ndarray] = []

    def flush() -> None:
        if pending:
            feats.append(encoder(np.stack(pending)))
            pending.clear()

    for image_id, load, boxes in jobs:
        img = load()
        for fdi in sorted(boxes):
            keys.append(ToothKey(image_id, fdi))
            pending.append(crop_tooth(img, boxes[fdi]))
            if len(pending) == batch:
                flush()
    flush()
    return keys, (np.concatenate(feats) if feats else np.zeros((0, 0)))
