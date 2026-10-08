from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dcai.data.near_duplicates import (
    near_duplicates,
    signature,
    signatures_for,
    threshold_table,
)
from dcai.data.synthetic_images import _anatomy


def png(img: np.ndarray, *, fmt: str = "PNG", bits16: bool = False) -> bytes:
    buf = io.BytesIO()
    if bits16:
        Image.fromarray((np.clip(img, 0, 1) * 65535).astype(np.uint16)).save(buf, format="PNG")
    else:
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
            buf, format=fmt, **({"quality": 70} if fmt == "JPEG" else {}))
    return buf.getvalue()


def sig(data: bytes):
    return signature(hashlib.sha256(data).hexdigest(), data)


@pytest.fixture(scope="module")
def images() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [np.clip(_anatomy(rng), 0, 1) for _ in range(6)]


def test_reencoded_copies_found_distinct_images_not(images: list[np.ndarray]) -> None:
    base = images[0]
    copies = [png(base), png(base, fmt="JPEG"), png(base, bits16=True),
              png(np.clip(base * 1.03, 0, 1))]  # re-exported, slightly brighter
    others = [png(x) for x in images[1:]]
    sigs = [sig(b) for b in copies + others]
    found = near_duplicates(sigs)
    copy_shas = {s.sha256 for s in sigs[:4]}
    assert {frozenset((p.a, p.b)) for p in found} == {
        frozenset((a, b)) for a in copy_shas for b in copy_shas if a < b}
    assert all(p.corr >= 0.95 and p.hamming <= 6 for p in found)


def test_threshold_table_is_monotone(images: list[np.ndarray]) -> None:
    sigs = [sig(png(images[0])), sig(png(images[0], fmt="JPEG")), *(sig(png(x)) for x in images[1:])]
    table = threshold_table(sigs)
    assert table[(10, 0.90)] >= table[(4, 0.95)] >= table[(0, 0.99)]
    assert table[(6, 0.95)] == 1


def test_signature_of_constant_image() -> None:
    s = sig(png(np.zeros((40, 80))))
    assert np.all(s.thumb == 0)


def test_signatures_from_zip_cached_and_deduplicated(images: list[np.ndarray], tmp_path: Path) -> None:
    data = [png(images[0]), png(images[1])]
    shas = [hashlib.sha256(d).hexdigest() for d in data]
    zpath = tmp_path / "x.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("a.png", data[0])
        z.writestr("b.png", data[1])
        z.writestr("a_copy.png", data[0])
    jobs = [(str(zpath), "a.png", shas[0]), (str(zpath), "b.png", shas[1]),
            (str(zpath), "a_copy.png", shas[0])]
    cache = tmp_path / "sigs.json"
    sigs = signatures_for(jobs, cache=cache, workers=2)
    assert set(sigs) == set(shas) and cache.exists()
    again = signatures_for(jobs, cache=cache, workers=2)
    assert np.array_equal(again[shas[0]].thumb, sigs[shas[0]].thumb)


def test_rgb_and_gray_copies_match(images: list[np.ndarray]) -> None:
    gray = (images[2] * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.stack([gray] * 3, axis=2)).save(buf, format="PNG")
    assert len(near_duplicates([sig(png(images[2])), sig(buf.getvalue())])) == 1
