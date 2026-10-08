"""Near-duplicate images: re-encoded, re-saved or lightly altered copies.

SHA-256 proves byte identity. It says nothing about the same radiograph saved
twice with a different encoder, bit depth or crop. A copy is declared here only
when two independent signals agree:

1. Perceptual hash (DCT pHash) Hamming distance at most `max_hamming`.
2. Pearson correlation of standardised low-resolution thumbnails at least
   `min_corr`.

pHash alone collides on similar-looking panoramics; thumbnail correlation alone
cannot separate a copy from the same patient re-imaged. Together they target
"the same pixels". Repeat visits of one patient are a different question,
answered by `probes.recover_groups`. Because thresholds are judgement calls,
`threshold_table` reports counts across a grid of them, so a claim does not rest
on one cutoff.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from dcai.probes.recover_groups import _resize, phash

THUMB = (32, 64)


@dataclass(frozen=True)
class Signature:
    sha256: str
    phash: np.ndarray  # 64 bools
    thumb: np.ndarray  # standardised thumbnail, flattened


def _gray(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    arr = arr.astype(float)
    return arr / (65535.0 if arr.max() > 255 else 255.0)


def signature(sha256: str, png: bytes) -> Signature:
    img = _gray(png)
    thumb = _resize(img, THUMB).ravel()
    sd = thumb.std()
    thumb = (thumb - thumb.mean()) / sd if sd > 0 else thumb * 0.0
    return Signature(sha256, phash(img), thumb)


def _from_zip(job: tuple[str, str, str]) -> tuple[str, list[bool], list[float]]:
    import zipfile

    zip_path, member, sha = job
    with zipfile.ZipFile(zip_path) as zf:
        sig = signature(sha, zf.read(member))
    return sha, sig.phash.tolist(), sig.thumb.tolist()


def signatures_for(
    jobs: Iterable[tuple[str, str, str]], *, cache: Path | None = None, workers: int = 8
) -> dict[str, Signature]:
    """Signatures for (zip path, member, sha256) jobs, one per distinct sha256.

    Decoding thousands of full-size panoramics is the slow part, so results are
    cached by content hash and computed in parallel.
    """
    cached: dict[str, list] = {}
    if cache is not None and cache.exists():
        cached = json.loads(cache.read_text())
    todo, seen = [], set(cached)
    for zip_path, member, sha in jobs:
        if sha not in seen:
            seen.add(sha)
            todo.append((zip_path, member, sha))
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for sha, bits, thumb in pool.map(_from_zip, todo, chunksize=8):
                cached[sha] = [bits, thumb]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cached))
    return {sha: Signature(sha, np.array(b, dtype=bool), np.array(t, dtype=float))
            for sha, (b, t) in cached.items()}


@dataclass(frozen=True)
class NearPair:
    a: str
    b: str
    hamming: int
    corr: float


def candidate_pairs(sigs: Sequence[Signature], *, max_hamming: int) -> list[NearPair]:
    """Every pair of distinct images within `max_hamming`, with its thumbnail correlation."""
    bits = np.packbits(np.array([s.phash for s in sigs]), axis=1).view(np.uint64).ravel()
    thumbs = np.array([s.thumb for s in sigs])
    out = []
    for i in range(len(sigs) - 1):
        ham = np.bitwise_count(bits[i] ^ bits[i + 1:])
        for k in np.flatnonzero(ham <= max_hamming):
            j = i + 1 + int(k)
            corr = float(thumbs[i] @ thumbs[j]) / thumbs.shape[1]
            out.append(NearPair(sigs[i].sha256, sigs[j].sha256, int(ham[k]), corr))
    return out


def near_duplicates(
    sigs: Sequence[Signature], *, max_hamming: int = 6, min_corr: float = 0.95
) -> list[NearPair]:
    return [p for p in candidate_pairs(sigs, max_hamming=max_hamming) if p.corr >= min_corr]


def threshold_table(
    sigs: Sequence[Signature],
    hammings: Sequence[int] = (0, 2, 4, 6, 8, 10),
    corrs: Sequence[float] = (0.90, 0.95, 0.98, 0.99),
) -> dict[tuple[int, float], int]:
    """Pair counts across a grid of thresholds: how robust is the copy count?"""
    pairs = candidate_pairs(sigs, max_hamming=max(hammings))
    return {(h, c): sum(p.hamming <= h and p.corr >= c for p in pairs)
            for h in hammings for c in corrs}
