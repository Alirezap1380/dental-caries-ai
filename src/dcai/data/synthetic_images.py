"""Synthetic radiograph *files*, for testing the group-recovery probes.

`synthetic.py` makes annotation records with no pixels. The recovery probes work
on the image files themselves, so this module writes small panoramic-like
images that carry the two structures the probes look for:

- **Patients.** Each patient gets a fixed anatomy (arch shape, tooth spacing and
  size, missing teeth, bright restorations). A repeat visit is the same anatomy,
  slightly shifted, with fresh noise.
- **Sites.** Each site has its own acquisition chain: image size, bit depth,
  file format, JPEG quality, exposure and noise texture. Caries prevalence also
  differs by site, so a model could learn "site" as a stand-in for "disease".

None of this resembles real anatomy beyond what the probes need. The files are
written to a caller-supplied directory and never committed (`*.png`/`*.jpg` are
gitignored).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, shift


@dataclass(frozen=True)
class SiteProfile:
    name: str
    size: tuple[int, int]  # (height, width)
    bit_depth: int  # 8 or 16
    file_format: str  # "PNG" or "JPEG"
    jpeg_quality: int | None
    exposure: float  # multiplies the image before noise
    noise_sd: float
    noise_corr: float  # gaussian sigma of the noise; 0 = white
    caries_prevalence: float


SITES = (
    SiteProfile("site_a", (128, 256), 8, "PNG", None, 0.95, 0.03, 0.0, 0.20),
    SiteProfile("site_b", (120, 264), 8, "JPEG", 80, 1.10, 0.05, 1.5, 0.35),
    SiteProfile("site_c", (136, 248), 16, "PNG", None, 0.80, 0.015, 0.0, 0.55),
)


@dataclass(frozen=True)
class SyntheticImage:
    image_id: str
    path: Path
    true_patient: str
    true_site: str
    has_lesion: bool


def _anatomy(rng: np.random.Generator, h: int = 256, w: int = 512) -> np.ndarray:
    """A patient-specific arch of teeth on a smooth background, on a fixed canvas."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = gaussian_filter(rng.normal(0.25, 0.05, (h, w)), 20)
    arch_depth = rng.uniform(30, 60)
    spacing = rng.uniform(26, 34)
    for row, sign in ((0.38, -1), (0.62, 1)):
        for i in range(16):
            if rng.random() < 0.15:  # missing tooth
                continue
            cx = w / 2 + (i - 7.5) * spacing
            cy = h * row + sign * arch_depth * ((cx - w / 2) / (w / 2)) ** 2
            rx, ry = rng.uniform(9, 13), rng.uniform(22, 30)
            tooth = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 2)
            brightness = 0.95 if rng.random() < 0.12 else rng.uniform(0.45, 0.6)  # restoration
            img += brightness * tooth
    return img


def _render(anatomy: np.ndarray, site: SiteProfile, rng: np.random.Generator,
            lesion: bool) -> np.ndarray:
    img = shift(anatomy, rng.uniform(-3, 3, size=2), mode="nearest")
    if lesion:
        cy, cx = rng.uniform(0.3, 0.7) * img.shape[0], rng.uniform(0.2, 0.8) * img.shape[1]
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        img -= 0.25 * np.exp(-(((xx - cx) / 6) ** 2 + ((yy - cy) / 6) ** 2))
    h, w = site.size
    img = np.asarray(Image.fromarray(img.astype(np.float32)).resize((w, h), Image.BILINEAR))
    img = img * site.exposure
    noise = rng.normal(0, 1, img.shape)
    if site.noise_corr:
        noise = gaussian_filter(noise, site.noise_corr)
        noise /= noise.std()
    return np.clip(img + site.noise_sd * noise, 0.0, 1.0)


def _save(img: np.ndarray, site: SiteProfile, path: Path) -> None:
    if site.bit_depth == 16:
        Image.fromarray((img * 65535).astype(np.uint16)).save(path, format="PNG")
    elif site.file_format == "JPEG":
        Image.fromarray((img * 255).astype(np.uint8)).save(path, format="JPEG",
                                                          quality=site.jpeg_quality)
    else:
        Image.fromarray((img * 255).astype(np.uint8)).save(path, format="PNG")


def make_image_study(
    out_dir: Path,
    *,
    n_patients: int,
    seed: int,
    repeat_fraction: float = 0.3,
    sites: tuple[SiteProfile, ...] = SITES,
) -> list[SyntheticImage]:
    """Write one image per patient, plus a repeat visit for `repeat_fraction` of them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rendered: list[tuple[np.ndarray, SiteProfile, str, bool]] = []
    for p in range(n_patients):
        site = sites[p % len(sites)]
        anatomy = _anatomy(rng)
        visits = 2 if rng.random() < repeat_fraction else 1
        for _ in range(visits):
            lesion = bool(rng.random() < site.caries_prevalence)
            rendered.append((_render(anatomy, site, rng, lesion), site, f"p{p:04d}", lesion))

    # Ids are assigned after shuffling: sequential ids would put a repeat visit
    # next to its partner, and the probe would be "recovering" the file listing.
    images: list[SyntheticImage] = []
    for i, j in enumerate(rng.permutation(len(rendered))):
        img, site, patient, lesion = rendered[j]
        ext = "jpg" if site.file_format == "JPEG" else "png"
        path = out_dir / f"img{i:04d}.{ext}"
        _save(img, site, path)
        images.append(SyntheticImage(f"img{i:04d}", path, patient, site.name, lesion))
    return images
