"""DENTEX loader (HuggingFace `ibrahimhamamci/DENTEX`, CC-BY-NC-SA 4.0).

Built from what `scripts/audit_dentex.py` found in the files, not from the card:

- **Identity is content, not file name.** File names are reused across subsets
  for *different* images, and the same image appears under different names in
  different subsets. Every image is SHA-256-hashed from the zip stream, and a
  record's `image_id` derives from that hash.
- **Category ids are never positional.** In the quadrant subset, id 0 is named
  "2". Every id is resolved through its category name.
- **Three annotation formats.** COCO for the training subsets. COCO again for
  validation, but with labels in a separate `validation_triple.json` outside its
  zip. LabelMe, one file per image, for test.
- **Junk is skipped by rule.** Only direct children of a subset's `xrays/` folder
  count, which drops `.ipynb_checkpoints/` copies.
- **The tooth inventory comes only from human enumeration.** The diagnosis
  subset boxes diseased teeth only. Where the same image (by content) exists in
  the enumeration subset, its full tooth enumeration becomes `teeth_present`.
  Images where the two subsets disagree on numbering are flagged and get no
  inventory. Tooth-level metrics then refuse them, as they refuse images with no
  enumerated copy at all.
- **The test set is evidence, not evaluation.** Its labels are treatment codes,
  not diagnoses (see README), so they are parsed into `TreatmentLabel`, a type
  no metric accepts. Test images still enter the content index, so copies are
  detected.

Records carry `meta["copies_in"]`: the training subsets holding a copy of the same
image. Figure 1 evaluates with and without those images.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dcai.data.schema import (
    CONSENSUS,
    Annotation,
    BBox,
    Finding,
    InventorySource,
    Modality,
    PatientIdSource,
    RadiographRecord,
)

TRAIN_ZIP, VAL_ZIP, TEST_ZIP = "training_data.zip", "validation_data.zip", "test_data.zip"
VAL_LABELS = "validation_triple.json"

SUBSETS: dict[str, tuple[str, str]] = {
    "quadrant": (TRAIN_ZIP, "training_data/quadrant/xrays/"),
    "enumeration": (TRAIN_ZIP, "training_data/quadrant_enumeration/xrays/"),
    "diagnosis": (TRAIN_ZIP, "training_data/quadrant-enumeration-disease/xrays/"),
    "unlabelled": (TRAIN_ZIP, "training_data/unlabelled/xrays/"),
    "validation": (VAL_ZIP, "validation_data/quadrant_enumeration_disease/xrays/"),
    "test": (TEST_ZIP, "disease/input/"),
}
# Subsets a model may train on (diagnosis is split by patient, so it is not here).
TRAINING_ONLY = ("quadrant", "enumeration", "unlabelled")
COCO_FILES = {
    "quadrant": "training_data/quadrant/train_quadrant.json",
    "enumeration": "training_data/quadrant_enumeration/train_quadrant_enumeration.json",
    "diagnosis": "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
}
TEST_LABEL_DIR = "disease/label/"

FINDINGS = {
    "Caries": Finding.CARIES,
    "Deep Caries": Finding.DEEP_CARIES,
    "Periapical Lesion": Finding.PERIAPICAL_LESION,
    "Impacted": Finding.IMPACTED_TOOTH,
}


# --- content index ----------------------------------------------------------------


@dataclass(frozen=True)
class ImageEntry:
    subset: str
    file_name: str
    sha256: str
    n_bytes: int
    zip_name: str
    member: str


def _members(zf: zipfile.ZipFile, prefix: str) -> list[zipfile.ZipInfo]:
    """PNG files directly inside `prefix`. Nested folders (.ipynb_checkpoints) are junk."""
    return [
        i for i in zf.infolist()
        if i.filename.startswith(prefix) and i.filename.endswith(".png")
        and "/" not in i.filename[len(prefix):]
    ]


def _sha256(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    h = hashlib.sha256()
    with zf.open(info) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_images(raw: Path, *, cache: Path | None = None) -> list[ImageEntry]:
    """SHA-256 of every image in every subset, streamed from the zips.

    The cache is keyed by zip size plus member CRC and size, so a re-downloaded
    or different release is re-hashed rather than trusted.
    """
    cached: dict[str, str] = {}
    if cache is not None and cache.exists():
        cached = json.loads(cache.read_text())
    entries: list[ImageEntry] = []
    zips: dict[str, zipfile.ZipFile] = {}
    try:
        for subset, (zname, prefix) in SUBSETS.items():
            zf = zips.setdefault(zname, zipfile.ZipFile(raw / zname))
            zsize = (raw / zname).stat().st_size
            for info in _members(zf, prefix):
                key = f"{zname}:{zsize}:{info.filename}:{info.CRC}:{info.file_size}"
                digest = cached.get(key) or _sha256(zf, info)
                cached[key] = digest
                entries.append(ImageEntry(subset, info.filename[len(prefix):], digest,
                                          info.file_size, zname, info.filename))
    finally:
        for zf in zips.values():
            zf.close()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached, sort_keys=True))
    return entries


# --- annotation formats -----------------------------------------------------------


def _ids_to_ints(categories: list[dict[str, Any]], allowed: range, what: str) -> dict[int, int]:
    """Category id -> the integer its *name* states. Never id + 1."""
    out = {}
    for c in categories:
        value = int(c["name"])
        if value not in allowed:
            raise ValueError(f"{what} category {c} names {value}, outside {allowed}")
        out[int(c["id"])] = value
    return out


def _bbox(xywh: list[float]) -> BBox:
    x, y, w, h = xywh
    return BBox(x, y, x + w, y + h)


@dataclass(frozen=True)
class CocoImage:
    file_name: str
    height: int
    width: int


def parse_diagnosis_coco(data: dict[str, Any]) -> dict[str, tuple[CocoImage, list[Annotation]]]:
    """Quadrant + enumeration + diagnosis COCO (diagnosis train and validation)."""
    quadrant = _ids_to_ints(data["categories_1"], range(1, 5), "quadrant")
    tooth = _ids_to_ints(data["categories_2"], range(1, 9), "tooth")
    diagnosis = {int(c["id"]): FINDINGS[c["name"]] for c in data["categories_3"]}
    images = {im["id"]: CocoImage(im["file_name"], int(im["height"]), int(im["width"]))
              for im in data["images"]}
    out: dict[str, tuple[CocoImage, list[Annotation]]] = {
        im.file_name: (im, []) for im in images.values()}
    for a in data["annotations"]:
        im = images[a["image_id"]]
        out[im.file_name][1].append(Annotation(
            finding=diagnosis[a["category_id_3"]],
            annotator_id=CONSENSUS,
            bbox=_bbox(a["bbox"]),
            tooth_fdi=quadrant[a["category_id_1"]] * 10 + tooth[a["category_id_2"]],
            meta={"coco_id": a["id"]},
        ))
    return out


def parse_enumeration_coco(data: dict[str, Any]) -> dict[str, list[tuple[int, BBox]]]:
    """Quadrant + enumeration COCO: every tooth box and its FDI number, per file name."""
    quadrant = _ids_to_ints(data["categories_1"], range(1, 5), "quadrant")
    tooth = _ids_to_ints(data["categories_2"], range(1, 9), "tooth")
    names = {im["id"]: im["file_name"] for im in data["images"]}
    out: dict[str, list[tuple[int, BBox]]] = {n: [] for n in names.values()}
    for a in data["annotations"]:
        out[names[a["image_id"]]].append(
            (quadrant[a["category_id_1"]] * 10 + tooth[a["category_id_2"]], _bbox(a["bbox"])))
    return out


@dataclass(frozen=True)
class TreatmentLabel:
    """A released test-set label: a treatment code on a tooth, NOT a diagnosis."""

    code: int
    word: str
    tooth_fdi: int


_LABELME = re.compile(r"(\d+)-(.+)-(\d{2})")


def parse_labelme(data: dict[str, Any]) -> list[TreatmentLabel]:
    out = []
    for shape in data["shapes"]:
        m = _LABELME.fullmatch(shape["label"])
        if m is None:
            raise ValueError(f"unexpected test label {shape['label']!r}")
        out.append(TreatmentLabel(int(m.group(1)), m.group(2), int(m.group(3))))
    return out


# --- the release ------------------------------------------------------------------


@dataclass(frozen=True)
class DentexRelease:
    records: tuple[RadiographRecord, ...]  # diagnosis train + validation: the evaluation set
    index: tuple[ImageEntry, ...]  # every image in every subset
    inventory_conflicts: dict[str, str]  # image id -> why it has no inventory despite a copy
    tooth_boxes: dict[str, dict[int, BBox]]  # image id -> FDI -> human tooth box, inventoried only
    test_labels: dict[str, list[TreatmentLabel]]  # sha256 -> labels; evidence only


def image_id_for(sha256: str) -> str:
    return f"dentex-{sha256[:16]}"


def load_dentex(raw: Path, *, store: Path | None = None,
                index_cache: Path | None = None) -> DentexRelease:
    """Load the evaluation set and the content index.

    `store`: where `extract_images` writes content-addressed copies. Record paths
    point there, as `<store>/<sha256>.png`.
    """
    index = index_images(raw, cache=index_cache)
    sha_of = {(e.subset, e.file_name): e.sha256 for e in index}
    subsets_of: dict[str, set[str]] = {}
    for e in index:
        subsets_of.setdefault(e.sha256, set()).add(e.subset)

    with zipfile.ZipFile(raw / TRAIN_ZIP) as zf:
        diagnosis = parse_diagnosis_coco(json.loads(zf.read(COCO_FILES["diagnosis"])))
        enumeration = parse_enumeration_coco(json.loads(zf.read(COCO_FILES["enumeration"])))
    validation = parse_diagnosis_coco(json.loads((raw / VAL_LABELS).read_text()))
    with zipfile.ZipFile(raw / TEST_ZIP) as zf:
        test_labels = {
            sha_of[("test", Path(i.filename).with_suffix(".png").name)]:
                parse_labelme(json.loads(zf.read(i)))
            for i in zf.infolist()
            if i.filename.startswith(TEST_LABEL_DIR) and i.filename.endswith(".json")
        }

    # Enumeration by content. A tooth boxed twice in one image is ambiguous (which
    # crop is "tooth 36"?), and two copies of an image in (b) must agree.
    boxes_by_sha: dict[str, dict[int, BBox]] = {}
    enum_problem: dict[str, str] = {}
    for name, teeth in enumeration.items():
        sha = sha_of[("enumeration", name)]
        fdis = [f for f, _ in teeth]
        boxes = dict(teeth)
        if len(fdis) != len(boxes):
            enum_problem[sha] = "an FDI number is boxed twice in the enumeration subset"
        elif sha in boxes_by_sha and set(boxes_by_sha[sha]) != set(boxes):
            enum_problem[sha] = "copies in the enumeration subset disagree"
        boxes_by_sha.setdefault(sha, boxes)

    records, seen = [], set()
    conflicts: dict[str, str] = {}
    tooth_boxes: dict[str, dict[int, BBox]] = {}
    for subset, labels in (("diagnosis", diagnosis), ("validation", validation)):
        for name, (im, anns) in labels.items():
            key = (subset, name)
            if key not in sha_of:
                raise ValueError(f"{subset} annotation file lists {name}, which is not in the zip")
            sha = sha_of[key]
            if sha in seen:
                raise ValueError(f"{subset}/{name} duplicates another evaluation image by content")
            seen.add(sha)
            image_id = image_id_for(sha)

            boxes = boxes_by_sha.get(sha)
            problem = enum_problem.get(sha)
            if boxes is not None and problem is None and not (
                    {a.tooth_fdi for a in anns} <= set(boxes)):
                problem = "a diagnosed tooth's FDI number is missing from the enumeration"
            conflict = problem is not None
            if conflict:
                conflicts[image_id] = problem
            inventory = None if conflict or boxes is None else frozenset(boxes)
            if inventory is not None:
                tooth_boxes[image_id] = boxes

            records.append(RadiographRecord(
                image_id=image_id,
                path=str(store / f"{sha}.png") if store else f"zip:{SUBSETS[subset][0]}:{name}",
                patient_id=image_id,
                patient_id_source=PatientIdSource.ASSUMED_UNIQUE,
                dataset="dentex",
                modality=Modality.PANORAMIC,
                readers=frozenset({CONSENSUS}),
                annotations=anns,
                teeth_present=inventory,
                teeth_present_source=InventorySource.ANNOTATED if inventory else None,
                width=im.width,
                height=im.height,
                meta={
                    "sha256": sha,
                    "source": f"{subset}/{name}",
                    "copies_in": sorted(subsets_of[sha] & set(TRAINING_ONLY)),
                    "inventory_conflict": conflict,
                },
            ))
    return DentexRelease(tuple(records), tuple(index), conflicts, tooth_boxes, test_labels)


def extract_images(release: DentexRelease, raw: Path, store: Path,
                   subsets: tuple[str, ...] = ("diagnosis", "validation", "enumeration")) -> int:
    """Write content-addressed copies (`<sha256>.png`) of the given subsets. Idempotent."""
    store.mkdir(parents=True, exist_ok=True)
    written = 0
    zips: dict[str, zipfile.ZipFile] = {}
    try:
        for e in release.index:
            target = store / f"{e.sha256}.png"
            if e.subset not in subsets or target.exists():
                continue
            zf = zips.setdefault(e.zip_name, zipfile.ZipFile(raw / e.zip_name))
            tmp = target.with_suffix(".part")
            tmp.write_bytes(zf.read(e.member))
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != e.sha256:
                tmp.unlink()
                raise ValueError(f"{e.member}: content changed since indexing")
            tmp.rename(target)
            written += 1
    finally:
        for zf in zips.values():
            zf.close()
    return written
