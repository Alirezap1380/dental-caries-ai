"""Loader tests on a fake release that reproduces every trap the audit found."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from dcai.data.dentex import (
    COCO_FILES,
    SUBSETS,
    TEST_LABEL_DIR,
    TreatmentLabel,
    extract_images,
    image_id_for,
    index_images,
    load_dentex,
    parse_labelme,
)
from dcai.data.schema import Finding, InventorySource
from dcai.eval.ratings import inventory

PNG = {name: f"png-bytes-{name}".encode() for name in ("A", "B", "C", "D", "E", "F", "G")}


def sha(key: str) -> str:
    return hashlib.sha256(PNG[key]).hexdigest()


def coco_triple(images: dict[str, str], boxes: list[tuple[str, int, int, str]]) -> dict:
    """Diagnosis COCO with *permuted* category ids, as the real release has."""
    quad_ids = {2: 0, 1: 1, 3: 2, 4: 3}  # id 0 is named "2"
    tooth_ids = {t: 8 - t for t in range(1, 9)}  # reversed
    diag = {"Impacted": 0, "Caries": 1, "Periapical Lesion": 2, "Deep Caries": 3}
    ids = {name: k + 1 for k, name in enumerate(images)}
    return {
        "images": [{"id": ids[n], "file_name": n, "height": 100, "width": 200} for n in images],
        "annotations": [
            {"id": k, "image_id": ids[n], "bbox": [10, 10, 20, 30],
             "category_id_1": quad_ids[fdi // 10], "category_id_2": tooth_ids[fdi % 10],
             "category_id_3": diag[d]}
            for k, (n, fdi, _, d) in enumerate(boxes)
        ],
        "categories_1": [{"id": i, "name": str(q)} for q, i in quad_ids.items()],
        "categories_2": [{"id": i, "name": str(t)} for t, i in tooth_ids.items()],
        "categories_3": [{"id": i, "name": n} for n, i in diag.items()],
    }


def coco_enum(images: dict[str, list[int]]) -> dict:
    ids = {name: k + 1 for k, name in enumerate(images)}
    return {
        "images": [{"id": ids[n], "file_name": n, "height": 100, "width": 200} for n in images],
        "annotations": [
            {"id": k, "image_id": ids[n], "bbox": [1, 1, 5, 5],
             "category_id_1": fdi // 10 - 1, "category_id_2": fdi % 10 - 1}
            for k, (n, fdi) in enumerate((n, f) for n, fs in images.items() for f in fs)
        ],
        "categories_1": [{"id": q - 1, "name": q} for q in range(1, 5)],  # int names, as in (b)
        "categories_2": [{"id": t - 1, "name": str(t)} for t in range(1, 9)],
    }


FULL_MOUTH = [11, 12, 21, 22, 31, 36, 41, 46]


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    pre = {k: v[1] for k, v in SUBSETS.items()}
    with zipfile.ZipFile(raw / "training_data.zip", "w") as z:
        # "train_1.png" is a different image in each subset (reused name).
        z.writestr(pre["quadrant"] + "train_1.png", PNG["A"])
        z.writestr(pre["quadrant"] + "train_2.png", PNG["B"])  # copy of diagnosis train_9
        z.writestr(pre["enumeration"] + "train_1.png", PNG["C"])
        z.writestr(pre["enumeration"] + "train_5.png", PNG["B"])  # same image as diagnosis train_9
        z.writestr(pre["enumeration"] + "train_6.png", PNG["E"])  # same as diagnosis train_3
        z.writestr(pre["diagnosis"] + "train_9.png", PNG["B"])
        z.writestr(pre["diagnosis"] + "train_3.png", PNG["E"])
        z.writestr(pre["diagnosis"] + "train_1.png", PNG["D"])  # no enumerated copy
        z.writestr(pre["unlabelled"] + "u_1.png", PNG["D"])
        z.writestr(COCO_FILES["diagnosis"], json.dumps(coco_triple(
            {"train_9.png": "B", "train_3.png": "E", "train_1.png": "D"},
            [("train_9.png", 36, 0, "Deep Caries"), ("train_9.png", 21, 0, "Caries"),
             ("train_3.png", 47, 0, "Caries"),  # 47 is not in (b)'s enumeration: conflict
             ("train_1.png", 18, 0, "Impacted")])))
        z.writestr(COCO_FILES["enumeration"], json.dumps(coco_enum({
            "train_1.png": FULL_MOUTH, "train_5.png": FULL_MOUTH, "train_6.png": FULL_MOUTH})))
        z.writestr(COCO_FILES["quadrant"], json.dumps({"images": [], "annotations": []}))
    with zipfile.ZipFile(raw / "validation_data.zip", "w") as z:
        z.writestr(pre["validation"] + "val_1.png", PNG["F"])
        z.writestr(pre["validation"] + ".ipynb_checkpoints/val_1-checkpoint.png", PNG["F"])
    (raw / "validation_triple.json").write_text(json.dumps(coco_triple(
        {"val_1.png": "F"}, [("val_1.png", 46, 0, "Periapical Lesion")])))
    with zipfile.ZipFile(raw / "test_data.zip", "w") as z:
        z.writestr(pre["test"] + "test_1.png", PNG["G"])
        z.writestr(TEST_LABEL_DIR + "test_1.json", json.dumps({"shapes": [
            {"label": "2-küretaj-31"}, {"label": "1-çürük-46"}]}))
    return raw


def test_index_hashes_content_and_skips_checkpoints(raw: Path, tmp_path: Path) -> None:
    cache = tmp_path / "index.json"
    entries = index_images(raw, cache=cache)
    assert {e.sha256 for e in entries if e.subset == "validation"} == {sha("F")}
    assert not any("ipynb" in e.member for e in entries)
    assert len(entries) == 11
    # A cached index is reused, and gives the same answer.
    assert index_images(raw, cache=cache) == entries


def test_identity_is_content_not_file_name(raw: Path) -> None:
    rel = load_dentex(raw)
    assert len(rel.records) == 4  # 3 diagnosis + 1 validation
    by_source = {r.meta["source"]: r for r in rel.records}
    # "train_1.png" in diagnosis is image D, not the quadrant/enumeration images of that name.
    assert by_source["diagnosis/train_1.png"].image_id == image_id_for(sha("D"))
    # train_9 has copies under other names in (a) and (b).
    assert by_source["diagnosis/train_9.png"].meta["copies_in"] == ["enumeration", "quadrant"]
    assert by_source["diagnosis/train_1.png"].meta["copies_in"] == ["unlabelled"]
    assert by_source["validation/val_1.png"].meta["copies_in"] == []


def test_category_ids_resolved_through_names(raw: Path) -> None:
    rec = {r.meta["source"]: r for r in load_dentex(raw).records}["diagnosis/train_9.png"]
    got = {(a.tooth_fdi, a.finding) for a in rec.annotations}
    assert got == {(36, Finding.DEEP_CARIES), (21, Finding.CARIES)}
    a = rec.annotations[0]
    assert (a.bbox.x1, a.bbox.y1, a.bbox.x2, a.bbox.y2) == (10, 10, 30, 40)  # xywh -> xyxy


def test_inventory_from_human_enumeration_only(raw: Path) -> None:
    rel = load_dentex(raw)
    by_source = {r.meta["source"]: r for r in rel.records}
    good = by_source["diagnosis/train_9.png"]
    assert good.teeth_present == frozenset(FULL_MOUTH)
    assert good.teeth_present_source is InventorySource.ANNOTATED
    # Diagnosed tooth 47 is missing from (b)'s enumeration: flagged, no inventory.
    bad = by_source["diagnosis/train_3.png"]
    assert bad.meta["inventory_conflict"] and bad.teeth_present is None
    assert set(rel.inventory_conflicts) == {bad.image_id}
    assert "missing from the enumeration" in rel.inventory_conflicts[bad.image_id]
    assert set(rel.tooth_boxes) == {good.image_id}
    assert set(rel.tooth_boxes[good.image_id]) == set(FULL_MOUTH)
    # No enumerated copy: no inventory, and tooth-level code refuses it.
    none = by_source["diagnosis/train_1.png"]
    assert none.teeth_present is None and not none.meta["inventory_conflict"]
    with pytest.raises(ValueError, match="no tooth inventory"):
        inventory(none)


def test_validation_labels_come_from_the_external_file(raw: Path) -> None:
    val = {r.meta["source"]: r for r in load_dentex(raw).records}["validation/val_1.png"]
    assert [(a.tooth_fdi, a.finding) for a in val.annotations] == [
        (46, Finding.PERIAPICAL_LESION)]


def test_test_labels_are_treatment_codes_kept_as_evidence(raw: Path) -> None:
    rel = load_dentex(raw)
    assert rel.test_labels[sha("G")] == [TreatmentLabel(2, "küretaj", 31),
                                         TreatmentLabel(1, "çürük", 46)]
    assert all(r.meta["source"].split("/")[0] != "test" for r in rel.records)
    with pytest.raises(ValueError, match="unexpected test label"):
        parse_labelme({"shapes": [{"label": "caries on 36"}]})


def test_extract_is_content_addressed_and_idempotent(raw: Path, tmp_path: Path) -> None:
    store = tmp_path / "store"
    rel = load_dentex(raw, store=store)
    n = extract_images(rel, raw, store)
    assert n == 5  # B, E, D, F, C: diagnosis + validation + enumeration, deduplicated
    assert (store / f"{sha('B')}.png").read_bytes() == PNG["B"]
    assert extract_images(rel, raw, store) == 0
    assert all(Path(r.path).exists() for r in rel.records)


def test_duplicate_evaluation_image_is_refused(raw: Path) -> None:
    with zipfile.ZipFile(raw / "validation_data.zip", "a") as z:
        z.writestr(SUBSETS["validation"][1] + "val_2.png", PNG["B"])  # same as diagnosis train_9
    triple = json.loads((raw / "validation_triple.json").read_text())
    triple["images"].append({"id": 99, "file_name": "val_2.png", "height": 1, "width": 1})
    (raw / "validation_triple.json").write_text(json.dumps(triple))
    with pytest.raises(ValueError, match="duplicates another evaluation image"):
        load_dentex(raw)


def test_annotation_for_missing_image_is_refused(raw: Path) -> None:
    triple = json.loads((raw / "validation_triple.json").read_text())
    triple["images"].append({"id": 98, "file_name": "val_9.png", "height": 1, "width": 1})
    (raw / "validation_triple.json").write_text(json.dumps(triple))
    with pytest.raises(ValueError, match="not in the zip"):
        load_dentex(raw)


def test_out_of_range_category_name_is_refused(raw: Path) -> None:
    triple = json.loads((raw / "validation_triple.json").read_text())
    triple["categories_1"][0]["name"] = "5"
    (raw / "validation_triple.json").write_text(json.dumps(triple))
    with pytest.raises(ValueError, match="outside"):
        load_dentex(raw)


def test_disagreeing_copies_inside_enumeration_are_a_conflict(raw: Path) -> None:
    # (b) holds image B twice under two names, enumerated differently.
    with zipfile.ZipFile(raw / "training_data.zip", "a") as z:
        z.writestr(SUBSETS["enumeration"][1] + "train_7.png", PNG["B"])
    with zipfile.ZipFile(raw / "training_data.zip") as z:
        files = {i.filename: z.read(i) for i in z.infolist()}
    enum = json.loads(files[COCO_FILES["enumeration"]])
    nxt = max(im["id"] for im in enum["images"]) + 1
    enum["images"].append({"id": nxt, "file_name": "train_7.png", "height": 1, "width": 1})
    enum["annotations"].append({"id": 999, "image_id": nxt, "bbox": [1, 1, 2, 2],
                                "category_id_1": 0, "category_id_2": 0})
    files[COCO_FILES["enumeration"]] = json.dumps(enum).encode()
    with zipfile.ZipFile(raw / "training_data.zip", "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    rel = load_dentex(raw)
    rec = {r.meta["source"]: r for r in rel.records}["diagnosis/train_9.png"]
    assert rec.meta["inventory_conflict"] and rec.teeth_present is None
    assert "disagree" in rel.inventory_conflicts[rec.image_id]


def test_tooth_boxed_twice_is_a_conflict(raw: Path) -> None:
    with zipfile.ZipFile(raw / "training_data.zip") as z:
        files = {i.filename: z.read(i) for i in z.infolist()}
    enum = json.loads(files[COCO_FILES["enumeration"]])
    first = next(a for a in enum["annotations"] if a["image_id"] == 2)  # train_5 = image B
    enum["annotations"].append({**first, "id": 998})
    files[COCO_FILES["enumeration"]] = json.dumps(enum).encode()
    with zipfile.ZipFile(raw / "training_data.zip", "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    rel = load_dentex(raw)
    rec = {r.meta["source"]: r for r in rel.records}["diagnosis/train_9.png"]
    assert rec.teeth_present is None
    assert "boxed twice" in rel.inventory_conflicts[rec.image_id]


def test_extract_refuses_content_that_changed_since_indexing(raw: Path, tmp_path: Path) -> None:
    rel = load_dentex(raw)
    with zipfile.ZipFile(raw / "training_data.zip") as z:
        files = {i.filename: z.read(i) for i in z.infolist()}
    files[SUBSETS["diagnosis"][1] + "train_9.png"] = b"tampered"
    files[SUBSETS["enumeration"][1] + "train_5.png"] = b"tampered"
    files[SUBSETS["quadrant"][1] + "train_2.png"] = b"tampered"
    with zipfile.ZipFile(raw / "training_data.zip", "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    with pytest.raises(ValueError, match="content changed since indexing"):
        extract_images(rel, raw, tmp_path / "store")
