"""Recover patient and site structure from image files and write a report.

    python scripts/recover_groups.py --synthetic --out results/synthetic_recovery

Real data needs a loader (blocked on data access). The synthetic mode writes
images to a temporary directory, never into the repo.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from dcai.data.schema import Modality, PatientIdSource, RadiographRecord
from dcai.data.synthetic_images import make_image_study
from dcai.probes.recover_groups import recovery_markdown, run_recovery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-patients", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    if not args.synthetic:
        print("real-data loaders are not implemented yet (blocked on data access)",
              file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        images = make_image_study(Path(tmp), n_patients=args.n_patients, seed=args.seed)
        # As a release with no patient ids would arrive: one "patient" per image.
        shipped = [
            RadiographRecord(
                image_id=i.image_id, path=str(i.path), patient_id=i.image_id,
                patient_id_source=PatientIdSource.ASSUMED_UNIQUE, dataset="synthetic_images",
                modality=Modality.PANORAMIC, readers=frozenset({"consensus"}),
            )
            for i in images
        ]
        report = run_recovery(
            shipped, {i.image_id: i.path for i in images},
            {i.image_id: i.has_lesion for i in images}, seed=args.seed, n_boot=args.n_boot,
        )

    truth = {frozenset((a.image_id, b.image_id)) for a in images for b in images
             if a.image_id < b.image_id and a.true_patient == b.true_patient}
    found = {frozenset((p.a, p.b)) for p in report.patients.pairs}
    hits = len(found & truth)
    validation = (
        "\n## Validation against the generator's ground truth\n\n"
        f"Only possible on synthetic data. True same-patient pairs: {len(truth)}; declared: "
        f"{len(found)}; correct: {hits} (precision {hits / max(len(found), 1):.3f}, recall "
        f"{hits / max(len(truth), 1):.3f}). These are the easy case by construction: repeat "
        "visits here differ only by a small shift and fresh noise.\n"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "report.md"
    path.write_text(recovery_markdown(report, synthetic=True) + validation)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
