from __future__ import annotations

from typing import Any

import pytest

from dcai.data.schema import InventorySource, Modality, PatientIdSource, RadiographRecord
from dcai.data.synthetic import make_records


def record(image_id: str, patient_id: str, **overrides: Any) -> RadiographRecord:
    """Minimal valid record, for tests that build partitions by hand."""
    fields: dict[str, Any] = {
        "image_id": image_id,
        "path": f"/x/{image_id}.png",
        "patient_id": patient_id,
        "patient_id_source": PatientIdSource.PROVIDED,
        "dataset": "ds",
        "modality": Modality.BITEWING,
        "readers": frozenset({"consensus"}),
    }
    fields.update(overrides)
    if fields.get("teeth_present") is not None:
        fields.setdefault("teeth_present_source", InventorySource.ANNOTATED)
    return RadiographRecord(**fields)


@pytest.fixture
def bitewing_sets() -> list[RadiographRecord]:
    """Every patient has a full set of four bitewings: maximum leakage potential."""
    return make_records(n_patients=60, images_per_patient=4, seed=1)


@pytest.fixture
def mixed_records() -> list[RadiographRecord]:
    return make_records(n_patients=120, images_per_patient=(1, 4), seed=2)
