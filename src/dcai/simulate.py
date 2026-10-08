"""Simulated model outputs for running the harness end to end before real data.

The "model" sees the latent truth through noise, not any reader's labels. That is
roughly what a model trained on consensus labels converges to, and it is what
lets the agreement ceiling be exercised honestly. Every number this produces is a
property of the profile below, not of any learned model. Reports built on it
must say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from dcai.data.synthetic import ReaderStudy, _inside, _jitter, tooth_box
from dcai.eval.detection import Detection
from dcai.eval.units import ToothPrediction


@dataclass(frozen=True)
class ModelProfile:
    # Mean latent logit for a sound, caries and deep-caries tooth.
    means: tuple[float, float, float]
    noise: float  # sd of the latent logit
    sharpness: float  # > 1 pushes probabilities to the extremes (overconfidence)
    box_floor: float  # emit a detection box when P(lesion) >= this


# Internal: shallow lesions are hard, deep ones easy, mildly overconfident.
INTERNAL = ModelProfile(means=(-2.2, -0.5, 0.5), noise=1.1, sharpness=1.3, box_floor=0.15)
# What an overfit network produces on a patient it trained on: it has seen this
# dentition, so it "recognises" the lesions rather than detecting them.
MEMORIZED = ModelProfile(means=(-4.5, 3.0, 4.0), noise=0.6, sharpness=1.3, box_floor=0.15)
# External: a domain shift that costs shallow lesions most.
EXTERNAL = ModelProfile(means=(-2.0, -1.1, 0.2), noise=1.3, sharpness=1.3, box_floor=0.15)


def simulate_model(
    study: ReaderStudy,
    profile: ModelProfile,
    *,
    seed: int,
    memorized: frozenset[str] = frozenset(),
) -> tuple[list[ToothPrediction], list[Detection]]:
    """`memorized`: patient group keys the model trained on (E0 leakage simulation)."""
    rng = np.random.default_rng(seed)
    patient_of = {r.image_id: r.group_key for r in study.records}
    predictions: list[ToothPrediction] = []
    detections: list[Detection] = []
    for (image_id, fdi), grade in sorted(study.truth.items()):
        prof = MEMORIZED if patient_of[image_id] in memorized else profile
        z = rng.normal(prof.means[grade], prof.noise)
        p = float(expit(prof.sharpness * z))
        deep_share = float(expit(2.0 * (z - 0.8)))
        probs = (1.0 - p, p * (1.0 - deep_share), p * deep_share)
        predictions.append(ToothPrediction(image_id, fdi, probs))
        if p >= prof.box_floor:
            box = (
                _jitter(study.lesion_boxes[(image_id, fdi)], rng)
                if grade else _inside(tooth_box(fdi), rng)
            )
            detections.append(Detection(image_id, box, p))
    return predictions, detections
