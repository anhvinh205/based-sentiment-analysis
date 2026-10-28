"""Per-aspect class weights for imbalanced sentiment labels (torch-free)."""

from __future__ import annotations

import numpy as np

from viabsa.constants import N_LABELS


def compute_class_weights(
    y: np.ndarray, power: float = 0.5, max_weight: float = 10.0, n_labels: int = N_LABELS
) -> np.ndarray:
    """Inverse-frequency weights, one row per aspect.

    ``w_c = (N / (K * n_c)) ** power`` where ``K`` is the number of classes that actually occur
    for that aspect. ``power=0`` disables weighting, ``power=1`` is fully balanced; the cap stops
    tiny classes (e.g. 15 negative price reviews) from dominating the loss.

    Returns an array of shape ``(n_aspects, n_labels)``; classes with no examples get weight 1.
    """
    n, n_aspects = y.shape
    weights = np.ones((n_aspects, n_labels), dtype="float32")
    for j in range(n_aspects):
        counts = np.bincount(y[:, j], minlength=n_labels).astype("float64")
        present = counts > 0
        raw = n / (present.sum() * counts[present])
        weights[j, present] = np.minimum(raw**power, max_weight)
    return weights
