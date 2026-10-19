"""Verification metrics. Convention: higher score = more likely impostor."""
from __future__ import annotations

import numpy as np


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Equal error rate: the operating point where FRR == FAR.

    Sweeps every observed score as a threshold (accept iff score <= thr) and
    interpolates linearly between the two thresholds where FRR - FAR changes sign."""
    g = np.sort(np.asarray(genuine, np.float64))
    i = np.sort(np.asarray(impostor, np.float64))
    thr = np.unique(np.r_[g, i, -np.inf])
    frr = 1.0 - np.searchsorted(g, thr, side="right") / len(g)
    far = np.searchsorted(i, thr, side="right") / len(i)
    d = frr - far  # decreasing in thr
    k = int(np.argmax(d <= 0))
    if k == 0:
        return float((frr[0] + far[0]) / 2)
    d0, d1 = d[k - 1], d[k]
    w = d0 / (d0 - d1) if d0 != d1 else 0.5
    return float((1 - w) * (frr[k - 1] + far[k - 1]) / 2 + w * (frr[k] + far[k]) / 2)


def far_frr_at(genuine, impostor, thr):
    return float(np.mean(np.asarray(impostor) <= thr)), float(np.mean(np.asarray(genuine) > thr))
