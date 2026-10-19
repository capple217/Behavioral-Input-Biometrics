"""Conformal calibration of the accept/reject threshold.

Biometric framing: genuine samples are the "inliers". With nonconformity score
s (any detector's anomaly score) and n held-out genuine calibration scores,

    p(s) = (1 + #{cal >= s}) / (n + 1)

and rejecting when p(s) <= alpha gives FRR <= alpha (marginally) **if** the
test genuine sample is exchangeable with the calibration samples. FAR has no
guarantee: it is whatever the detector's separation buys at that threshold.

Exchangeability is exactly what time breaks: typing on day 6 is not drawn from
the day-1 distribution. Two responses are evaluated:
  * cross-session calibration: calibrate on a different session than the one
    the detector was trained on, so the calibration scores already contain
    between-session variation
  * ACI (Gibbs & Candes 2021): alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    updated on genuine samples; long-run FRR -> alpha for any sequence. It
    needs to know which samples are genuine, i.e. an explicit re-authentication
    signal (password / Touch ID) in deployment.

Fusion of modalities happens on p-values:
  * fisher: -2 sum log p ~ chi2(2k) under independence of the modalities
  * mean2:  min(1, 2 * mean p), valid under arbitrary dependence (Vovk & Wang 2020)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def pvalues(cal_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    c = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(c)
    n_ge = n - np.searchsorted(c, np.asarray(scores, dtype=np.float64), side="left")
    return (1.0 + n_ge) / (n + 1.0)


def aci(cal_scores: np.ndarray, genuine_stream: np.ndarray, alpha: float, gamma: float):
    """Run ACI over a time-ordered stream of genuine scores.
    Returns (rejected bool array, alpha_t used for each step)."""
    p = pvalues(cal_scores, genuine_stream)
    a = alpha
    rej = np.zeros(len(p), dtype=bool)
    hist = np.zeros(len(p))
    for t in range(len(p)):
        hist[t] = a
        rej[t] = p[t] <= a
        a = a + gamma * (alpha - float(rej[t]))
    return rej, hist


def fuse(pv: list[np.ndarray], method: str = "fisher") -> np.ndarray:
    P = np.clip(np.stack(pv), 1e-12, 1.0)
    if method == "fisher":
        return stats.chi2.sf(-2 * np.log(P).sum(0), 2 * len(pv))
    if method == "mean2":
        return np.minimum(1.0, 2.0 * P.mean(0))
    raise ValueError(method)
