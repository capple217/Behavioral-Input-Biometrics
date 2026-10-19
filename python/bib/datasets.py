"""Public benchmark loaders (fetch with scripts/fetch_data.sh).

CMU Keystroke Dynamics Benchmark (Killourhy & Maxion, DSN 2009)
  data/raw/maxion/projects/keystroke-benchmark/DSL-StrongPasswordData.csv
  51 subjects x 8 sessions x 50 reps of ".tie5Roanl"; 31 timing features
  (H.*, DD.*, UD.*) in seconds. Published EERs for 14 detectors.
CMU lab / field extension (same lab, 126 subjects incl. the 51; field = 105
  different subjects typing on their own machines, up to 15 sessions)
  .../lab-field/FULL_lab_data.csv, FULL_field_data.csv
Balabit Mouse Dynamics Challenge (2016)
  data/raw/balabit/{training_files,test_files}/userN/session_*, public_labels.csv
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("BIB_DATA", ROOT / "data" / "raw"))
CMU_DSL = RAW / "maxion/projects/keystroke-benchmark/DSL-StrongPasswordData.csv"
CMU_LAB = RAW / "maxion/projects/lab-field/FULL_lab_data.csv"
CMU_FIELD = RAW / "maxion/projects/lab-field/FULL_field_data.csv"
BALABIT = RAW / "balabit"

# password key order; "Shift.r" is the capital R typed with shift
CMU_KEYS = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]


def cmu_columns():
    cols = []
    for k, a in enumerate(CMU_KEYS):
        cols.append(f"H.{a}")
        if k + 1 < len(CMU_KEYS):
            b = CMU_KEYS[k + 1]
            cols += [f"DD.{a}.{b}", f"UD.{a}.{b}"]
    return cols


def load_cmu(path: Path = CMU_DSL) -> dict[str, dict]:
    """subject -> {"X": (n,31) seconds, "session": (n,) 1-based session index, "rep": (n,)}
    rows ordered by (session, rep)."""
    cols = cmu_columns()
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["subject"]].append((int(r["sessionIndex"]), int(r["rep"]), [float(r[c]) for c in cols]))
    out = {}
    for s, rr in rows.items():
        rr.sort(key=lambda z: (z[0], z[1]))
        out[s] = {"X": np.array([z[2] for z in rr]), "session": np.array([z[0] for z in rr]),
                  "rep": np.array([z[1] for z in rr])}
    return out


def cmu_to_sequences(X: np.ndarray):
    """(n,31) CMU vectors -> per-key sequence form used by the LSTM model:
    tokens (n,11) and x (n,11,4) = [squash(H), squash(DD), squash(UD), boundary] (ms in)."""
    from bib.features.keystroke import TOKEN, squash

    names = {"period": ".", "five": "5", "Shift.r": "r", "Return": "return"}
    tok = np.array([TOKEN[names.get(k, k)] for k in CMU_KEYS])
    n = len(X)
    H = X[:, 0::3] * 1000.0                      # 11 holds
    DD = np.c_[np.zeros(n), X[:, 1::3] * 1000.0]  # 10 down-down, first masked
    UD = np.c_[np.zeros(n), X[:, 2::3] * 1000.0]
    b = np.zeros((n, 11))
    b[:, 0] = 1
    x = np.stack([squash(H), squash(DD), squash(UD), b], axis=2).astype(np.float32)
    return np.tile(tok, (n, 1)), x


def load_balabit(root: Path = BALABIT):
    """Returns train: user -> list of session paths; test: user -> list of (path, is_illegal)."""
    labels = {}
    with open(root / "public_labels.csv") as f:
        for r in csv.DictReader(f):
            labels[r["filename"]] = int(r["is_illegal"])
    train, test = {}, {}
    for u in sorted(os.listdir(root / "training_files")):
        train[u] = sorted(str(p) for p in (root / "training_files" / u).iterdir())
        test[u] = [(str(p), labels[p.name]) for p in sorted((root / "test_files" / u).iterdir()) if p.name in labels]
    return train, test
