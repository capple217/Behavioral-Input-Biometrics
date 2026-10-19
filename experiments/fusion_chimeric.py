"""Keystroke + mouse fusion on chimeric users.

No public dataset has the same people's keystrokes AND mouse data, so the
standard multimodal-biometrics device is used: a "chimeric user" pairs a CMU
keystroke subject with a Balabit mouse user. This assumes the two modalities
are independent given identity (which is also what makes Fisher's combination
valid). Results are averaged over 200 random pairings.

An attempt = one password entry + the first k=10 mouse actions of a session.
  genuine attempt:  genuine keystroke sample + genuine mouse session
  impostor attempt: impostor keystroke sample + impostor mouse session
Per modality: conformal p-value with cross-session calibration
  keystroke: Manhattan (scaled), train sessions 1-3, calibrate session 4, test 5-8
  mouse:     chosen AE, train training-sessions[:-1], calibrate the last one
Fused: Fisher (independence) and 2*mean (any dependence).

    python experiments/fusion_chimeric.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bib.conformal import fuse, pvalues  # noqa: E402
from bib.datasets import CMU_DSL, load_cmu  # noqa: E402
from bib.metrics import eer  # noqa: E402
from bib.models.detectors import AEConfig, DenseAE, ManhattanScaled  # noqa: E402
from bib.parallel import pmap  # noqa: E402
from mouse_balabit import load_cache, window_means  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
K = 10
ALPHA = 0.05
G = {}


def ks_subject(s):
    data, subjects = G["dsl"], G["subjects"]
    X, sess = data[s]["X"], data[s]["session"]
    det = ManhattanScaled().fit(X[sess <= 3])
    cal = det.score(X[sess == 4])
    imp = np.concatenate([data[o]["X"][:5] for o in subjects if o != s])
    return pvalues(cal, det.score(X[sess >= 5])), pvalues(cal, det.score(imp))


def mouse_user(u):
    F, meta, cfg = G["F"], G["meta"], G["cfg"]
    tr = [F[m["key"]] for m in meta if m["split"] == "train" and m["user"] == u]
    det = DenseAE(AEConfig(**cfg)).fit(np.concatenate(tr[:-1]))
    cal = window_means(det.score(tr[-1]), K)
    g, i = [], []
    for m in meta:
        if m["split"] == "test" and m["user"] == u and len(F[m["key"]]) >= K:
            p = float(pvalues(cal, np.array([det.score(F[m["key"]][:K]).mean()]))[0])
            (g if m["label"] == 0 else i).append(p)
    return np.array(g), np.array(i)


def main():
    t0 = time.time()
    dsl = load_cmu(CMU_DSL)
    subjects = sorted(dsl)
    G["dsl"], G["subjects"] = dsl, subjects
    ks = dict(zip(subjects, pmap(ks_subject, subjects)))
    F, meta = load_cache()
    users = sorted({m["user"] for m in meta})
    G["F"], G["meta"] = F, meta
    G["cfg"] = json.loads((OUT / "mouse_balabit.json").read_text())["chosen_ae"]
    ms = dict(zip(users, pmap(mouse_user, users)))
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        pick = rng.choice(subjects, len(users), replace=False)
        for s, u in zip(pick, users):
            kg, ki = ks[s]
            mg, mi = ms[u]
            kgs, kis = rng.choice(kg, len(mg)), rng.choice(ki, len(mi))
            r = {}
            for name, pg, pi in [("keystroke", kgs, kis), ("mouse", mg, mi),
                                 ("fused_fisher", fuse([kgs, mg], "fisher"), fuse([kis, mi], "fisher")),
                                 ("fused_mean2", fuse([kgs, mg], "mean2"), fuse([kis, mi], "mean2"))]:
                r[f"{name}_eer"] = eer(-np.log(pg), -np.log(pi))
                r[f"{name}_frr"] = float(np.mean(pg <= ALPHA))
                r[f"{name}_far"] = float(np.mean(pi > ALPHA))
            rows.append(r)
    res = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
    res |= {"alpha": ALPHA, "k_actions": K, "n_pairings": 200, "runtime_s": round(time.time() - t0, 1)}
    for name in ("keystroke", "mouse", "fused_fisher", "fused_mean2"):
        print(f"  {name:13s} EER {res[name + '_eer']:.3f}   at alpha={ALPHA}: FRR {res[name + '_frr']:.3f}  FAR {res[name + '_far']:.3f}")
    (OUT / "fusion_chimeric.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
