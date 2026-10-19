"""Mouse-dynamics benchmark on the Balabit Mouse Dynamics Challenge.

Per user (10): a one-class detector is trained on that user's training
sessions only (no impostor data). Each labeled test session in the user's
folder is either genuine or an impostor session (public_labels.csv).
A session is scored by the mean anomaly score of its first k actions, so k
controls how much interaction the system waits for before deciding.

Model selection uses training files only: train on sessions[:-1] of a user,
genuine = that user's held-out last session, impostors = other users'
training sessions. Test labels are touched once, in the final evaluation.

Conformal thresholds are calibrated on the held-out last training session
(k-action sliding windows), i.e. cross-session calibration.

    python experiments/mouse_balabit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bib.conformal import pvalues  # noqa: E402
from bib.datasets import load_balabit  # noqa: E402
from bib.features.mouse import actions_to_features, samples_from_balabit, segment_actions  # noqa: E402
from bib.metrics import eer  # noqa: E402
from bib.models.detectors import AEConfig, DenseAE, ManhattanScaled  # noqa: E402
from bib.parallel import pmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "balabit_features.npz"
KS = (1, 5, 10, 20, 50)
ALPHAS = (0.05, 0.10)
G = {}


def _feat(path):
    return actions_to_features(segment_actions(samples_from_balabit(path)))


def build_cache():
    train, test = load_balabit()
    jobs = [("train", u, i, p, -1) for u in train for i, p in enumerate(train[u])]
    jobs += [("test", u, i, p, lab) for u in test for i, (p, lab) in enumerate(test[u])]
    feats = pmap(_feat, [j[3] for j in jobs])
    arrays, meta = {}, []
    for (split, u, i, p, lab), F in zip(jobs, feats):
        key = f"{split}|{u}|{i}"
        arrays[key] = F
        meta.append({"key": key, "split": split, "user": u, "idx": i, "label": lab, "file": Path(p).name})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, meta=np.array(json.dumps(meta)), **arrays)


def load_cache():
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE)
    meta = json.loads(str(z["meta"]))
    return {m["key"]: z[m["key"]] for m in meta}, meta


def make(kind, cfg, seed):
    return ManhattanScaled() if kind == "manhattan_scaled" else DenseAE(AEConfig(**cfg), seed=seed)


def window_means(s, k):
    if len(s) < k:
        return np.empty(0)
    c = np.r_[0, np.cumsum(s)]
    return (c[k:] - c[:-k]) / k


def tune_user(args):
    u, kind, cfg = args
    F, meta = G["F"], G["meta"]
    tr = [F[m["key"]] for m in meta if m["split"] == "train" and m["user"] == u]
    other = np.concatenate([F[m["key"]] for m in meta if m["split"] == "train" and m["user"] != u])
    det = make(kind, cfg, 0).fit(np.concatenate(tr[:-1]))
    g, i = det.score(tr[-1]), det.score(other)
    return {k: eer(window_means(g, k), window_means(i, k)) for k in (1, 10)}


def eval_user(args):
    u, kind, cfg = args
    F, meta = G["F"], G["meta"]
    tr = [F[m["key"]] for m in meta if m["split"] == "train" and m["user"] == u]
    tests = [(F[m["key"]], m["label"]) for m in meta if m["split"] == "test" and m["user"] == u]
    det = make(kind, cfg, 0).fit(np.concatenate(tr[:-1]))
    cal = det.score(tr[-1])
    # refit on all training sessions for EER; keep the held-out-session model for conformal
    det_all = make(kind, cfg, 0).fit(np.concatenate(tr))
    out = {"n_genuine": sum(1 for _, l in tests if l == 0), "n_impostor": sum(1 for _, l in tests if l == 1)}
    sc_all = [(det_all.score(X), l) for X, l in tests if len(X)]
    sc_cal = [(det.score(X), l) for X, l in tests if len(X)]
    out["action_eer"] = eer(np.concatenate([s for s, l in sc_all if l == 0]), np.concatenate([s for s, l in sc_all if l == 1]))
    for k in KS + ("all",):
        def sess_score(s):
            return s.mean() if k == "all" else s[:k].mean()
        ok = [(sess_score(s), l) for s, l in sc_all if k == "all" or len(s) >= k]
        g = [v for v, l in ok if l == 0]
        i = [v for v, l in ok if l == 1]
        out[f"k={k}"] = {"eer": eer(g, i) if g and i else float("nan"), "n_sessions": len(ok)}
        if k != "all":
            cw = window_means(cal, k)
            okc = [(sess_score(s), l) for s, l in sc_cal if len(s) >= k]
            gc = np.array([v for v, l in okc if l == 0])
            ic = np.array([v for v, l in okc if l == 1])
            for a in ALPHAS:
                if len(cw) and len(gc) and len(ic):
                    out[f"k={k}"][f"frr@{a}"] = float(np.mean(pvalues(cw, gc) <= a))
                    out[f"k={k}"][f"far@{a}"] = float(np.mean(pvalues(cw, ic) > a))
    out["session_len_actions"] = [int(len(s)) for s, _ in sc_all]
    return out


def main():
    t0 = time.time()
    F, meta = load_cache()
    G["F"], G["meta"] = F, meta
    users = sorted({m["user"] for m in meta})
    res = {"users": users, "n_test_sessions": sum(1 for m in meta if m["split"] == "test")}
    lens = [len(F[m["key"]]) for m in meta if m["split"] == "test"]
    res["test_session_actions"] = {"median": float(np.median(lens)), "p10": float(np.quantile(lens, 0.1)),
                                   "p90": float(np.quantile(lens, 0.9))}
    print(f"users {len(users)}, test sessions {res['n_test_sessions']}, actions/session median {np.median(lens)}")

    grid = [dict(hidden=h, bottleneck=b, noise=0.1, transform="none", score_norm=sn, n_seeds=1, epochs=300)
            for h in (32, 64) for b in (6, 12) for sn in ("l1", "l2")]
    tune = []
    for cfg in grid:
        r = pmap(tune_user, [(u, "ae", cfg) for u in users])
        m = {k: float(np.mean([x[k] for x in r])) for k in (1, 10)}
        tune.append({"cfg": cfg, "train_only_eer_k1": round(m[1], 4), "train_only_eer_k10": round(m[10], 4)})
        print("  tune", cfg, m)
    best = min(tune, key=lambda t: t["train_only_eer_k10"])["cfg"] | {"n_seeds": 3}
    res["tuning"], res["chosen_ae"] = tune, best
    r = pmap(tune_user, [(u, "manhattan_scaled", {}) for u in users])
    res["tuning_manhattan"] = {k: float(np.mean([x[k] for x in r])) for k in (1, 10)}

    for kind, cfg, name in [("manhattan_scaled", {}, "Manhattan (scaled)"), ("ae", best, "Autoencoder (ours)")]:
        rs = pmap(eval_user, [(u, kind, cfg) for u in users])
        agg = {"action_eer": round(float(np.mean([x["action_eer"] for x in rs])), 4)}
        for k in KS + ("all",):
            key = f"k={k}"
            agg[key] = {"eer": round(float(np.nanmean([x[key]["eer"] for x in rs])), 4),
                        "n_sessions": int(sum(x[key]["n_sessions"] for x in rs))}
            for a in ALPHAS:
                vals = [x[key].get(f"frr@{a}") for x in rs if f"frr@{a}" in x[key]]
                if vals:
                    agg[key][f"frr@{a}"] = round(float(np.mean(vals)), 4)
                    agg[key][f"far@{a}"] = round(float(np.mean([x[key][f"far@{a}"] for x in rs if f"far@{a}" in x[key]])), 4)
        agg["per_user_eer_all"] = {u: round(x["k=all"]["eer"], 4) for u, x in zip(users, rs)}
        res[name] = agg
        print(f"== {name}: action EER {agg['action_eer']:.3f}; " +
              "; ".join(f"k={k}: {agg[f'k={k}']['eer']:.3f}" for k in KS + ("all",)))
        for k in KS:
            a = agg[f"k={k}"]
            if "frr@0.05" in a:
                print(f"   k={k:>3}: FRR@.05 {a['frr@0.05']:.3f} FAR@.05 {a['far@0.05']:.3f} | FRR@.10 {a['frr@0.1']:.3f} FAR@.10 {a['far@0.1']:.3f}")
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "mouse_balabit.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
