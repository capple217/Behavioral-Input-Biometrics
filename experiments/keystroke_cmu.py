"""Keystroke benchmark on the CMU strong-password data (Killourhy & Maxion 2009).

  1. Harness validation: re-implement three of their detectors and require our
     EERs to match the published ones before trusting anything else.
  2. Hyperparameter tuning for our models on a DISJOINT tuning population:
     the 75 lab subjects that are not among the 51 benchmark subjects.
  3. Benchmark: frozen configs on the 51 subjects, K&M protocol exactly
     (train on reps 1-200, genuine test 201-400, impostors = first 5 reps of
     each of the other 50 subjects), mean (sd) EER across subjects.
  4. Conformal thresholds: does the calibrated FRR hold on later sessions?

    python experiments/keystroke_cmu.py [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bib.conformal import pvalues  # noqa: E402
from bib.datasets import CMU_DSL, CMU_LAB, cmu_to_sequences, load_cmu  # noqa: E402
from bib.metrics import eer  # noqa: E402
from bib.models.detectors import AEConfig, DenseAE, Euclidean, MahalanobisNN, ManhattanScaled  # noqa: E402
from bib.models.seq_ae import SeqAE, SeqAEConfig  # noqa: E402
from bib.parallel import pmap  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
PUBLISHED = {"Manhattan (scaled)": (0.096, 0.069), "Nearest neighbor (Mahalanobis)": (0.100, 0.064),
             "Euclidean": (0.171, 0.095), "Neural Network (auto-assoc) [K&M]": (0.161, 0.080)}

G = {}  # per-process globals set before pmap (fork inherits them)


def make(kind, cfg, seed):
    if kind == "euclidean":
        return Euclidean()
    if kind == "manhattan_scaled":
        return ManhattanScaled()
    if kind == "mahalanobis_nn":
        return MahalanobisNN()
    if kind == "ae":
        return DenseAE(AEConfig(**cfg), seed=seed)
    if kind == "seq":
        return SeqAE(SeqAEConfig(**cfg), seed=seed)
    raise ValueError(kind)


def fit_score(det, Xtr, Xtests):
    if isinstance(det, SeqAE):
        det.fit(*cmu_to_sequences(Xtr))
        return [det.score(*cmu_to_sequences(X)) for X in Xtests]
    det.fit(Xtr)
    return [det.score(X) for X in Xtests]


def km_subject(args):
    """K&M protocol for one genuine subject. Returns EER."""
    s, kind, cfg = args
    data, subjects = G["data"], G["subjects"]
    X = data[s]["X"]
    imp = np.concatenate([data[o]["X"][:5] for o in subjects if o != s])
    det = make(kind, cfg, seed=subjects.index(s))
    g, i = fit_score(det, X[:200], [X[200:400], imp])
    return eer(g, i)


def run_km(data, subjects, kind, cfg=None):
    G["data"], G["subjects"] = data, subjects
    e = np.array(pmap(km_subject, [(s, kind, cfg or {}) for s in subjects]))
    return float(e.mean()), float(e.std(ddof=1)), e


def conformal_subject(args):
    """Train/calibrate on sessions 1-4 two ways, test on sessions 5-8."""
    s, kind, cfg, mode = args
    data, subjects = G["data"], G["subjects"]
    d = data[s]
    X, sess = d["X"], d["session"]
    enroll = np.nonzero(sess <= 4)[0]
    if mode == "random":
        rng = np.random.default_rng(subjects.index(s))
        cal = rng.choice(enroll, 50, replace=False)
        tr = np.setdiff1d(enroll, cal)
    else:  # cross-session: train on sessions 1-3, calibrate on session 4
        tr, cal = np.nonzero(sess <= 3)[0], np.nonzero(sess == 4)[0]
    test = np.nonzero(sess >= 5)[0]
    imp = np.concatenate([data[o]["X"][:5] for o in subjects if o != s])
    det = make(kind, cfg, seed=subjects.index(s))
    c, g, i = fit_score(det, X[tr], [X[cal], X[test], imp])
    out = {"eer": eer(g, i)}
    pg, pi = pvalues(c, g), pvalues(c, i)
    gs = sess[test]
    for a in (0.05, 0.10):
        out[f"frr@{a}"] = float(np.mean(pg <= a))
        out[f"far@{a}"] = float(np.mean(pi > a))
        out[f"frr_by_session@{a}"] = [float(np.mean(pg[gs == k] <= a)) for k in range(5, 9)]
    return out


def run_conformal(data, subjects, kind, cfg, mode):
    G["data"], G["subjects"] = data, subjects
    rs = pmap(conformal_subject, [(s, kind, cfg, mode) for s in subjects])
    agg = {"eer_mean": float(np.mean([r["eer"] for r in rs]))}
    for a in (0.05, 0.10):
        frr = np.array([r[f"frr@{a}"] for r in rs])
        agg[f"frr@{a}"] = float(frr.mean())
        agg[f"frr_subjects_over_2x@{a}"] = float(np.mean(frr > 2 * a))
        agg[f"far@{a}"] = float(np.mean([r[f"far@{a}"] for r in rs]))
        agg[f"frr_by_session@{a}"] = np.mean([r[f"frr_by_session@{a}"] for r in rs], axis=0).round(4).tolist()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids, for smoke testing")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    dsl = load_cmu(CMU_DSL)
    lab = load_cmu(CMU_LAB)
    bench = sorted(dsl)
    tune_subj = sorted(s for s in lab if s not in dsl)
    assert not set(bench) & set(tune_subj)
    res = {"n_benchmark_subjects": len(bench), "n_tuning_subjects": len(tune_subj)}

    # 1. harness validation
    print("== 1. harness validation (51 benchmark subjects)")
    res["validation"] = {}
    for kind, name in [("euclidean", "Euclidean"), ("manhattan_scaled", "Manhattan (scaled)"),
                       ("mahalanobis_nn", "Nearest neighbor (Mahalanobis)")]:
        m, sd, _ = run_km(dsl, bench, kind)
        res["validation"][name] = {"ours": [round(m, 4), round(sd, 4)], "published": PUBLISHED[name]}
        print(f"  {name:32s} ours {m:.3f} ({sd:.3f})   published {PUBLISHED[name][0]:.3f} ({PUBLISHED[name][1]:.3f})")

    # 2. tuning on disjoint subjects
    print(f"== 2. tuning on {len(tune_subj)} disjoint lab subjects")
    if args.quick:
        ae_grid = [dict(hidden=32, bottleneck=8, noise=0.1, transform="log", n_seeds=1, epochs=100)]
        seq_grid = [dict(hidden=16, bottleneck=4, epochs=20)]
        tune_subj = tune_subj[:12]
    else:
        ae_grid = [dict(hidden=h, bottleneck=b, noise=nz, transform=tf, score_norm=sn, n_seeds=1)
                   for h, b, nz, tf, sn in itertools.product([16, 64], [4, 12], [0.0, 0.2], ["log", "none"], ["l1", "l2"])]
        seq_grid = [dict(hidden=h, bottleneck=b, epochs=ep)
                    for h, b, ep in itertools.product([16, 32], [4, 8], [60, 150])]
    tune = {"ae": [], "seq": []}
    for kind, grid in [("ae", ae_grid), ("seq", seq_grid)]:
        for cfg in grid:
            m, sd, _ = run_km(lab, tune_subj, kind, cfg)
            tune[kind].append({"cfg": cfg, "eer": round(m, 4), "sd": round(sd, 4)})
            print(f"  {kind} {cfg} -> {m:.4f}")
    best_ae = min(tune["ae"], key=lambda r: r["eer"])["cfg"] | {"n_seeds": 1 if args.quick else 3}
    best_seq = min(tune["seq"], key=lambda r: r["eer"])["cfg"]
    res["tuning"] = tune
    res["chosen"] = {"ae": best_ae, "seq": best_seq}
    print("  chosen:", res["chosen"])

    # 3. benchmark with frozen configs
    print("== 3. benchmark (51 subjects, frozen configs)")
    res["benchmark"] = {}
    for kind, cfg, name in [("manhattan_scaled", {}, "Manhattan (scaled)"), ("ae", best_ae, "Autoencoder (ours)"),
                            ("seq", best_seq, "LSTM autoencoder (ours)")]:
        m, sd, per = run_km(dsl, bench, kind, cfg)
        res["benchmark"][name] = {"eer": round(m, 4), "sd": round(sd, 4), "per_subject": per.round(4).tolist()}
        print(f"  {name:28s} {m:.3f} ({sd:.3f})")
    a = np.array(res["benchmark"]["Autoencoder (ours)"]["per_subject"])
    b = np.array(res["benchmark"]["Manhattan (scaled)"]["per_subject"])
    from scipy.stats import wilcoxon
    res["benchmark"]["ae_vs_manhattan_wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
    res["benchmark"]["ae_better_subjects"] = int(np.sum(a < b))

    # 4. conformal thresholds
    print("== 4. conformal calibration, test on sessions 5-8")
    res["conformal"] = {}
    for kind, cfg, name in [("manhattan_scaled", {}, "Manhattan (scaled)"), ("ae", best_ae, "Autoencoder (ours)")]:
        for mode in ("random", "cross_session"):
            agg = run_conformal(dsl, bench, kind, cfg, mode)
            res["conformal"][f"{name} / {mode}"] = agg
            print(f"  {name:22s} {mode:13s} FRR@.05={agg['frr@0.05']:.3f} FAR@.05={agg['far@0.05']:.3f} "
                  f"FRR@.10={agg['frr@0.1']:.3f} FAR@.10={agg['far@0.1']:.3f}  by-session@.05={agg['frr_by_session@0.05']}")
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / ("keystroke_cmu_quick.json" if args.quick else "keystroke_cmu.json")).write_text(json.dumps(res, indent=1))
    print(f"done in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
