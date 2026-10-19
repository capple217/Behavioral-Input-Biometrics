"""Enroll the owner from recorded sessions, and measure what can be measured
with one person's data: the genuine-side false-reject rate on a session the
model has never seen.

    python -m bib.enroll --sessions data/sessions --out data/model.pt

With sessions s_1..s_n (time order):
  evaluation (n >= 2): enroll on s_1..s_{n-1}, report FRR on the unseen s_n
  final model:         enroll on all sessions
Enrollment pools every session and holds out 25% of contiguous blocks of each
session for calibration (blocked split: adjacent keystroke windows overlap by
half, so a random window-level split would leak training keystrokes into the
calibration set and make the threshold too tight). The CMU drift study is why
this is pooled rather than "calibrate on a later session": training on fewer,
earlier sessions drifted further from later typing than the calibration
session could absorb (FRR 0.12 vs 0.05 at alpha=0.05 on 126 lab subjects).
Remaining drift is handled online by ACI in bib.live.

FAR cannot be measured from your own data. It needs other people typing and
mousing on your machine (see README: "collecting impostor sessions").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bib.clock import resolve_times
from bib.conformal import fuse, pvalues
from bib.features import keystroke as K
from bib.features import mouse as M
from bib.models.detectors import AEConfig, DenseAE, ManhattanScaled
from bib.models.seq_ae import SeqAE, SeqAEConfig
from bib.record import load_session

ROOT = Path(__file__).resolve().parents[2]
WIDTH, STRIDE, K_ACTIONS = 50, 25, 10


def session_features(path):
    ev, gaps, meta = load_session(path)
    t, fit = resolve_times(ev)
    ks = K.events_to_keystrokes(ev, t, gaps)
    tm = K.timings(ks)
    tok, x, _ = K.sequence_windows(tm, WIDTH, STRIDE)
    summ = K.summary_vectors(tm, WIDTH, STRIDE)
    acts = M.actions_to_features(M.segment_actions(M.samples_from_events(ev, t)))
    return {"path": str(path), "meta": meta, "clock": fit.__dict__, "tok": tok, "x": x, "summ": summ, "mouse": acts}


def _cat(fs, key):
    arrs = [f[key] for f in fs if len(f[key])]
    return np.concatenate(arrs) if arrs else None


def window_means(s, k):
    if len(s) < k:
        return np.empty(0)
    c = np.r_[0, np.cumsum(s)]
    return (c[k:] - c[:-k]) / k


def mouse_cfg():
    p = ROOT / "results" / "mouse_balabit.json"
    if p.exists():
        return json.loads(p.read_text())["chosen_ae"]
    return dict(hidden=32, bottleneck=6, noise=0.1, transform="none", score_norm="l1", n_seeds=3, epochs=300)


def blocked_split(n, block, rng, frac=0.25, guard=1):
    """Boolean masks (train, cal) over n time-ordered samples: whole blocks go to
    calibration; `guard` samples on each side of a calibration block are dropped
    from training (they share raw keystrokes with calibration windows)."""
    if n == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    b = np.arange(n) // block
    nb = int(b.max()) + 1
    k = max(1, int(round(frac * nb))) if nb > 1 else 0
    cal_blocks = rng.choice(nb, k, replace=False) if k else np.array([], int)
    cal = np.isin(b, cal_blocks)
    near = np.zeros(n, bool)
    for g in range(1, guard + 1):
        near[:-g] |= cal[g:]
        near[g:] |= cal[:-g]
    return ~cal & ~near, cal


def split_sessions(feats, seed=0):
    """Per session blocked split -> (train_parts, cal_parts) lists of dicts."""
    rng = np.random.default_rng(seed)
    tr, ca = [], []
    for f in feats:
        mt, mc = blocked_split(len(f["tok"]), 20, rng, guard=1)
        at, ac = blocked_split(len(f["mouse"]), 50, rng, guard=0)
        tr.append({"tok": f["tok"][mt], "x": f["x"][mt], "summ": f["summ"][mt], "mouse": f["mouse"][at]})
        ca.append({"tok": f["tok"][mc], "x": f["x"][mc], "summ": f["summ"][mc], "mouse": f["mouse"][ac]})
    return tr, ca


def fit_models(sessions):
    out = {}
    train, cal = split_sessions(sessions)
    tok, x = _cat(train, "tok"), _cat(train, "x")
    if tok is not None and _cat(cal, "tok") is not None:
        seq = SeqAE(SeqAEConfig(hidden=32, bottleneck=8, epochs=60)).fit(tok, x)
        out["seq"] = seq
        out["seq_cal"] = seq.score(_cat(cal, "tok"), _cat(cal, "x"))
        man = ManhattanScaled().fit(_cat(train, "summ"))
        out["summ"] = man
        out["summ_cal"] = man.score(_cat(cal, "summ"))
    mt, mc = _cat(train, "mouse"), _cat(cal, "mouse")
    if mt is not None and mc is not None and len(mt) > 50:
        ae = DenseAE(AEConfig(**mouse_cfg())).fit(mt)
        out["mouse"] = ae
        out["mouse_cal"] = window_means(ae.score(mc), K_ACTIONS)
    return out


def evaluate(models, test, alphas=(0.05, 0.10)):
    rep = {}
    pk = pm = None
    if "seq" in models and _cat(test, "tok") is not None:
        pk = pvalues(models["seq_cal"], models["seq"].score(_cat(test, "tok"), _cat(test, "x")))
        ps = pvalues(models["summ_cal"], models["summ"].score(_cat(test, "summ")))
        for a in alphas:
            rep[f"keystroke_lstm FRR@{a}"] = float(np.mean(pk <= a))
            rep[f"keystroke_summary_manhattan FRR@{a}"] = float(np.mean(ps <= a))
        rep["keystroke windows"] = int(len(pk))
    if "mouse" in models and _cat(test, "mouse") is not None:
        pm = pvalues(models["mouse_cal"], window_means(models["mouse"].score(_cat(test, "mouse")), K_ACTIONS))
        for a in alphas:
            rep[f"mouse(k={K_ACTIONS}) FRR@{a}"] = float(np.mean(pm <= a))
        rep["mouse windows"] = int(len(pm))
    if pk is not None and pm is not None and len(pk) and len(pm):
        # time alignment is unavailable here; pair in order (both are genuine, so FRR is what matters)
        n = min(len(pk), len(pm))
        pf = fuse([pk[:n], pm[:n]], "fisher")
        for a in alphas:
            rep[f"fused FRR@{a}"] = float(np.mean(pf <= a))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/sessions")
    ap.add_argument("--out", default="data/model.pt")
    args = ap.parse_args(argv)
    paths = sorted(Path(args.sessions).glob("*.npz"))
    if len(paths) < 2:
        sys.exit(f"need at least 2 recorded sessions in {args.sessions} (have {len(paths)})")
    feats = sorted((session_features(p) for p in paths), key=lambda f: f["meta"]["start"])
    for f in feats:
        print(f"  {Path(f['path']).name}: {len(f['tok'])} keystroke windows, {len(f['mouse'])} mouse actions, "
              f"clock={f['clock']['source']}/{f['clock']['unit_guess']} "
              f"(callback latency median {f['clock']['latency_median_us']:.0f} us, p99 {f['clock']['latency_p99_us']:.0f} us)")
    print("held-out check: enroll on sessions[:-1], false-reject rate on the unseen last session")
    report = evaluate(fit_models(feats[:-1]), [feats[-1]])
    for k, v in report.items():
        print(f"  {k:40s} {v:.3f}" if isinstance(v, float) else f"  {k:40s} {v}")
    final = fit_models(feats)
    clock = feats[-1]["clock"]
    blob = {"width": WIDTH, "stride": STRIDE, "k_actions": K_ACTIONS, "clock": clock, "eval": report,
            "sessions": [f["path"] for f in feats]}
    if "seq" in final:
        blob |= {"seq": final["seq"].state(), "seq_cal": final["seq_cal"]}
    if "mouse" in final:
        blob |= {"mouse": final["mouse"].state(), "mouse_cal": final["mouse_cal"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
