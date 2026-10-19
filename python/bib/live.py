"""Continuous authentication from the live ring.

    python -m bib.live --model data/model.pt [--alpha 0.05]

Every time 25 new keystrokes arrive, the last 50 are scored by the LSTM
autoencoder; every 10 new mouse actions, their mean score is scored by the
mouse autoencoder. Each score becomes a conformal p-value against the
enrollment calibration set; the latest p-value of each modality (if fresher
than --horizon seconds) is fused with Fisher's method. Fused p <= alpha means
"this does not look like the owner": the system would ask for an explicit
re-authentication (here it just prints CHALLENGE).

Threshold adaptation (ACI, --aci GAMMA, default 0.02): the threshold alpha_t
moves after every decision,
    accepted                         -> alpha_t += gamma * alpha          (stricter)
    challenged, owner confirms "me"  -> alpha_t += gamma * (alpha - 1)    (looser)
    challenged, nobody confirms      -> no update (treated as an impostor)
The only label used is the outcome of an explicit re-authentication. If an
impostor slips through unchallenged it is counted as "no error", which makes
the threshold stricter, so mislabeling errs in the safe direction. On the CMU
lab/field data this kept every subject's FRR under 2*alpha across 6-8 later
sessions and cut FAR at alpha=0.10 from 0.26 to 0.16 (results/keystroke_drift.json).
"""
from __future__ import annotations

import argparse
import select
import sys
import time
from collections import deque

import numpy as np
import torch

from bib.conformal import fuse, pvalues
from bib.features import keystroke as K
from bib.features import mouse as M
from bib.models.detectors import DenseAE
from bib.models.seq_ae import SeqAE
from bib.ring import KIND_MOUSE, RingReader, stream


def times_for(ev, clock):
    if clock.get("source") == "event_time":
        return (ev["event_time"].astype(np.float64) * clock["ns_per_unit"]).astype(np.int64)
    return ev["timestamp"].astype(np.int64)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/model.pt")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--horizon", type=float, default=120.0, help="seconds a modality's p-value stays valid")
    ap.add_argument("--keep", type=float, default=300.0, help="seconds of raw events kept for feature extraction")
    ap.add_argument("--aci", type=float, default=0.02, help="ACI step size gamma (0 = fixed threshold)")
    args = ap.parse_args(argv)
    m = torch.load(args.model, weights_only=False)
    seq = SeqAE.from_state(m["seq"]) if "seq" in m else None
    mouse = DenseAE.from_state(m["mouse"]) if "mouse" in m else None
    W, S, KA = m["width"], m["stride"], m["k_actions"]
    buf = deque()
    last_press, last_action_end = -1, -1.0  # newest keystroke / action already scored
    latest = {}
    alpha_t = args.alpha
    pending_challenge = None  # wall time of an unconfirmed challenge
    with RingReader(wait_s=30) as rr:
        print(f"live: alpha={args.alpha}  modalities: {'keystroke ' if seq else ''}{'mouse' if mouse else ''}")
        for batch in stream(rr, poll_s=0.2):
            now = time.monotonic()
            buf.append(batch)
            ev = np.concatenate(buf)
            t = times_for(ev, m["clock"])
            cutoff = t[-1] - int(args.keep * 1e9)
            while len(buf) > 1 and times_for(buf[0][-1:], m["clock"])[0] < cutoff:
                buf.popleft()
            # "new" is decided by timestamps, not counts: the buffer is trimmed from the front
            if seq is not None:
                ks = K.events_to_keystrokes(ev, t)
                n_new = int(np.sum(ks["press"] > last_press))
                if len(ks) >= W and n_new >= S:
                    tm = K.timings(ks[-W:])
                    tok, x, _ = K.sequence_windows(tm, W, W)
                    latest["keystroke"] = (now, float(pvalues(m["seq_cal"], seq.score(tok, x))[0]))
                    last_press = int(ks["press"][-1])
            if mouse is not None and np.any(ev["kind"] == KIND_MOUSE):
                F, ends = M.actions_to_features(M.segment_actions(M.samples_from_events(ev, t)), return_end_times=True)
                if len(F) >= KA and int(np.sum(ends > last_action_end)) >= KA:
                    s = mouse.score(F[-KA:]).mean()
                    latest["mouse"] = (now, float(pvalues(m["mouse_cal"], np.array([s]))[0]))
                    last_action_end = float(ends[-1])
            fresh = {k: p for k, (ts, p) in latest.items() if now - ts <= args.horizon}
            if fresh and any(ts == now for ts, _ in latest.values()):
                pf = float(fuse([np.array([p]) for p in fresh.values()], "fisher")[0]) if len(fresh) > 1 else next(iter(fresh.values()))
                challenged = pf <= alpha_t
                if not challenged and args.aci > 0:
                    alpha_t += args.aci * args.alpha
                if challenged and pending_challenge is None:
                    pending_challenge = now
                verdict = "CHALLENGE (type 'me' + Enter if this is you)" if challenged else "ok"
                parts = "  ".join(f"{k} p={p:.3f}" for k, p in fresh.items())
                c = rr.counters()
                print(f"{time.strftime('%H:%M:%S')}  {parts}  fused p={pf:.3f}  thr={max(alpha_t, 0):.3f} -> {verdict}"
                      f"{'  (drops: %d)' % (c.dropped_ring + c.dropped_tap) if c.dropped_ring + c.dropped_tap else ''}",
                      flush=True)
            # explicit re-authentication stand-in: the owner confirms a challenge on stdin
            if pending_challenge is not None and sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline().strip().lower() == "me" and args.aci > 0:
                    alpha_t += args.aci * (args.alpha - 1.0)
                    print(f"   confirmed owner: threshold -> {max(alpha_t, 0):.3f}", flush=True)
                pending_challenge = None
            elif pending_challenge is not None and now - pending_challenge > args.horizon:
                pending_challenge = None  # unconfirmed: treated as impostor, no update


if __name__ == "__main__":
    main()
