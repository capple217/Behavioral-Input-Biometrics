"""Synthetic InputEvent sessions, for exercising the pipeline without a Mac.

Not a model of real behaviour and never used for any reported number: it
exists so record -> enroll -> live and the feature extractors can be tested
on Linux CI. A "person" is a seed that fixes per-key hold times, a typing
speed, and a mouse speed/curvature style.

    python -m bib.synth --out data/synth --person 1 --sessions 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bib.features.keystroke import MOD_BITS, VK
from bib.record import save_session
from bib.ring import (DOP_DOWN, DOP_DRAG, DOP_NONE, DOP_UP, EVENT_DTYPE, KIND_FLAGS, KIND_KEY, KIND_MOUSE,
                      SIDE_LEFT, SIDE_NONE)

LETTER_CODES = [c for c, n in VK.items() if len(n) == 1 and n.isalpha()] + [49] * 6  # extra spaces


def person(seed):
    r = np.random.default_rng(seed)
    return {"hold": {c: r.uniform(60, 140) for c in VK}, "flight": r.uniform(90, 220), "mouse_speed": r.uniform(0.6, 1.6),
            "curve": r.uniform(-0.3, 0.3), "shift_hold": r.uniform(120, 260)}


def session(p, seed, minutes=3.0, t0_ns=10**12):
    r = np.random.default_rng(seed)
    rows, t = [], float(t0_ns)

    def add(kind, dop, code=0, flags=0, side=SIDE_NONE, x=0.0, y=0.0, dx=0, dy=0, aux=0, at=None):
        rows.append((kind, dop, code, flags, side, aux, x, y, dx, dy, int(at), int(at)))

    end = t0_ns + minutes * 60e9
    x, y = 500.0, 400.0
    while t < end:
        if r.random() < 0.6:  # a burst of typing
            for _ in range(r.integers(20, 80)):
                code = int(r.choice(LETTER_CODES))
                use_shift = r.random() < 0.05
                if use_shift:
                    add(KIND_FLAGS, DOP_NONE, 56, MOD_BITS[56] | 0x20000, at=t)
                    t += r.normal(60, 10) * 1e6
                h = max(20.0, r.normal(p["hold"][code], 12)) * 1e6
                add(KIND_KEY, DOP_DOWN, code, at=t)
                if r.random() < 0.01:  # an autorepeat burst that must be ignored
                    add(KIND_KEY, DOP_DOWN, code, aux=1, at=t + 0.5 * h)
                add(KIND_KEY, DOP_UP, code, at=t + h)
                if use_shift:
                    add(KIND_FLAGS, DOP_NONE, 56, 0, at=t + h + 20e6)
                t += max(30.0, r.lognormal(np.log(p["flight"]), 0.35)) * 1e6
            t += r.uniform(0.5, 3) * 1e9
        else:  # a few pointing movements, some with clicks or drags
            for _ in range(r.integers(2, 6)):
                tx, ty = r.uniform(0, 1400), r.uniform(0, 850)
                dist = np.hypot(tx - x, ty - y)
                dur = (0.25 + dist / 1500.0) / p["mouse_speed"]
                n = max(5, int(dur / 0.008))
                s = np.linspace(0, 1, n)
                prof = 10 * s**3 - 15 * s**4 + 6 * s**5  # minimum-jerk
                bend = p["curve"] * dist * np.sin(np.pi * prof)
                nx, ny = (-(ty - y), tx - x) / max(dist, 1e-9)
                drag = r.random() < 0.15
                if drag:
                    add(KIND_MOUSE, DOP_DOWN, side=SIDE_LEFT, x=x, y=y, at=t)
                px, py = x, y
                for k in range(n):
                    cx = x + (tx - x) * prof[k] + nx * bend[k] + r.normal(0, 0.4)
                    cy = y + (ty - y) * prof[k] + ny * bend[k] + r.normal(0, 0.4)
                    tt = t + k * 0.008e9
                    add(KIND_MOUSE, DOP_DRAG if drag else DOP_NONE, side=SIDE_LEFT if drag else SIDE_NONE, x=cx, y=cy,
                        dx=int(cx - px), dy=int(cy - py), at=tt)
                    px, py = cx, cy
                t += n * 0.008e9
                x, y = tx, ty
                if drag or r.random() < 0.5:
                    if not drag:
                        add(KIND_MOUSE, DOP_DOWN, side=SIDE_LEFT, x=x, y=y, at=t)
                    t += r.normal(90, 15) * 1e6
                    add(KIND_MOUSE, DOP_UP, side=SIDE_LEFT, x=x, y=y, at=t)
                t += r.uniform(0.6, 2.0) * 1e9
    ev = np.array(rows, dtype=EVENT_DTYPE)
    return ev[np.argsort(ev["timestamp"], kind="stable")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--person", type=int, default=1)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--bin", action="store_true", help="also write raw .bin for fake_producer --replay")
    args = ap.parse_args(argv)
    p = person(args.person)
    for i in range(args.sessions):
        ev = session(p, seed=args.person * 100 + i, minutes=args.minutes)
        path = Path(args.out) / f"synth_p{args.person}_{i:02d}.npz"
        save_session(path, ev, [], {"start": f"2026-01-{i + 1:02d}T00:00:00", "synthetic": True, "person": args.person})
        if args.bin:
            ev.tofile(path.with_suffix(".bin"))
        print(path, len(ev))


if __name__ == "__main__":
    main()
