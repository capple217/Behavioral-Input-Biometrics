"""Mouse dynamics: pointer samples -> actions -> per-action feature vectors.

Both sources (our InputEvents and the Balabit CSVs) are first converted to a
common sample stream (t seconds, x, y, etype, button) so the same segmentation
and features are used for the benchmark and for live data.

Actions (after Ahmed & Traore 2007; Antal & Egyed-Zsigmond 2019):
  MM  mouse move: a run of move samples ended by a pause > `pause_s` (2 s; chosen
      over 0.5 s and 10 s on Balabit *training* files only)
  PC  point-and-click: a move run ended by a button press+release
  DD  drag-and-drop: press, drag samples, release
Scroll samples end the current run and are otherwise ignored.

Features per action (scale-free where possible, heavy tails log-compressed):
  type one-hot, duration, #points, path length, straight distance,
  straightness, end-to-end direction (sin/cos), velocity / acceleration / jerk
  statistics, angular velocity, curvature, max deviation from the chord,
  fraction of high-curvature points, time-to-peak-velocity ratio,
  acceleration-time ratio, button hold time; v2 adds per-axis velocity
  stats, min velocity, initial-acceleration duration, total turning and sharp
  turns (from Antal & Egyed-Zsigmond 2019), which lowered training-only EER
  at k=10 actions from 0.31 to 0.28.
"""
from __future__ import annotations

import csv

import numpy as np

from bib.ring import DOP_DOWN, DOP_DRAG, DOP_NONE, DOP_UP, KIND_MOUSE, SIDE_SCROLL

MOVE, DRAG, DOWN, UP, SCROLL = 0, 1, 2, 3, 4
SAMPLE_DTYPE = np.dtype([("t", "<f8"), ("x", "<f8"), ("y", "<f8"), ("etype", "<i4")])

FEATURE_NAMES = [
    "is_MM", "is_PC", "is_DD", "log_duration", "log_npoints", "log_path", "log_chord", "straightness",
    "dir_sin", "dir_cos", "log_v_mean", "log_v_std", "log_v_max", "a_mean", "a_std", "log_a_absmax",
    "j_absmean", "w_absmean", "w_std", "curv_absmean", "curv_std", "log_maxdev_ratio", "crit_frac",
    "t_vmax_ratio", "accel_time_ratio", "log_hold",
    # v2 (Antal & Egyed-Zsigmond 2019 additions)
    "log_vx_absmean", "log_vy_absmean", "log_vx_std", "log_vy_std", "log_v_min", "log_a_beg_time",
    "sum_abs_angle", "sharp_angles",
]


def samples_from_events(ev: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    m = ev["kind"] == KIND_MOUSE
    e, t = ev[m], t_ns[m]
    s = np.empty(len(e), SAMPLE_DTYPE)
    s["t"] = t / 1e9  # absolute seconds on the session clock (only differences are used)
    s["x"], s["y"] = e["x"], e["y"]
    et = np.full(len(e), MOVE, np.int32)
    et[e["dop"] == DOP_DRAG] = DRAG
    et[e["dop"] == DOP_DOWN] = DOWN
    et[e["dop"] == DOP_UP] = UP
    et[(e["mouse_side"] == SIDE_SCROLL) & (e["dop"] == DOP_NONE)] = SCROLL
    s["etype"] = et
    return s


def samples_from_balabit(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for rec in r:
            try:
                t, button, state, x, y = float(rec[1]), rec[2], rec[3], float(rec[4]), float(rec[5])
            except (ValueError, IndexError):
                continue
            if button == "Scroll":
                et = SCROLL
            elif state == "Move":
                et = MOVE
            elif state == "Drag":
                et = DRAG
            elif state == "Pressed":
                et = DOWN
            elif state == "Released":
                et = UP
            else:
                continue
            rows.append((t, x, y, et))
    s = np.array(rows, dtype=SAMPLE_DTYPE)
    # Balabit logs off-screen positions as 65535-ish sentinels: drop them
    bad = (s["x"] >= 10000) | (s["y"] >= 10000)
    return s[~bad]


def segment_actions(s: np.ndarray, pause_s: float = 2.0, min_points: int = 4):
    """Split a sample stream into actions. Returns list of (type, points (n,3) [t,x,y], hold_s)."""
    actions = []
    run: list = []        # current move run
    drag: list | None = None
    press_t = None
    last_t = None

    def flush_move(kind="MM", hold=0.0):
        nonlocal run
        if len(run) >= min_points:
            actions.append((kind, np.array(run), hold))
        run = []

    for t, x, y, et in s:
        if last_t is not None and t - last_t > pause_s and drag is None:
            flush_move()
        last_t = t
        if et == MOVE:
            run.append((t, x, y))
        elif et == SCROLL:
            flush_move()
        elif et == DOWN:
            run.append((t, x, y))
            press_t = t
            drag = [(t, x, y)]
        elif et == DRAG:
            if drag is not None:
                drag.append((t, x, y))
        elif et == UP:
            hold = (t - press_t) if press_t is not None else 0.0
            if drag is not None and len(drag) >= min_points:
                drag.append((t, x, y))
                flush_move()  # the approach movement before the press becomes its own MM action
                actions.append(("DD", np.array(drag), hold))
            else:
                flush_move("PC", hold)
            drag, press_t = None, None
    flush_move()
    return actions


def _dedupe(p: np.ndarray) -> np.ndarray:
    keep = np.r_[True, np.diff(p[:, 0]) > 1e-6]
    return p[keep]


def action_features(kind: str, p: np.ndarray, hold: float) -> np.ndarray | None:
    p = _dedupe(p)
    if len(p) < 4:
        return None
    t, x, y = p[:, 0], p[:, 1], p[:, 2]
    dt = np.diff(t)
    dx, dy = np.diff(x), np.diff(y)
    ds = np.hypot(dx, dy)
    path = ds.sum()
    chord = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    dur = float(t[-1] - t[0])
    if dur <= 0 or path <= 0:
        return None
    v = ds / dt
    tm = (t[1:] + t[:-1]) / 2
    a = np.diff(v) / np.diff(tm) if len(v) > 1 else np.zeros(1)
    j = np.diff(a) / np.diff(tm[1:]) if len(a) > 1 else np.zeros(1)
    theta = np.unwrap(np.arctan2(dy, dx))
    dtheta = np.diff(theta)
    w = dtheta / np.diff(tm) if len(theta) > 1 else np.zeros(1)
    seg = ds[1:]
    curv = np.where(seg > 0, dtheta / np.maximum(seg, 1e-9), 0.0) if len(theta) > 1 else np.zeros(1)
    if chord > 0:
        nx, ny = (x[-1] - x[0]) / chord, (y[-1] - y[0]) / chord
        dev = np.abs((x - x[0]) * ny - (y - y[0]) * nx)
        maxdev = float(dev.max()) / chord
        ang = np.arctan2(y[-1] - y[0], x[-1] - x[0])
    else:
        maxdev, ang = 0.0, 0.0
    f = [
        kind == "MM", kind == "PC", kind == "DD", np.log(dur), np.log(len(p)), np.log1p(path), np.log1p(chord),
        min(chord / path, 1.0), np.sin(ang), np.cos(ang), np.log1p(v.mean()), np.log1p(v.std()), np.log1p(v.max()),
        np.arcsinh(a.mean() / 1e3), np.arcsinh(a.std() / 1e3), np.log1p(np.abs(a).max()),
        np.arcsinh(np.abs(j).mean() / 1e5), np.log1p(np.abs(w).mean()), np.log1p(w.std()),
        np.log1p(np.abs(curv).mean()), np.log1p(curv.std()), np.log1p(maxdev), float(np.mean(np.abs(curv) > 0.1)),
        float((tm[np.argmax(v)] - t[0]) / dur), float(np.sum(dt[1:][a > 0]) / dur) if len(a) == len(dt) - 1 else 0.0,
        np.log1p(hold * 1e3),
    ]
    vx, vy = dx / dt, dy / dt
    a_pos = a > 0
    beg = int(np.argmin(a_pos)) if len(a_pos) and not a_pos.all() else len(a_pos)  # length of initial acceleration run
    a_beg = float(tm[beg] - tm[0]) if beg < len(tm) else dur
    f += [np.log1p(np.abs(vx).mean()), np.log1p(np.abs(vy).mean()), np.log1p(vx.std()), np.log1p(vy.std()),
          np.log1p(v.min()), np.log1p(a_beg * 1e3), float(np.abs(dtheta).sum()) if len(theta) > 1 else 0.0,
          float(np.sum(np.abs(dtheta) > 0.5)) if len(theta) > 1 else 0.0]
    return np.array(f, dtype=np.float64)


def actions_to_features(actions, return_end_times: bool = False):
    rows, ends = [], []
    for k, p, h in actions:
        r = action_features(k, p, h)
        if r is not None and np.all(np.isfinite(r)):
            rows.append(r)
            ends.append(float(p[-1, 0]))
    F = np.array(rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
    return (F, np.array(ends)) if return_end_times else F
