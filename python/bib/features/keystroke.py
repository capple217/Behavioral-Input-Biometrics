"""Keystroke dynamics: raw events -> keystrokes -> timing features.

Pipeline (the producer recorded only integers + timestamps; all meaning is assigned here):
  1. events_to_keystrokes: pair each press with its release per key code.
     - OS autorepeat keyDowns (aux == 1) are dropped: they are not presses.
     - Modifiers arrive as KIND_FLAGS; press vs release is read from the
       device-dependent L/R bits of the flags word after the event.
     - Open presses are discarded at recorded drop gaps (the matching release
       may be among the lost events).
  2. timings: per keystroke i (sorted by press time)
        H  = release_i - press_i                 (hold / dwell)
        DD = press_i   - press_{i-1}             (down-down latency)
        UD = press_i   - release_{i-1}           (up-down / flight, negative = rollover)
     A new "segment" starts after a pause > pause_ms or a data gap; DD/UD
     across a segment boundary are meaningless and are masked.
  3. Representations for the models:
     - sequence windows (LSTM autoencoder): per step [key token, H, DD, UD, boundary]
     - summary vectors (dense AE / distance detectors): ~24 robust statistics
       of a window, grouped by hand / transition type so they exist for any text.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bib.ring import DOP_DOWN, DOP_UP, KIND_FLAGS, KIND_KEY

# macOS ANSI virtual key codes (Carbon kVK_*) -> canonical names
VK = {
    0: "a", 1: "s", 2: "d", 3: "f", 4: "h", 5: "g", 6: "z", 7: "x", 8: "c", 9: "v", 11: "b", 12: "q", 13: "w",
    14: "e", 15: "r", 16: "y", 17: "t", 18: "1", 19: "2", 20: "3", 21: "4", 22: "6", 23: "5", 24: "=", 25: "9",
    26: "7", 27: "-", 28: "8", 29: "0", 30: "]", 31: "o", 32: "u", 33: "[", 34: "i", 35: "p", 36: "return",
    37: "l", 38: "j", 39: "'", 40: "k", 41: ";", 42: "\\", 43: ",", 44: "/", 45: "n", 46: "m", 47: ".",
    48: "tab", 49: "space", 50: "`", 51: "backspace", 53: "escape", 54: "cmd", 55: "cmd", 56: "shift",
    57: "capslock", 58: "option", 59: "control", 60: "shift", 61: "option", 62: "control", 63: "fn",
    117: "fwd_delete", 123: "left", 124: "right", 125: "down", 126: "up",
}
# device-dependent modifier bits (IOKit NX_DEVICE*KEYMASK) by key code
MOD_BITS = {59: 0x1, 56: 0x2, 60: 0x4, 55: 0x8, 54: 0x10, 58: 0x20, 61: 0x40, 62: 0x2000, 63: 0x800000}

VOCAB = sorted(set(VK.values())) + ["other"]
TOKEN = {name: i for i, name in enumerate(VOCAB)}

LEFT = set("qwertasdfgzxcvb12345`") | {"tab", "capslock"}
RIGHT = set("yuiophjklnm67890-=[]\\;',./") | {"return", "backspace"}


def key_name(code: int) -> str:
    return VK.get(int(code), "other")


def hand(name: str) -> str:
    if name == "space":
        return "S"
    if name in LEFT:
        return "L"
    if name in RIGHT:
        return "R"
    return "M"  # modifiers / navigation / other


KS_DTYPE = np.dtype([("key", "<i8"), ("press", "<i8"), ("release", "<i8"), ("seg_break", "?")])


def events_to_keystrokes(ev: np.ndarray, t_ns: np.ndarray, gaps: np.ndarray | None = None) -> np.ndarray:
    """Raw events (+ resolved times) -> keystrokes sorted by press time."""
    gapset = set(int(g) for g in (gaps if gaps is not None else []))
    is_key = (ev["kind"] == KIND_KEY) | (ev["kind"] == KIND_FLAGS)
    idx = np.nonzero(is_key)[0]
    open_: dict[int, int] = {}
    out, breaks = [], set()
    last_gap_before = -1
    gap_sorted = sorted(gapset)
    gi = 0
    for i in idx:
        while gi < len(gap_sorted) and gap_sorted[gi] <= i:
            open_.clear()  # a lost release would otherwise pair with a much later one
            last_gap_before = gap_sorted[gi]
            gi += 1
        e = ev[i]
        code, t = int(e["key_code"]), int(t_ns[i])
        if e["kind"] == KIND_KEY:
            if e["dop"] == DOP_DOWN:
                if e["aux"] == 1:
                    continue
                open_[code] = t
            elif e["dop"] == DOP_UP and code in open_:
                out.append((code, open_.pop(code), t, last_gap_before))
        else:
            bit = MOD_BITS.get(code)
            if bit is None:
                continue
            down = bool(int(e["flags"]) & bit)
            if down and code not in open_:
                open_[code] = t
            elif not down and code in open_:
                out.append((code, open_.pop(code), t, last_gap_before))
    if not out:
        return np.empty(0, KS_DTYPE)
    arr = np.array(out, dtype=[("key", "<i8"), ("press", "<i8"), ("release", "<i8"), ("g", "<i8")])
    arr = arr[np.argsort(arr["press"], kind="stable")]
    ks = np.empty(len(arr), KS_DTYPE)
    ks["key"], ks["press"], ks["release"] = arr["key"], arr["press"], arr["release"]
    ks["seg_break"] = np.r_[True, arr["g"][1:] != arr["g"][:-1]]  # crossed a data gap
    return ks


@dataclass
class Timings:
    keys: np.ndarray       # canonical token ids
    names: list
    H: np.ndarray          # ms
    DD: np.ndarray         # ms (nan at segment starts)
    UD: np.ndarray         # ms (nan at segment starts)
    boundary: np.ndarray   # bool: first keystroke of a segment


def timings(ks: np.ndarray, pause_ms: float = 1500.0) -> Timings:
    names = [key_name(k) for k in ks["key"]]
    p, r = ks["press"].astype(np.float64) / 1e6, ks["release"].astype(np.float64) / 1e6
    H = r - p
    DD = np.r_[np.nan, np.diff(p)]
    UD = np.r_[np.nan, p[1:] - r[:-1]]
    boundary = ks["seg_break"].copy() | ~(DD <= pause_ms)  # nan compares False -> boundary
    DD[boundary], UD[boundary] = np.nan, np.nan
    keys = np.array([TOKEN.get(n, TOKEN["other"]) for n in names], dtype=np.int64)
    return Timings(keys, names, H, DD, UD, boundary)


def squash(ms: np.ndarray, scale: float = 50.0) -> np.ndarray:
    """asinh(ms/scale): ~linear below `scale` ms, logarithmic above, defined for negative UD."""
    return np.arcsinh(ms / scale)


def sequence_windows(tm: Timings, width: int = 50, stride: int = 25, data_breaks: bool = True):
    """Fixed-width windows for the sequence model.

    Returns tokens (n, W) int64 and x (n, W, 4) float32 with channels
    [squash(H), squash(DD), squash(UD), boundary]; DD/UD are 0 where masked.
    Windows never span a recorded data gap (pauses are allowed: they are
    marked by the boundary channel)."""
    n = len(tm.H)
    feats = np.stack([squash(tm.H), squash(np.nan_to_num(tm.DD)), squash(np.nan_to_num(tm.UD)),
                      tm.boundary.astype(np.float64)], axis=1).astype(np.float32)
    starts = []
    s = 0
    while s + width <= n:
        starts.append(s)
        s += stride
    toks = np.stack([tm.keys[s:s + width] for s in starts]) if starts else np.empty((0, width), np.int64)
    x = np.stack([feats[s:s + width] for s in starts]) if starts else np.empty((0, width, 4), np.float32)
    return toks, x, np.array(starts, dtype=np.int64)


SUMMARY_NAMES = (
    [f"H_{g}" for g in ["L", "R", "S", "shift", "backspace", "M"]]
    + [f"{m}_{t}" for t in ["LL", "LR", "RL", "RR", "xS", "Sx"] for m in ["DD", "UD"]]
    + ["DD_med", "DD_iqr", "H_med", "H_iqr", "backspace_rate", "rollover_rate"]
)


def summary_vectors(tm: Timings, width: int = 50, stride: int = 25) -> np.ndarray:
    """Per-window robust statistics (medians of squashed timings). NaN where a
    group is absent in the window; callers impute with training means."""
    names = np.array(tm.names)
    hands = np.array([hand(n) for n in tm.names])
    prev_hand = np.concatenate([np.array(["X"]), hands[:-1]])
    trans = np.char.add(prev_hand.astype("<U1"), hands.astype("<U1"))
    H, DD, UD = squash(tm.H), squash(tm.DD), squash(tm.UD)
    rows = []
    s = 0
    while s + width <= len(H):
        sl = slice(s, s + width)
        nm, hd, tr = names[sl], hands[sl], trans[sl]
        h, dd, ud = H[sl], DD[sl], UD[sl]

        def med(v, m):
            v = v[m & ~np.isnan(v)]
            return float(np.median(v)) if len(v) else np.nan

        row = [med(h, hd == "L"), med(h, hd == "R"), med(h, hd == "S"), med(h, nm == "shift"),
               med(h, nm == "backspace"), med(h, (hd == "M") & (nm != "shift"))]
        for t in ["LL", "LR", "RL", "RR"]:
            row += [med(dd, tr == t), med(ud, tr == t)]
        to_s = np.char.endswith(tr, "S") & ~np.char.startswith(tr, "S")
        from_s = np.char.startswith(tr, "S") & ~np.char.endswith(tr, "S")
        row += [med(dd, to_s), med(ud, to_s), med(dd, from_s), med(ud, from_s)]
        ddv = dd[~np.isnan(dd)]
        q = np.quantile(ddv, [0.25, 0.5, 0.75]) if len(ddv) else [np.nan] * 3
        hq = np.quantile(h, [0.25, 0.5, 0.75])
        udv = tm.UD[sl][~np.isnan(tm.UD[sl])]
        row += [q[1], q[2] - q[0], hq[1], hq[2] - hq[0], float(np.mean(nm == "backspace")),
                float(np.mean(udv < 0)) if len(udv) else np.nan]
        rows.append(row)
        s += stride
    return np.array(rows, dtype=np.float64).reshape(-1, len(SUMMARY_NAMES))
