"""Record raw input sessions from the ring to disk (enrollment data).

    python -m bib.record --out data/sessions --minutes 30

Writes data/sessions/<host>_<YYYYmmdd-HHMMSS>.npz containing
    events  structured array (bib.ring.EVENT_DTYPE), oldest first
    gaps    indices i such that events were lost right before events[i]
            (from the drop counters), so features never bridge a gap
    meta    json: start/end wall time, counts, drops, clock fit

Privacy: this is a keylogger for your own account. Files hold every key you
type during the session (key codes + timings, not characters, but key codes
map 1:1 to characters). Keep data/ out of git (it is in .gitignore). macOS
Secure Event Input (password fields, some terminals with "Secure Keyboard
Entry") hides keystrokes from all event taps, so passwords typed into secure
fields are not captured.
"""
from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from bib.clock import resolve_times
from bib.ring import EVENT_DTYPE, KIND_FLAGS, KIND_KEY, KIND_MOUSE, RingReader


def save_session(path: Path, events: np.ndarray, gaps: list[int], meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, events=events, gaps=np.asarray(gaps, dtype=np.int64), meta=np.array(json.dumps(meta)))


def load_session(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(path, allow_pickle=False)
    return z["events"], z["gaps"], json.loads(str(z["meta"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sessions")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--tag", default="", help="free-form label stored in meta (e.g. 'coding', 'email')")
    ap.add_argument("--export-bin", action="store_true", help="also write raw 80-byte records for fake_producer --replay")
    args = ap.parse_args(argv)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    rr = RingReader(wait_s=30)
    chunks, gaps, n = [], [], 0
    last_drop = 0
    c0 = rr.counters()
    last_drop = c0.dropped_ring + c0.dropped_tap
    t_start, wall_start = time.monotonic(), datetime.now()
    last_print = t_start
    print(f"recording... (Ctrl-C to stop){' for %.1f min' % args.minutes if args.minutes else ''}", file=sys.stderr)
    while not stop["flag"]:
        b = rr.drain()
        c = rr.counters()
        d = c.dropped_ring + c.dropped_tap
        if d != last_drop:
            gaps.append(n)  # loss happened somewhere before the next event we hold
            last_drop = d
        if len(b):
            chunks.append(b)
            n += len(b)
        elif not c.producer_alive:
            print("producer exited", file=sys.stderr)
            break
        now = time.monotonic()
        if now - last_print > 10:
            allk = np.concatenate(chunks) if chunks else np.empty(0, EVENT_DTYPE)
            nk = int(np.sum(allk["kind"] == KIND_KEY) // 2) if len(allk) else 0
            print(f"  {now - t_start:6.0f}s  events={n}  ~keystrokes={nk}  dropped={d - (c0.dropped_ring + c0.dropped_tap)}",
                  file=sys.stderr)
            last_print = now
        if args.minutes and now - t_start > args.minutes * 60:
            break
        time.sleep(0.01)
    ev = np.concatenate(chunks) if chunks else np.empty(0, EVENT_DTYPE)
    c = rr.counters()
    rr.close()
    _, fit = resolve_times(ev)
    meta = {
        "host": platform.node(), "start": wall_start.isoformat(timespec="seconds"),
        "end": datetime.now().isoformat(timespec="seconds"), "events": int(len(ev)),
        "keys": int(np.sum(ev["kind"] == KIND_KEY)), "flags": int(np.sum(ev["kind"] == KIND_FLAGS)),
        "mouse": int(np.sum(ev["kind"] == KIND_MOUSE)),
        "dropped_total": int(c.dropped_ring + c.dropped_tap - (c0.dropped_ring + c0.dropped_tap)),
        "gaps": len(gaps), "tag": args.tag, "clock": fit.__dict__,
    }
    stem = f"{platform.node().split('.')[0]}_{wall_start.strftime('%Y%m%d-%H%M%S')}"
    out = Path(args.out) / f"{stem}.npz"
    save_session(out, ev, gaps, meta)
    if args.export_bin:
        ev.tofile(out.with_suffix(".bin"))
    print(f"saved {out}: {json.dumps(meta, indent=1)}", file=sys.stderr)


if __name__ == "__main__":
    main()
