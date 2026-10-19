"""Consumer side of the shared-memory ring (C++ producer -> Python).

The protocol is in include/bib/shared_ring.hpp. This module keeps the logic in
Python (what to read, when to hand slots back) and calls into libbibring only
for the three things Python cannot do itself: map the segment without
resource_tracker interference, load write_idx with ACQUIRE semantics, and
store read_idx with RELEASE semantics.

    w = load_acquire(write_idx)      # slots [r, w) are complete and visible
    batch = slots[r..w) (copied)     # numpy fancy-index copy, handles wrap
    store_release(read_idx, w)       # copy finished before producer reuses slots
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RING_NAME = b"/BIBSharedMem"

# Mirrors bib::InputEvent. Offsets are explicit and verified against the shim.
EVENT_DTYPE = np.dtype(
    {
        "names": ["kind", "dop", "key_code", "flags", "mouse_side", "aux", "x", "y",
                  "delta_x", "delta_y", "timestamp", "event_time"],
        "formats": ["<i4", "<i4", "<i8", "<u8", "<i4", "<i4", "<f8", "<f8", "<i8", "<i8", "<i8", "<u8"],
        "offsets": [0, 4, 8, 16, 24, 28, 32, 40, 48, 56, 64, 72],
        "itemsize": 80,
    }
)

KIND_KEY, KIND_MOUSE, KIND_FLAGS = 0, 1, 2
DOP_DOWN, DOP_UP, DOP_DRAG, DOP_NONE = 0, 1, 2, 3
SIDE_NONE, SIDE_LEFT, SIDE_RIGHT, SIDE_SCROLL = 0, 1, 2, 3


def _find_lib() -> Path:
    env = os.environ.get("BIB_RING_LIB")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[2]
    names = ["libbibring.dylib", "libbibring.so"]
    dirs = [Path(os.environ["BIB_BUILD"])] if "BIB_BUILD" in os.environ else []
    for d in dirs + [root / "build", root]:
        for n in names:
            if (d / n).exists():
                return d / n
    raise FileNotFoundError("libbibring not found: build with cmake (see README) or set BIB_RING_LIB")


def load_lib(path: Path | None = None) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path or _find_lib()))
    vp, u64 = ctypes.c_void_p, ctypes.c_uint64
    lib.bib_ring_layout.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
    lib.bib_ring_layout.restype = ctypes.c_int
    lib.bib_ring_attach.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.bib_ring_attach.restype = vp
    lib.bib_ring_detach.argtypes = [vp]
    lib.bib_ring_detach.restype = None
    lib.bib_ring_load_write_idx.argtypes = [vp]
    lib.bib_ring_load_write_idx.restype = u64
    lib.bib_ring_load_read_idx.argtypes = [vp]
    lib.bib_ring_load_read_idx.restype = u64
    lib.bib_ring_store_read_idx.argtypes = [vp, u64]
    lib.bib_ring_store_read_idx.restype = None
    lib.bib_ring_counters.argtypes = [vp, ctypes.POINTER(u64), ctypes.POINTER(u64), ctypes.POINTER(ctypes.c_uint32)]
    lib.bib_ring_counters.restype = None
    lib.bib_ring_slots.argtypes = [vp]
    lib.bib_ring_slots.restype = vp
    return lib


def check_layout(lib: ctypes.CDLL) -> int:
    """Fail loudly if the Python dtype and the compiled struct disagree. Returns ring capacity."""
    buf = (ctypes.c_int64 * 32)()
    n = lib.bib_ring_layout(buf, 32)
    v = list(buf[:n])
    size, offs, cap = v[0], v[1:13], v[13]
    want = [EVENT_DTYPE.fields[name][1] for name in EVENT_DTYPE.names]
    if size != EVENT_DTYPE.itemsize or offs != want:
        raise RuntimeError(f"InputEvent layout mismatch: C++ size={size} offsets={offs}, "
                           f"Python size={EVENT_DTYPE.itemsize} offsets={want}")
    return cap


@dataclass
class Counters:
    dropped_ring: int
    dropped_tap: int
    producer_alive: bool


class RingReader:
    def __init__(self, name: bytes = RING_NAME, lib: ctypes.CDLL | None = None, wait_s: float = 0.0):
        self.lib = lib or load_lib()
        self.capacity = check_layout(self.lib)
        self.mask = self.capacity - 1
        self.name = name
        self._ring = None
        deadline = time.monotonic() + wait_s
        while True:
            err = ctypes.c_int(0)
            p = self.lib.bib_ring_attach(name, ctypes.byref(err))
            if p:
                break
            if time.monotonic() >= deadline:
                why = "bad magic/version (producer from a different build?)" if err.value == -1 else os.strerror(err.value)
                raise ConnectionError(f"cannot attach to {name.decode()}: {why}. Is input_sensor running?")
            time.sleep(0.05)
        self._ring = p
        raw = (ctypes.c_uint8 * (self.capacity * EVENT_DTYPE.itemsize)).from_address(self.lib.bib_ring_slots(p))
        self._slots = np.frombuffer(raw, dtype=EVENT_DTYPE)  # zero-copy view of shared memory
        self._r = int(self.lib.bib_ring_load_read_idx(p))

    def drain(self, max_events: int | None = None) -> np.ndarray:
        """Copy out every published, unread event (oldest first) and release the slots."""
        w = int(self.lib.bib_ring_load_write_idx(self._ring))  # ACQUIRE
        n = w - self._r
        if n <= 0:
            return np.empty(0, dtype=EVENT_DTYPE)
        if max_events is not None:
            n = min(n, max_events)
        idx = (np.arange(self._r, self._r + n, dtype=np.uint64) & np.uint64(self.mask)).astype(np.intp)
        batch = self._slots[idx]  # fancy indexing => a private copy, not a view
        self._r += n
        self.lib.bib_ring_store_read_idx(self._ring, self._r)  # RELEASE
        return batch

    def counters(self) -> Counters:
        d, dt, a = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint32()
        self.lib.bib_ring_counters(self._ring, ctypes.byref(d), ctypes.byref(dt), ctypes.byref(a))
        return Counters(int(d.value), int(dt.value), bool(a.value))

    def close(self):
        if self._ring:
            self._slots = None
            self.lib.bib_ring_detach(self._ring)
            self._ring = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def stream(reader: RingReader, poll_s: float = 0.005, stop_when_producer_exits: bool = True):
    """Yield non-empty batches until the producer shuts down cleanly (or forever)."""
    while True:
        b = reader.drain()
        if len(b):
            yield b
            continue
        if stop_when_producer_exits and not reader.counters().producer_alive:
            b = reader.drain()
            if len(b):
                yield b
            return
        time.sleep(poll_s)


if __name__ == "__main__":  # quick tail: python -m bib.ring
    with RingReader(wait_s=10) as rr:
        for batch in stream(rr):
            for e in batch:
                print(e)
        print(rr.counters(), file=sys.stderr)
