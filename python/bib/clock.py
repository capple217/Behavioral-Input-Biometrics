"""Choosing the timestamp that feeds the timing features.

Each event carries two clocks:
  timestamp   steady_clock ns sampled inside the tap callback. Includes the
              delivery latency of the run loop (scheduling jitter, other taps).
  event_time  CGEventGetTimestamp(): when the HID event was generated. Better
              for dwell/flight times, but its unit is not pinned down in
              practice: nanoseconds on Intel Macs, while on Apple Silicon it has
              been observed to be raw mach_absolute_time ticks (24 MHz =>
              41.67 ns/tick). Rather than trusting either claim, we fit the
              ratio against the steady clock on each session.

resolve_times() returns event-clock nanoseconds mapped onto the steady-clock
axis, plus a diagnostic of the callback latency (steady - event), which is the
noise you'd be adding to every timing feature by using `timestamp` instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KNOWN_NS_PER_UNIT = {"ns": 1.0, "mach_24MHz": 125.0 / 3.0}


@dataclass
class ClockFit:
    source: str            # "event_time" or "timestamp"
    ns_per_unit: float
    unit_guess: str
    latency_median_us: float
    latency_p99_us: float


def resolve_times(ev: np.ndarray) -> tuple[np.ndarray, ClockFit]:
    ts = ev["timestamp"].astype(np.int64)
    et = ev["event_time"].astype(np.float64)
    if len(ev) < 20 or np.all(et == 0):
        return ts.copy(), ClockFit("timestamp", 1.0, "n/a", float("nan"), float("nan"))
    span_t = float(ts[-1] - ts[0])
    span_e = float(et[-1] - et[0])
    if span_t < 1e9 or span_e <= 0:
        return ts.copy(), ClockFit("timestamp", 1.0, "n/a", float("nan"), float("nan"))
    # robust slope: median over many well-separated pairs
    k = max(1, len(ev) // 50)
    i = np.arange(0, len(ev) - k)
    dt, de = (ts[i + k] - ts[i]).astype(np.float64), et[i + k] - et[i]
    ok = (de > 0) & (dt > 5e6)
    slope = float(np.median(dt[ok] / de[ok])) if ok.sum() > 10 else span_t / span_e
    unit, ns_per = "fitted", slope
    for name, v in KNOWN_NS_PER_UNIT.items():
        if abs(slope / v - 1) < 0.02:
            unit, ns_per = name, v
            break
    t_ev = (et - et[0]) * ns_per
    lat = ts - (t_ev + ts[0])
    # anchor: the smallest observed delay is the closest thing to zero latency
    lat = lat - np.quantile(lat, 0.001)
    t_ev_aligned = (t_ev + ts[0] + np.quantile(ts - (t_ev + ts[0]), 0.001)).astype(np.int64)
    fit = ClockFit("event_time", ns_per, unit, float(np.median(lat) / 1e3), float(np.quantile(lat, 0.99) / 1e3))
    return t_ev_aligned, fit
