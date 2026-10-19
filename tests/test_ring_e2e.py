"""End-to-end test of the C++ -> shared memory -> Python path (Linux or macOS).

Runs build/fake_producer, which pushes events whose key_code is a sequence
number through the same SpscQueue + Bridge + SharedRing as input_sensor.
Checks: every received event is intact (no torn fields), arrives in order, and
received + dropped == sent, i.e. every loss is accounted for by the counters.
"""
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from bib.ring import RingReader, stream

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = Path(os.environ.get("BIB_BUILD", ROOT / "build")) / "fake_producer"


def _run(count, rate, reader_sleep):
    proc = subprocess.Popen([str(PRODUCER), "--count", str(count), "--rate", str(rate),
                             "--startup", "0.5", "--linger", "1.0"], stderr=subprocess.PIPE, text=True)
    got = []
    with RingReader(wait_s=5) as rr:
        for batch in stream(rr, poll_s=0.001):
            got.append(batch)
            if reader_sleep:
                time.sleep(reader_sleep)
        c = rr.counters()
    proc.wait(timeout=60)
    ev = np.concatenate(got) if got else np.empty(0)
    return ev, c, proc.stderr.read()


def _check_intact(ev):
    i = ev["key_code"].astype(np.uint64)
    assert np.all(ev["flags"] == ~i), "torn or corrupted flags field"
    assert np.all(ev["x"] == ev["key_code"] * 0.5)
    assert np.all(ev["delta_y"] == -ev["key_code"] * 3)
    assert np.all(np.diff(ev["key_code"]) > 0), "out of order or duplicated"


@pytest.mark.skipif(not PRODUCER.exists(), reason="build fake_producer first")
def test_fast_reader_no_loss():
    n = 300_000
    ev, c, log = _run(n, rate=200_000, reader_sleep=0)
    _check_intact(ev)
    assert len(ev) + c.dropped_ring + c.dropped_tap == n, log
    assert c.dropped_ring + c.dropped_tap < n * 0.01, log


@pytest.mark.skipif(not PRODUCER.exists(), reason="build fake_producer first")
def test_slow_reader_drops_are_accounted():
    n = 200_000
    ev, c, log = _run(n, rate=0, reader_sleep=0.02)  # producer flat out, reader dawdles
    _check_intact(ev)
    missing = n - len(ev)
    assert missing > 0, "expected backpressure drops in this configuration"
    assert missing == c.dropped_ring + c.dropped_tap, (missing, c, log)
    # newest-drop: what we received is a prefix-heavy subset, never overwritten data
    assert ev["key_code"][0] == 0
