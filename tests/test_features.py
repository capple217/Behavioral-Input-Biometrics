"""Unit tests for the consumer-side interpretation of raw events."""
import numpy as np

from bib.clock import resolve_times
from bib.conformal import aci, fuse, pvalues
from bib.features import keystroke as K
from bib.features import mouse as M
from bib.metrics import eer
from bib.ring import DOP_DOWN, DOP_NONE, DOP_UP, EVENT_DTYPE, KIND_FLAGS, KIND_KEY
from bib.synth import person, session


def _ev(rows):
    out = np.zeros(len(rows), EVENT_DTYPE)
    for i, (kind, dop, code, flags, aux, t) in enumerate(rows):
        out[i]["kind"], out[i]["dop"], out[i]["key_code"] = kind, dop, code
        out[i]["flags"], out[i]["aux"], out[i]["timestamp"] = flags, aux, t
    return out


MS = 1_000_000


def test_pairing_autorepeat_and_modifiers():
    ev = _ev([
        (KIND_FLAGS, DOP_NONE, 56, K.MOD_BITS[56], 0, 0),          # L-shift down
        (KIND_KEY, DOP_DOWN, 0, 0, 0, 50 * MS),                    # a down
        (KIND_KEY, DOP_DOWN, 0, 0, 1, 400 * MS),                   # a autorepeat: ignored
        (KIND_KEY, DOP_UP, 0, 0, 0, 600 * MS),                     # a up
        (KIND_FLAGS, DOP_NONE, 56, 0, 0, 700 * MS),                # L-shift up
        (KIND_KEY, DOP_DOWN, 1, 0, 0, 800 * MS),                   # s down
        (KIND_KEY, DOP_UP, 1, 0, 0, 900 * MS),
    ])
    ks = K.events_to_keystrokes(ev, ev["timestamp"])
    assert [K.key_name(k) for k in ks["key"]] == ["shift", "a", "s"]
    tm = K.timings(ks)
    np.testing.assert_allclose(tm.H, [700, 550, 100])
    assert np.isnan(tm.DD[0]) and tm.DD[1] == 50 and tm.UD[2] == 200  # s pressed 200 ms after a released
    assert tm.UD[1] == 50 - 700  # rollover: a pressed while shift still down


def test_gap_discards_open_press():
    ev = _ev([(KIND_KEY, DOP_DOWN, 0, 0, 0, 0), (KIND_KEY, DOP_UP, 0, 0, 0, 5000 * MS),
              (KIND_KEY, DOP_DOWN, 1, 0, 0, 6000 * MS), (KIND_KEY, DOP_UP, 1, 0, 0, 6100 * MS)])
    ks = K.events_to_keystrokes(ev, ev["timestamp"], gaps=np.array([1]))  # loss right before event 1
    assert [K.key_name(k) for k in ks["key"]] == ["s"]


def test_pause_starts_segment():
    ev = _ev([(KIND_KEY, DOP_DOWN, 0, 0, 0, 0), (KIND_KEY, DOP_UP, 0, 0, 0, 80 * MS),
              (KIND_KEY, DOP_DOWN, 1, 0, 0, 3000 * MS), (KIND_KEY, DOP_UP, 1, 0, 0, 3080 * MS)])
    tm = K.timings(K.events_to_keystrokes(ev, ev["timestamp"]), pause_ms=1500)
    assert tm.boundary.tolist() == [True, True] and np.isnan(tm.DD[1])


def test_synthetic_session_end_to_end_features():
    ev = session(person(1), seed=1, minutes=2)
    t, fit = resolve_times(ev)
    assert fit.unit_guess == "ns"
    tm = K.timings(K.events_to_keystrokes(ev, t))
    assert len(tm.H) > 100 and np.all(tm.H > 0)
    tok, x, _ = K.sequence_windows(tm, 50, 25)
    assert tok.shape[1:] == (50,) and x.shape[1:] == (50, 4)
    S = K.summary_vectors(tm, 50, 25)
    assert S.shape == (len(tok), len(K.SUMMARY_NAMES))
    F = M.actions_to_features(M.segment_actions(M.samples_from_events(ev, t)))
    assert F.shape[1] == len(M.FEATURE_NAMES) and len(F) > 10 and np.all(np.isfinite(F))


def test_clock_detects_mach_ticks():
    ev = session(person(2), seed=3, minutes=1)
    ev["event_time"] = (ev["timestamp"] / (125 / 3)).astype(np.uint64)  # Apple-Silicon-style ticks
    ev["timestamp"] += np.random.default_rng(0).integers(0, 300_000, len(ev))  # callback jitter <= 0.3 ms
    t, fit = resolve_times(ev)
    assert fit.unit_guess == "mach_24MHz"
    assert 0 < fit.latency_median_us < 300


def test_conformal_and_metrics():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=500), rng.normal(size=200_000)
    assert abs(np.mean(pvalues(cal, test) <= 0.1) - 0.1) < 0.02
    rej, _ = aci(cal, rng.normal(1.0, 1.5, 50_000), 0.05, 0.01)
    assert abs(rej.mean() - 0.05) < 0.01
    assert abs(eer(rng.normal(0, 1, 50_000), rng.normal(2, 1, 50_000)) - 0.1587) < 0.01
    assert fuse([np.array([1.0]), np.array([1.0])])[0] > 0.99
