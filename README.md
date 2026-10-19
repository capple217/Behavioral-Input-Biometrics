# Behavioral Input Biometrics

Continuous authentication on macOS from *how* you type and move the mouse.
A C++ producer taps raw HID input with `CGEventTapCreate`, hands fixed-layout
events through a hand-written lock-free SPSC queue to a bridge thread, which
publishes them into an SPSC ring in POSIX shared memory. A Python consumer
reads the ring with acquire/release semantics, turns raw key codes and pointer
samples into timing / kinematic features, scores them with one-class
autoencoders, and turns scores into **conformal p-values** whose
false-reject rate is controlled, adapted online with **ACI**.

Everything below the "Results" heading is measured on public data with the
scripts in `experiments/`; JSON outputs are in `results/`.

## Results

### Keystroke: CMU benchmark (51 subjects, Killourhy & Maxion DSN 2009 protocol)

Harness check first: three of their detectors, re-implemented, must reproduce
the published numbers before any of our models are scored.

| Detector | EER, ours | EER, published |
|---|---|---|
| Euclidean | 0.171 (0.095) | 0.171 (0.095) |
| Manhattan (scaled) | 0.096 (0.069) | 0.096 (0.069) |
| Nearest neighbor (Mahalanobis) | 0.100 (0.064) | 0.100 (0.064) |
| Neural net (auto-assoc), K&M's autoencoder | — | 0.161 (0.080) |
| **Autoencoder (ours)** | **0.096 (0.057)** | |
| LSTM autoencoder (ours) | 0.111 (0.071) | |

Mean (sd) EER over subjects. Our models' hyperparameters were chosen on the 75
CMU lab subjects that are **not** in the benchmark (32-config and 8-config
grids) and frozen before the benchmark run. The autoencoder ties the best
published detector (lower on 19/51 subjects, Wilcoxon p = 0.37, not
significant; lower spread and lower worst case, 0.255 vs 0.325) and is 40%
below the published autoencoder. It does not beat Manhattan (scaled).
What made the difference vs. the published AE: scoring with L1 instead of
squared reconstruction error (0.077 vs 0.083 on the tuning set) and raw rather
than log timings (0.08 vs 0.13).

### Does a conformal false-reject guarantee survive time?

Split conformal promises FRR ≤ α if tomorrow's typing is exchangeable with the
calibration data. `experiments/keystroke_drift.py`: enroll on sessions 1–2,
test on every later session, 126 lab subjects (8 sessions) and 105 *different*
field subjects typing on their own machines (10–15 sessions). Manhattan
(scaled) detector shown; the autoencoder behaves the same way.

| Threshold policy (α = 0.10) | Lab FRR | Lab FAR | subjects with FRR > 2α | Field FRR | Field FAR | subjects > 2α |
|---|---|---|---|---|---|---|
| static split conformal | 0.103 | 0.262 | 15% | 0.103 | 0.262 | 14% |
| calibrate on a later session | 0.274 | 0.148 | 47% | 0.256 | 0.149 | 46% |
| login re-calibration (10 reps/session) | 0.169 | 0.157 | 33% | 0.164 | 0.154 | 28% |
| **ACI, γ = 0.02** | **0.092** | **0.164** | **0%** | **0.094** | **0.165** | **0%** |

* The static guarantee holds *on average* but decays: lab FRR by test session
  goes 0.057 → 0.083 → 0.081 → 0.099 → 0.142 → 0.155, and one subject in seven
  ends above 2α.
* Calibrating on a later session (my initial design) is worse: the detector is
  trained on less, earlier data and drifts further than one session of
  calibration absorbs.
* ACI (Gibbs & Candès 2021) holds every subject under 2α **and** cuts FAR by
  37% (0.262 → 0.164), because practice makes typing more consistent and ACI
  tightens the threshold when FRR runs under target.
* ACI only needs the outcome of explicit re-authentication: unchallenged
  decisions count as "no error" (stricter), confirmed challenges as an error
  (looser). An impostor who slips through unchallenged therefore pushes the
  threshold in the safe direction. This is what `bib.live` runs.

### Mouse: Balabit Mouse Dynamics Challenge (10 users, 816 labeled test sessions)

One-class: each user's model sees only that user's training sessions.
Segmentation and feature set chosen on training files only.

| Detector | 1 action | 10 actions | 20 actions | whole session (median 62 actions) |
|---|---|---|---|---|
| Manhattan (scaled) | 0.389 | 0.357 | 0.324 | 0.328 |
| **Autoencoder (ours)** | 0.335 | 0.249 | **0.229** | 0.222 |
| Antal & Egyed-Zsigmond 2019, **two-class** (trained on other users' data) | — | — | 0.188 | AUC 0.92 |

Mouse is the weak modality. The comparison row uses impostor-class training
data that a one-class model never sees, so it is a reference, not a like-for-like.

### Fusion (chimeric users)

No public dataset has the same people's keystrokes and mouse data, so each
of 200 random pairings maps CMU subjects onto Balabit users (standard
"chimeric" evaluation; assumes the modalities are independent given identity).
One attempt = one password entry + the first 10 mouse actions.

| | EER | FRR @ α=0.05 | FAR @ α=0.05 |
|---|---|---|---|
| keystroke only | 0.116 | 0.071 | 0.272 |
| mouse only | 0.264 | 0.062 | 0.572 |
| Fisher fusion of p-values | **0.110** | 0.062 | 0.341 |
| 2×mean p (valid under any dependence) | 0.176 | 0.002 | 0.733 |

Fusion helps EER a little. At a fixed α it trades FAR for FRR, because a weak
modality dilutes the combined evidence.

### Systems

* `tests/spsc_queue_test.cpp`: single-thread edge cases plus a 2-thread
  transfer of 5M 80-byte events checking order and field integrity. Clean under
  ThreadSanitizer. Throughput on a 2-vCPU x86 VM: 14.5–15.9 M events/s vs
  12.6–13.2 M for `boost::lockfree::spsc_queue` (same harness). Re-run on an
  M-series Mac for real numbers.
* `tests/test_ring_e2e.py`: `fake_producer` pushes sequence-numbered events
  through the same queue → bridge → shared ring; Python checks every event is
  untorn and in order, and that `received + dropped == sent` under
  backpressure. Passes on x86-64 and on aarch64 (weakly ordered).

## How it works

```
 HID ──► CGEventTap callback (main run loop)          C++ process: input_sensor
            │ raw ints + 2 timestamps, nothing else
            ▼
         SpscQueue<InputEvent, 8192>   hand-written, cache-line padded,
            │                          cached indices, newest-drop
            ▼
         Bridge thread ──► SharedRing (POSIX shm, 4096 slots, newest-drop,
                           drop counters, layout pinned by static_assert)
                             │  write_idx: release / acquire
                             │  read_idx:  release / acquire
                             ▼
         libbibring (C ABI, LDAR/STLR) ◄── ctypes ──  Python process
            │
            ▼
   events → keystrokes (press/release pairing, autorepeat + modifiers + gaps)
          → H / DD / UD timings → 50-key windows → LSTM autoencoder
          → pointer samples → MM / PC / DD actions → 34 kinematic features → autoencoder
          → conformal p-values → Fisher fusion → ACI threshold → ok / CHALLENGE
```

Design decisions worth knowing:

* **Producer-lean.** The callback stores integers and two clocks:
  `steady_clock` at callback time, and `CGEventGetTimestamp` (when the HID
  event happened). The consumer fits the event clock's unit per session
  (nanoseconds on Intel; it has been reported as 24 MHz mach ticks on Apple
  Silicon) and uses it for timing features, which removes callback-delivery
  jitter. `bib.enroll` prints the measured callback latency.
* **Fixed binary layout.** `InputEvent` is 80 bytes of fixed-width fields with
  explicit padding. Every offset is `static_assert`ed in C++ and re-checked by
  Python against the shim at attach time. The old struct embedded a
  `std::chrono::time_point`, which is not an ABI.
* **Newest-drop, observable loss.** A full ring refuses the new event rather
  than overwriting one Python may be reading. Drops are counted at both stages,
  and the recorder marks gaps so no timing feature ever spans lost events.
* **Why a C shim for Python.** Plain ctypes loads/stores have no ordering. On
  arm64, Python could read a slot older than the index it just saw. The shim
  exports exactly `load_acquire(write_idx)`, `store_release(read_idx)` and the
  mapping. It also avoids `multiprocessing.shared_memory`, whose
  resource_tracker (Python < 3.13) unlinks the producer's segment when the
  consumer exits.
* **macOS details handled.** Input Monitoring permission check
  (`CGPreflightListenEventAccess`). Re-enable on `kCGEventTapDisabledByTimeout`.
  Shift/Cmd/Opt/Ctrl via `kCGEventFlagsChanged`, using device-dependent L/R
  bits. Autorepeat tagged and dropped. Scroll captured. Stale segments
  unlinked (macOS allows one `ftruncate` per shm object). Segment mode 0600
  (it carries every keystroke). Ctrl-C via a dispatch signal source, because
  `CFRunLoopStop` isn't async-signal-safe.

## Use it (macOS)

```fish
brew install cmake          # Boost no longer needed (optional, only for the queue benchmark)
cmake -B build && cmake --build build -j
python3 -m venv .venv; source .venv/bin/activate.fish
pip install -e '.[dev]'

./build/input_sensor        # first run: grant Input Monitoring, then re-run
# in another terminal, record a few sessions on different days (each = one file)
python -m bib.record --out data/sessions --minutes 30
python -m bib.enroll --sessions data/sessions --out data/model.pt
python -m bib.live --model data/model.pt --alpha 0.05      # type 'me' + Enter to confirm a challenge
```

`bib.enroll` reports the false-reject rate on your most recent session using a
model enrolled on the earlier ones. That is all your own data can measure. A
false-accept rate needs other people: have a few friends use the machine while
`bib.record --tag impostor` runs.

Privacy: `data/` holds keystroke logs (key codes + timings) and is
git-ignored. Password fields with Secure Event Input are invisible to every
event tap.

## Reproduce

```fish
./scripts/fetch_data.sh                 # CMU + Balabit from GitHub (~300 MB)
python experiments/keystroke_cmu.py     # ~20 min on 2 cores (tuning grids dominate)
python experiments/keystroke_drift.py   # ~4 min
python experiments/mouse_balabit.py     # ~4 min
python experiments/fusion_chimeric.py   # ~2 min
pytest                                   # needs build/ for the ring test
./build/spsc_queue_test 5000000
```

## Limitations

* The keystroke benchmark is fixed-text (a password). The live system scores
  free text with the LSTM autoencoder, which has no public impostor
  evaluation here. Its FAR on free text is unmeasured.
* Mouse fusion results rely on the chimeric-independence assumption.
* Single-sample keystroke FAR at α = 0.05 is ~0.27–0.45. One password entry is
  weak evidence on its own; the continuous setting accumulates many windows.
* The input_sensor target compiles only on macOS. It was written against the
  documented API, but CI here is Linux (`fake_producer` covers the shared code
  path).

## Layout

```
include/bib/   input_event.hpp  spsc_queue.hpp  shared_ring.hpp  bridge.hpp
src/           input_sensor.cpp (macOS)  bibring.cpp (Python shim)  fake_producer.cpp
python/bib/    ring  record  clock  enroll  live  synth  conformal  metrics  datasets
               features/{keystroke,mouse}  models/{detectors,seq_ae}
experiments/   keystroke_cmu  keystroke_drift  mouse_balabit  fusion_chimeric
tests/         spsc_queue_test.cpp  test_ring_e2e.py  test_features.py
results/       *.json (+ logs) from the experiments above
```

References: Killourhy & Maxion, *Comparing Anomaly-Detection Algorithms for
Keystroke Dynamics*, DSN 2009 · Gibbs & Candès, *Adaptive Conformal Inference
Under Distribution Shift*, NeurIPS 2021 · Antal & Egyed-Zsigmond, *Intrusion
Detection Using Mouse Dynamics*, IET Biometrics 2019 · Vovk & Wang, *Combining
p-values via averaging*, Biometrika 2020.
