"""Does a conformally calibrated false-reject rate survive time?

Split conformal promises FRR <= alpha only if tomorrow's genuine typing is
exchangeable with today's calibration samples. This experiment measures how
far that promise breaks across sessions (days) and what restores it, on two
populations:
  lab    126 subjects, 8 sessions, CMU lab machine
  field  105 different subjects, 10-15 sessions on their own computers

Enrollment = sessions 1-2. Test = sessions 3+ (reps 11-50 of each session, so
every variant is scored on identical samples). Impostors = first 5 reps of
every other subject in the same population.

Variants (threshold policy; the detector is the same within a row):
  static-random   train on a random 75 enrollment reps, calibrate on the other 25
  static-cross    train on session 1, calibrate on session 2
  aci-g           static-random + ACI on the genuine stream (gamma = g)
  login-recal     at the start of each session the first 10 reps are treated as
                  explicitly authenticated (password / Touch ID) and replace the
                  oldest calibration scores (window of 50)
FAR for adaptive variants is computed with the threshold in force at the start
of each session, averaged over sessions.

    python experiments/keystroke_drift.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bib.conformal import aci, pvalues  # noqa: E402
from bib.datasets import CMU_FIELD, CMU_LAB, load_cmu  # noqa: E402
from bib.models.detectors import AEConfig, DenseAE, ManhattanScaled  # noqa: E402
from bib.parallel import pmap  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
ALPHAS = (0.05, 0.10)
GAMMAS = (0.005, 0.02)
G = {}


def det_for(kind, seed):
    if kind == "manhattan_scaled":
        return ManhattanScaled()
    return DenseAE(AEConfig(**G["ae_cfg"]), seed=seed)


def subject(args):
    s, kind = args
    data, subjects = G["data"], G["subjects"]
    d = data[s]
    X, sess = d["X"], d["session"]
    pos_in_sess = np.zeros(len(X), int)
    for k in np.unique(sess):
        idx = np.nonzero(sess == k)[0]
        pos_in_sess[idx] = np.arange(len(idx))
    enroll = np.nonzero(sess <= 2)[0]
    later = sess >= 3
    test = np.nonzero(later & (pos_in_sess >= 10))[0]
    login = np.nonzero(later & (pos_in_sess < 10))[0]
    imp = np.concatenate([data[o]["X"][:5] for o in subjects if o != s])
    rng = np.random.default_rng(subjects.index(s))
    test_sessions = sorted(np.unique(sess[test]))
    out = {}

    # static-random (also the base for ACI and login-recal)
    cal = rng.choice(enroll, 25, replace=False)
    tr = np.setdiff1d(enroll, cal)
    det = det_for(kind, subjects.index(s)).fit(X[tr])
    sc_cal, sc_test, sc_imp, sc_login = det.score(X[cal]), det.score(X[test]), det.score(imp), det.score(X[login])
    # static-cross
    det2 = det_for(kind, subjects.index(s)).fit(X[sess == 1])
    c2, t2, i2 = det2.score(X[sess == 2]), det2.score(X[test]), det2.score(imp)

    def static(name, c, t, i):
        pt, pi = pvalues(c, t), pvalues(c, i)
        for a in ALPHAS:
            rej = pt <= a
            out[(name, a)] = {"frr": float(rej.mean()), "far": float(np.mean(pi > a)),
                              "frr_by_session": {int(k): float(rej[sess[test] == k].mean()) for k in test_sessions}}

    static("static-random", sc_cal, sc_test, sc_imp)
    static("static-cross", c2, t2, i2)

    # ACI over the time-ordered genuine stream (test reps are already in time order)
    for g in GAMMAS:
        for a in ALPHAS:
            rej, a_t = aci(sc_cal, sc_test, a, g)
            pi = pvalues(sc_cal, sc_imp)
            fars = []
            for k in test_sessions:
                a_start = a_t[np.nonzero(sess[test] == k)[0][0]]
                fars.append(float(np.mean(pi > a_start)))
            out[(f"aci-{g}", a)] = {"frr": float(rej.mean()), "far": float(np.mean(fars)),
                                    "frr_by_session": {int(k): float(rej[sess[test] == k].mean()) for k in test_sessions}}

    # login-recal: rolling calibration window fed by 10 explicitly-authenticated reps per session
    for a in ALPHAS:
        window = list(sc_cal)
        rejs, fars = np.zeros(len(test), bool), []
        for k in test_sessions:
            window = (window + list(sc_login[sess[login] == k]))[-50:]
            m = sess[test] == k
            rejs[m] = pvalues(np.array(window), sc_test[m]) <= a
            fars.append(float(np.mean(pvalues(np.array(window), sc_imp) > a)))
        out[("login-recal", a)] = {"frr": float(rejs.mean()), "far": float(np.mean(fars)),
                                   "frr_by_session": {int(k): float(rejs[sess[test] == k].mean()) for k in test_sessions}}
    return out


def aggregate(rs):
    agg = {}
    for key in rs[0]:
        name, a = key
        frr = np.array([r[key]["frr"] for r in rs])
        far = np.array([r[key]["far"] for r in rs])
        by = {}
        for r in rs:
            for k, v in r[key]["frr_by_session"].items():
                by.setdefault(k, []).append(v)
        agg[f"{name} @ alpha={a}"] = {
            "frr": round(float(frr.mean()), 4), "far": round(float(far.mean()), 4),
            "subjects_frr_over_2alpha": round(float(np.mean(frr > 2 * a)), 4),
            "frr_by_session": {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items()) if len(v) >= 10},
        }
    return agg


def main():
    t0 = time.time()
    cmu = OUT / "keystroke_cmu.json"
    ae_cfg = json.loads(cmu.read_text())["chosen"]["ae"] if cmu.exists() else {}
    G["ae_cfg"] = ae_cfg
    res = {"ae_cfg": ae_cfg}
    for pop, path in [("lab", CMU_LAB), ("field", CMU_FIELD)]:
        data = load_cmu(path)
        subjects = sorted(s for s in data if data[s]["session"].max() >= 5)
        G["data"], G["subjects"] = data, subjects
        for kind in ("manhattan_scaled", "ae"):
            rs = pmap(subject, [(s, kind) for s in subjects])
            res[f"{pop} / {kind}"] = {"n_subjects": len(subjects), **aggregate(rs)}
            print(f"== {pop} / {kind} ({len(subjects)} subjects)")
            for k, v in res[f"{pop} / {kind}"].items():
                if isinstance(v, dict):
                    print(f"  {k:32s} FRR {v['frr']:.3f}  FAR {v['far']:.3f}  >2a {v['subjects_frr_over_2alpha']:.2f}  "
                          f"by session {list(v['frr_by_session'].values())}")
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "keystroke_drift.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
