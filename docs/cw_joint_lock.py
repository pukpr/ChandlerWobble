#!/usr/bin/env python3
"""Joint chance of the two pre-specified lunar phase locks, and the expected strength of the Mm line (2026-10-23).
Discussion: cw_data_tests.md, section 5; cw_synchronized_wobble.md, section 4. Needs data/finals2000A.all (fetch_eop.py).

Pre-specified lines: the draconic B^2 alias (CW, 0.8440 /yr) and the anomalistic alias (Mm, 14 - A = 0.7448 /yr).
Null model: a free wobble (432.3 d, Q = 50, 85, 150) driven by white noise, plus a small white floor, sampled and
windowed like the data (four 13.5-yr windows, 1973-2027). Counted: CW phase spread <= observed; Mm phase spread
<= observed; Mm amplitude max/min <= observed (a forced line predicts a steady amplitude).
Mm strength: kicks sin^2(beta) (a/r)^3 ~ (1/2 - cos(2F)/2)(1 + 3e cos M) on the semiannual comb; each comb harmonic
k attenuated by Gaussian pulses of rms width tau; response through the prograde mode (Q, 432.3 d).
    python3 cw_joint_lock.py  -> cw_joint_lock.json
"""
import json
from pathlib import Path

import numpy as np

import cw_eop_closure as E
import cw_peaks as K

HERE = Path(__file__).resolve().parent
YEAR = 365.2422
F_D, F_A = YEAR / 27.212221, YEAR / 27.554550
F_CW, F_MM = 2 * F_D - 26, 14 - F_A
F0 = YEAR / 432.3
ECC = 0.0549
WINS = [(1973, 1986.5), (1986.5, 2000), (2000, 2013.5), (2013.5, 2027)]


def spread(c):
    ph = np.angle(c); mean = np.angle(np.exp(1j * ph).mean(0))
    return np.degrees(np.abs(np.angle(np.exp(1j * (ph - mean)))).max(0))


def window_fit(t, z):
    cs, ms = [], []
    for a, b in WINS:
        m = (t >= a) & (t < b); tt = t[m]
        G = np.column_stack([np.exp(2j * np.pi * f * tt) for f in (F_CW, F_MM, 1.0, -1.0)] + [np.ones_like(tt)])
        c = np.linalg.lstsq(G, z[m], rcond=None)[0]; cs.append(c[0]); ms.append(c[1])
    return np.array(cs), np.array(ms)


def observed():
    tb, xb, yb = E.load_finals(); p = xb - 1j * yb
    cs, ms = window_fit(tb, p)
    return dict(cw_spread=float(spread(cs[:, None])[0]), mm_spread=float(spread(ms[:, None])[0]),
                mm_amp_ratio=float(np.abs(ms).max() / np.abs(ms).min()),
                cw_amp_mas=[float(v) for v in np.abs(cs) * 1e3], mm_amp_mas=[float(v) for v in np.abs(ms) * 1e3])


def null_mc(obs, q, batches=10, nsim=4000, seed=11):
    rng = np.random.default_rng(seed + q); dt = 0.05; n_keep = int(54 / dt); n = n_keep + int(200 / dt)
    t = np.arange(n_keep) * dt + 1973
    step = np.exp((2j * np.pi * F0 - np.pi * F0 / q) * dt)
    cnt = dict(cw=0, mm=0, mm_amp=0, mm_both=0, cw_mm=0, all=0); tot = 0
    for _ in range(batches):
        acc = np.zeros(nsim, complex); z = np.empty((n_keep, nsim), complex)
        for i in range(n):
            acc = acc * step + rng.normal(size=nsim) + 1j * rng.normal(size=nsim)
            if i >= n - n_keep:
                z[i - (n - n_keep)] = acc
        z = z / np.std(z, axis=0) + 0.014 * (rng.normal(size=(n_keep, nsim)) + 1j * rng.normal(size=(n_keep, nsim)))
        cs, ms = window_fit(t, z)
        A = spread(cs) <= obs["cw_spread"]; B = spread(ms) <= obs["mm_spread"]
        C = np.abs(ms).max(0) / np.abs(ms).min(0) <= obs["mm_amp_ratio"]
        for k, v in dict(cw=A, mm=B, mm_amp=C, mm_both=B & C, cw_mm=A & B, all=A & B & C).items():
            cnt[k] += int(v.sum())
        tot += nsim
    return {k: v / tot for k, v in cnt.items()} | {"n": tot, "hits_all": cnt["all"]}


def mm_strength():
    lines = {"CW": (F_CW, 0.5, 26), "Mm": (F_MM, 1.5 * ECC, 14), "Drac2 1.156": (28 - 2 * F_D, 0.5, 28),
             "2F-M 0.4115": (14 - (2 * F_D - F_A), 0.75 * ECC, 14), "semiannual": (2.0, 0.5, 2)}
    out = {}
    for q, tau_d in ((85, 0), (85, 4), (85, 6), (5, 0), (20, 0)):
        lam = F0 / (2 * q); tau = tau_d / YEAR
        H = lambda f: 1 / abs(1j * (f - F0) + lam)
        g = lambda k: np.exp(-0.5 * (2 * np.pi * k * tau) ** 2)
        amp = {nm: a * g(k) * H(f) for nm, (f, a, k) in lines.items()}
        out[f"Q={q}, pulse rms {tau_d} d"] = {nm: float(v / amp["CW"]) for nm, v in amp.items()}
    return out


def main():
    obs = observed()
    res = {"observed": obs, "null": {f"Q={q}": null_mc(obs, q) for q in (50, 85, 150)}, "mm_strength_model": mm_strength()}
    tb, xb, yb = E.load_finals(); p = xb - 1j * yb; m = (tb >= 1973) & (tb < 2013.5)
    fr = [F_CW, F_MM, 28 - 2 * F_D, 14 - (2 * F_D - F_A), 2.0, 1.0, -1.0, -2.0]
    c, _, _ = E.fit_lines(tb[m], p[m], fr)
    res["mm_strength_observed_1973_2013"] = {nm: float(abs(c[f]) / abs(c[F_CW])) for nm, f in
                                             (("Mm", F_MM), ("Drac2 1.156", 28 - 2 * F_D), ("2F-M 0.4115", 14 - (2 * F_D - F_A)),
                                              ("semiannual", 2.0))}
    (HERE / "cw_joint_lock.json").write_text(json.dumps(res, indent=1))
    print("observed:", obs)
    for k, v in res["null"].items():
        print(k, {kk: (f"{vv:.5f}" if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("Mm strength, observed 1973-2013:", {k: round(v, 3) for k, v in res["mm_strength_observed_1973_2013"].items()})
    for k, v in res["mm_strength_model"].items():
        print("  model", k, {kk: round(vv, 3) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
