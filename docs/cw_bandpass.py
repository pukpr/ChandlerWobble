#!/usr/bin/env python3
"""A bandpass that lets every comb line emerge at its observed strength (2026-10-23).
Discussion: cw_synchronized_wobble.md, section 3; cw_data_tests.md, section 6. Needs data/finals2000A.all (fetch_eop.py).

No consensus constraints: the natural frequency f0 and Q are free (a forced wobble needs no high Q to keep ringing),
and so are the pulse shape and the north/south asymmetry.

Forcing (wavenumber 0, sign-blind): kick = sin^2(beta) (a/r)^3 ~ (1/2 - cos 2F / 2)(1 + 3e cos M) sampled by a
semiannual comb whose alternate pulses are 1 + a and 1 - a. Comb harmonic k carries weight 1 (even k) or a (odd k),
times the pulse-shape factor w(k) = exp(-(2 pi k tau)^2 / 2) [(1 - b) + b k / 14]: tau is the rms width, b = 0 a
one-way pulse, b = 1 a push-pull (rate-of-change) pulse. Each real line at nu reaches polar motion through the
prograde mode H(f) = 1 / (lambda - i (f - f0)), lambda = f0 / 2Q, at +nu and at -nu.
Observed: prograde and retrograde amplitudes of every line, fitted jointly to raw complex polar motion 1973-2026.
Fit: log-amplitude misfit (weights favour the clean, strong lines), differential evolution over f0, Q, tau, b, a.
    python3 cw_bandpass.py  -> cw_bandpass.png, cw_bandpass.json
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import differential_evolution

import cw_eop_closure as E

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
YEAR = 365.2422
F2F = 2 * YEAR / 27.212221          # 26.8438, twice the draconic frequency
FM = YEAR / 27.554550               # 13.2553, anomalistic
ECC = 0.0549

# (label, source frequency, source amplitude, comb harmonic k)
SOURCES = {"DC": (0.0, 0.5), "2F": (F2F, 0.5), "M": (FM, 1.5 * ECC), "2F-M": (F2F - FM, 0.75 * ECC), "2F+M": (F2F + FM, 0.75 * ECC)}
LINES = [("CW", "2F", 26), ("2−CW", "2F", 28), ("2+CW", "2F", 24), ("1−CW", "2F", 27), ("1+CW", "2F", 25),
         ("Mm", "M", 14), ("annual", "DC", 1), ("semiannual", "DC", 2), ("2F−M", "2F-M", 14)]
# Fitted lines (log misfit) and upper-limit lines (penalized only when the model exceeds observed + noise floor).
# The annual line also has its own excitation: the seasonal mass shift has a sign (north, then south), so its direct
# effect alternates and feeds the odd harmonics (annual); the lunar coupling rides on the magnitude of the inertia
# pulse and feeds the even ones (CW, Mm, semiannual). The direct annual amplitude is a free parameter.
FIT = {"CW": 3, "Mm": 2, "semiannual": 1.5, "annual": 1.0}
UPPER = {"2−CW": 1, "1+CW": 1, "2+CW": 1, "1−CW": 1, "2F−M": 1}
FLOOR_MAS = 1.0                      # typical amplitude of a noise line in raw p near these frequencies


def line_freq(src, k):
    return abs(k - SOURCES[src][0]) if src != "DC" else float(k)


def observed():
    tb, xb, yb = E.load_finals(); p = xb - 1j * yb
    fr = []
    for name, src, k in LINES:
        nu = line_freq(src, k); fr += [nu, -nu]
    c, _, _ = E.fit_lines(tb, p, list(dict.fromkeys(fr)))
    return {name: (float(abs(c[line_freq(src, k)]) * 1e3), float(abs(c[-line_freq(src, k)]) * 1e3)) for name, src, k in LINES}


def model(x):
    f0, logq, tau_d, b, a, ann = x
    lam = f0 / (2 * 10 ** logq); tau = tau_d / YEAR
    H = lambda f: 1 / abs(lam - 1j * (f - f0))
    w = lambda k: np.exp(-0.5 * (2 * np.pi * k * tau) ** 2) * abs((1 - b) + b * k / 14)
    out = {}
    for name, src, k in LINES:
        nu = line_freq(src, k); amp = SOURCES[src][1] * w(k) * (a if k % 2 else 1.0) / 2
        if name == "annual":
            amp = amp + ann                                   # direct seasonal excitation (relative units)
        out[name] = (amp * H(nu), amp * H(-nu))
    return out


def misfit(x, obs):
    m = model(x)
    s = obs["CW"][0] / m["CW"][0]                           # scale set by the CW
    r = 0.0
    for n, wgt in FIT.items():
        r += wgt * np.log(s * m[n][0] / obs[n][0]) ** 2
    for n, wgt in UPPER.items():
        r += wgt * max(0.0, np.log(s * m[n][0] / (obs[n][0] + FLOOR_MAS))) ** 2
    return r


def main():
    obs = observed()
    bounds = [(0.70, 1.30), (0.0, 2.5), (0.0, 12.0), (0.0, 1.0), (0.0, 1.0), (0.0, 50.0)]
    runs = {}
    for label, bnd in {"all free": bounds,
                       "consensus (f0 = 432.3 d, Q = 85)": [(YEAR / 432.3, YEAR / 432.3 + 1e-9), (np.log10(85), np.log10(85) + 1e-9)] + bounds[2:],
                       "f0 free, Q >= 50": [bounds[0], (np.log10(50), 2.5)] + bounds[2:]}.items():
        res = differential_evolution(misfit, bnd, args=(obs,), seed=2, popsize=40, maxiter=600, tol=1e-10, polish=True)
        m = model(res.x); s = obs["CW"][0] / m["CW"][0]
        runs[label] = dict(f0=res.x[0], period_days=YEAR / res.x[0], Q=10 ** res.x[1], tau_days=res.x[2], b=res.x[3],
                           a=res.x[4], annual_direct=res.x[5], misfit=res.fun,
                           lines={n: dict(obs_pro=obs[n][0], obs_retro=obs[n][1], mod_pro=s * m[n][0], mod_retro=s * m[n][1]) for n in m})
    # how flat is the optimum in f0 and Q? profile the misfit
    prof = []
    for f0 in np.linspace(0.74, 0.95, 22):
        r = differential_evolution(misfit, [(f0, f0 + 1e-9)] + bounds[1:], args=(obs,), seed=3, popsize=20, maxiter=300, tol=1e-9)
        prof.append((f0, r.fun, 10 ** r.x[1], r.x[2], r.x[3]))
    qprof = []
    for lq in np.linspace(0.0, 2.5, 11):
        r = differential_evolution(misfit, [bounds[0], (lq, lq + 1e-9)] + bounds[2:], args=(obs,), seed=3, popsize=20, maxiter=300, tol=1e-9)
        qprof.append((10 ** lq, r.fun, r.x[0], r.x[2], r.x[3]))
    out = {"observed_mas": obs, "runs": runs, "f0_profile": prof, "Q_profile": qprof}
    (HERE / "cw_bandpass.json").write_text(json.dumps(out, indent=1, default=float))
    report(out)
    plot(out)


def report(out):
    for label, r in out["runs"].items():
        print(f"== {label}: f0 {r['f0']:.4f} /yr ({r['period_days']:.0f} d)  Q {r['Q']:.1f}  pulse rms {r['tau_days']:.1f} d  "
              f"push-pull b {r['b']:.2f}  N/S asymmetry a {r['a']:.3f}  direct annual {r['annual_direct']:.2f}  misfit {r['misfit']:.3f}")
        for n, d in r["lines"].items():
            print(f"   {n:10s} observed {d['obs_pro']:7.2f} / {d['obs_retro']:6.2f}   model {d['mod_pro']:7.2f} / {d['mod_retro']:6.2f}  mas (pro / retro)")
    print("f0 profile (f0, misfit, Q, tau d, b):", [(round(f, 3), round(m, 3), round(q, 1), round(t, 1), round(b, 2)) for f, m, q, t, b in out["f0_profile"]])
    print("Q profile (Q, misfit, f0, tau d, b):", [(round(q, 1), round(m, 3), round(f, 3), round(t, 1), round(b, 2)) for q, m, f, t, b in out["Q_profile"]])


def plot(out):
    r = out["runs"]["all free"]; c = out["runs"]["consensus (f0 = 432.3 d, Q = 85)"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    ax = axs[0]
    f = np.linspace(-3, 3, 3001)
    for rr, col, lab in ((r, "#d94a4a", "fitted"), (c, "0.5", "consensus")):
        lam = rr["f0"] / (2 * rr["Q"]); H = 1 / np.abs(lam - 1j * (f - rr["f0"]))
        ax.semilogy(f, H / H.max(), color=col, lw=1.5,
                    label=f"{lab}: f0 = {rr['f0']:.3f} /yr ({rr['period_days']:.0f} d), Q = {rr['Q']:.1f}")
    for name, src, k in LINES:
        nu = line_freq(src, k); ax.axvline(nu, color="g", ls=":", lw=0.8)
        ax.text(nu, 1.4, name, rotation=90, fontsize=7, color="g", ha="center", va="bottom")
    ax.set_xlim(-2.2, 3.0); ax.set_ylim(1e-4, 8); ax.set_xlabel("frequency (cycles/yr; + prograde)"); ax.set_ylabel("|H| (normalized)")
    ax.legend(fontsize=7, loc="lower left"); ax.set_title("(a) The bandpass: fitted vs consensus", fontsize=10)
    for ax, rr, ttl in ((axs[1], r, "(b) Fitted bandpass: line amplitudes"), (axs[2], c, "(c) Consensus f0, Q")):
        names = list(rr["lines"]); x = np.arange(len(names))
        ax.bar(x - 0.2, [rr["lines"][n]["obs_pro"] for n in names], 0.4, color="k", label="observed (raw IERS)")
        ax.bar(x + 0.2, [rr["lines"][n]["mod_pro"] for n in names], 0.4, color="#d94a4a" if rr is r else "0.5", label="model")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, fontsize=8); ax.set_ylabel("prograde amplitude (mas)")
        ax.set_ylim(0.05, 400); ax.legend(fontsize=7); ax.set_title(ttl, fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "cw_bandpass.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
