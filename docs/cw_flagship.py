#!/usr/bin/env python3
"""Flagship figure for docs/README.md: the essential fit, on raw IERS polar motion (2026-10-23).

The reconciled device (cw_synchronized_wobble.md):
  forcing   sin^2(beta) (a/r)^3 from Meeus' lunar elements (draconic latitude, anomalistic distance): astronomical phase
  carrier   semiannual comb of push-pull inertia pulses that gate the lunar torque: the drive is torque(t) times the
            pulse train, each pulse a Gaussian/derivative-of-Gaussian mix with the rms width from cw_bandpass.py,
            alternate pulses 1 +- a (gating, not instantaneous sampling: the pulse window averages the torque)
  response  one prograde mode, natural frequency f0 and Q from cw_bandpass.py (the lowest Q the line set allows)
  annual    its own seasonal excitation (regression terms at +-1 /yr), as in geodesy
Fitted: the comb timing (one parameter) and linear terms (complex scale, annual, quadratic trend) on 1973-2000;
2000-2026.8 held out. Data: raw complex polar motion p = x - i y (fetch_eop.py).
    python3 cw_flagship.py  -> cw_flagship.png (needs cw_bandpass.json)
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import fftconvolve

import cw_eop_closure as E

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
YEAR = 365.2422
F_CW, F_MM = 2 * YEAR / 27.212221 - 26, 14 - YEAR / 27.554550
DT = 0.002
TG = np.arange(1880.0, 2027.0, DT)
TRAIN, HOLD = (1973.0, 2000.0), (2000.0, 2026.8)
rad = np.radians


def moon(t):
    T = (t - 2000.0) * 365.25 / 36525.0
    D, Mp, F = (rad(v) for v in (297.8501921 + 445267.1114034 * T, 134.9633964 + 477198.8675055 * T,
                                  93.2720950 + 483202.0175233 * T))
    beta = rad(5.128) * np.sin(F) + rad(0.281) * np.sin(Mp + F) + rad(0.278) * np.sin(Mp - F) + rad(0.173) * np.sin(2 * D - F)
    r = 385000.56 - 20905.355 * np.cos(Mp) - 3699.111 * np.cos(2 * D - Mp) - 2955.968 * np.cos(2 * D) - 569.925 * np.cos(2 * Mp)
    return (np.sin(beta) / np.sin(rad(5.145))) ** 2 * (385000.56 / r) ** 3


def params():
    bp = json.loads((HERE / "cw_bandpass.json").read_text())
    q, misfit, f0, tau, b = min(bp["Q_profile"], key=lambda r: (r[1] > 0.01, r[0]))   # lowest Q that still fits
    a = bp["runs"]["all free"]["a"]
    return dict(f0=f0, Q=q, tau_days=tau, b=b, a=a)


def kernel(P):
    """Pulse shape (one-way Gaussian mixed with its derivative), and the mode's impulse response."""
    tau = P["tau_days"] / YEAR; s = np.arange(-6 * tau, 6 * tau + DT, DT)
    g = np.exp(-0.5 * (s / tau) ** 2); g /= g.sum()
    dg = -s / tau ** 2 * g                                   # derivative: spectrum 2 pi f G(f)
    pulse = (1 - P["b"]) * g + P["b"] * dg / (2 * np.pi * 14)   # matches w(k) = [(1 - b) + b k / 14] G(k) of cw_bandpass.py
    lam = P["f0"] / (2 * P["Q"])
    h = np.exp((2j * np.pi * P["f0"] - 2 * np.pi * lam) * np.arange(0, 8 * P["Q"] / P["f0"], DT)) * DT
    return pulse, len(s) // 2, h


LUNAR = moon(TG)


def response(t0, P, K):
    pulse, lag, h = K
    years = np.arange(np.floor(TG[0]), np.ceil(TG[-1]))
    tn = np.sort(np.concatenate([years + t0, years + t0 + 0.5])); tn = tn[(tn > TG[0]) & (tn < TG[-1])]
    north = np.isclose((tn - t0) % 1.0, 0.0, atol=1e-6) | np.isclose((tn - t0) % 1.0, 1.0, atol=1e-6)
    w = np.where(north, 1 + P["a"], 1 - P["a"])              # pulse weights only; the torque enters by gating below
    imp = np.zeros(len(TG)); np.add.at(imp, np.round((tn - TG[0]) / DT).astype(int), w)
    gate = fftconvolve(imp, pulse)[lag: lag + len(TG)]           # the pulse train (with its N/S weights)
    return fftconvolve(LUNAR * gate, h)[: len(TG)]               # torque gated by the pulses, through the mode


def design(t, z):
    tc = (t - 1995) / 50
    return np.column_stack([z, np.exp(2j * np.pi * t), np.exp(-2j * np.pi * t), np.ones_like(t), tc, tc ** 2])


def fit(t, p, z, win):
    m = (t >= win[0]) & (t < win[1])
    c, *_ = np.linalg.lstsq(design(t[m], z[m]), p[m], rcond=None)
    return c


def chandler_band(t, z):
    """Remove long-period drift (< 0.3 /yr) so the comparison is not set by trend extrapolation."""
    n = len(z); f = np.fft.fftfreq(n, float(np.median(np.diff(t))))
    Z = np.fft.fft(z - np.polyval(np.polyfit(t - t.mean(), z, 1), t - t.mean())); Z[np.abs(f) < 0.3] = 0
    return np.fft.ifft(Z)


def corr(a, b):
    return float(np.abs(np.vdot(a - a.mean(), b - b.mean())) / (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean())))


def main():
    P = params(); K = kernel(P)
    tb, xb, yb = E.load_finals(); p = xb - 1j * yb
    best = None
    for t0 in np.arange(0, 0.5, 1 / YEAR):
        z = np.interp(tb, TG, response(t0, P, K).real) + 1j * np.interp(tb, TG, response(t0, P, K).imag)
        c = fit(tb, p, z, TRAIN); m = (tb >= TRAIN[0]) & (tb < TRAIN[1])
        r = corr(design(tb[m], z[m]) @ c, p[m])
        if best is None or r > best[0]:
            best = (r, t0, z, c)
    r_tr, t0, z, c = best
    model = design(tb, z) @ c
    mh = (tb >= HOLD[0]) & (tb < HOLD[1]); mt = (tb >= TRAIN[0]) & (tb < TRAIN[1])
    # remove the shared annual + trend part to compare the lunar (Chandler-band) content alone
    ann = c[1] * np.exp(2j * np.pi * tb) + c[2] * np.exp(-2j * np.pi * tb)
    lun_obs, lun_mod = chandler_band(tb, p - ann), chandler_band(tb, c[0] * z)
    r_ho = corr(model[mh], p[mh]); r_lun_tr = corr(lun_mod[mt], lun_obs[mt]); r_lun_ho = corr(lun_mod[mh], lun_obs[mh])
    s = np.exp(2j * np.pi * F_CW * tb)                                  # baseline: a sinusoid at the CW alias
    cs = fit(tb, p, s, TRAIN)
    ann_s = cs[1] * np.exp(2j * np.pi * tb) + cs[2] * np.exp(-2j * np.pi * tb)
    r_sin_ho = corr(chandler_band(tb, cs[0] * s)[mh], chandler_band(tb, p - ann_s)[mh])
    print(f"params {P}; comb t0 day {t0 * YEAR:.0f}")
    print(f"full p: train r {r_tr:.3f}, held-out r {r_ho:.3f}; Chandler-band part: train {r_lun_tr:.3f}, held-out {r_lun_ho:.3f}; "
          f"sinusoid held-out {r_sin_ho:.3f}")
    plot(tb, p, model, lun_obs, lun_mod, P, t0, dict(r_tr=r_tr, r_ho=r_ho, r_lun_tr=r_lun_tr, r_lun_ho=r_lun_ho, r_sin_ho=r_sin_ho))


def plot(tb, p, model, lun_obs, lun_mod, P, t0, R):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([0.06, 0.56, 0.9, 0.38])
    ax.plot(tb, 1e3 * lun_obs.real, color="k", lw=0.9, label="raw IERS x-pole, annual removed, periods > 3.3 yr filtered out")
    ax.plot(tb, 1e3 * lun_mod.real, color="#d94a4a", lw=1.1,
            label=f"device: lunar B²(a/r)³ on a push-pull semiannual comb, f0 = {P['f0']:.3f} /yr, Q = {P['Q']:.0f}")
    ax.axvspan(*HOLD, color="0.93", zorder=0); ax.text(2013, 230, "held out", ha="center", fontsize=9)
    ax.set_xlim(1972, 2027.5); ax.set_ylim(-280, 280); ax.set_ylabel("mas"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"(a) Chandler band, fit 1973–2000 (r = {R['r_lun_tr']:.2f}), held out 2000–2026 (r = {R['r_lun_ho']:.2f}; "
                 f"a fixed sinusoid at 0.8440: {R['r_sin_ho']:.2f}). One fitted timing parameter.", fontsize=10)
    ax = fig.add_axes([0.06, 0.07, 0.55, 0.38])
    for z, col, lab in ((p, "k", "raw IERS p = x − iy"), (model, "#d94a4a", "device + annual")):
        zz = (z - z.mean()) * np.hanning(len(z)); n = len(z) * 8
        f = np.fft.fftshift(np.fft.fftfreq(n, 0.05)); A = np.abs(np.fft.fftshift(np.fft.fft(zz, n))) / len(z) * 4e3
        ax.semilogy(f, A, color=col, lw=1, label=lab)
    for v, lab in ((F_CW, "CW"), (F_MM, "Mm"), (1, "annual"), (2, "semi"), (2 - F_CW, "2−CW"), (-1, "annual (retro)")):
        ax.axvline(v, color="g", ls=":", lw=0.8); ax.text(v + 0.015, 250, lab, fontsize=7, color="g", va="top")
    ax.set_xlim(-1.3, 2.3); ax.set_ylim(0.05, 300); ax.set_xlabel("frequency (cycles/yr; + prograde)"); ax.set_ylabel("amplitude (mas)")
    ax.legend(fontsize=8, loc="lower left"); ax.set_title("(b) Spectrum: every comb line at about the observed strength", fontsize=10)
    ax = fig.add_axes([0.68, 0.07, 0.29, 0.38])
    f = np.linspace(-1.3, 2.3, 2000); lam = P["f0"] / (2 * P["Q"]); H = 1 / np.abs(lam - 1j * (f - P["f0"]))
    ax.semilogy(f, H / H.max(), color="#d94a4a")
    lamc = (YEAR / 432.3) / 170; Hc = 1 / np.abs(lamc - 1j * (f - YEAR / 432.3)); ax.semilogy(f, Hc / Hc.max(), color="0.6", lw=1)
    for v in (F_MM, F_CW, 1, 2, 2 - F_CW):
        ax.axvline(v, color="g", ls=":", lw=0.8)
    ax.set_ylim(1e-4, 2); ax.set_xlabel("frequency (cycles/yr)"); ax.set_ylabel("|H| (normalized)")
    ax.set_title(f"(c) Fitted bandpass (red, Q = {P['Q']:.0f}) vs\nconsensus free mode (grey, 432.3 d, Q = 85)", fontsize=10)
    fig.savefig(HERE / "cw_flagship.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
