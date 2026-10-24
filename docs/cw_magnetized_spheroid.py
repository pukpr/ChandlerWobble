#!/usr/bin/env python3
"""Chandler wobble as a polar-magnetized, pear-shaped spheroid in a pulsed coaxial electromagnet (2026-10-23).
Derivation and results: cw_magnetized_spheroid.md (the device appendix, same directory).

The device (a macrospin, i.e. the uniform k = 0 Kittel mode of a ferromagnetic ellipsoid):
  spin            unit magnetization m, polar (along z) at rest; shape-anisotropy tensor N plays the inverse inertia
                  tensor I^-1, so free precession of m is Euler's free nutation (Kittel mode = Chandler wobble)
  damping         Gilbert alpha = 1 / (2 Q), Q = 85
  electromagnet   coaxial (wavenumber 0) coil, field B(t) proportional to the Moon's ecliptic latitude sin(beta), whose
                  period is the draconic month; the torque on the tilted, shape-anisotropic body is quadratic, B^2
                  (an induced moment is torqued regardless of the field's sign, as a tidal bulge is)
  inertia comb    the shape (moment of inertia) is pulsed twice a year; the torque reaches the wobble only through
                  these pulses: kick_n = w_n * B(t_n)^2, the comb being the carrier and B^2 the modulator
  pear shape      unequal north/south pulses, w_n = 1 + a, 1 - a alternately (the comb gains odd harmonics)

Outputs:
  cw_spheroid_equivalence.png  Euler rigid body vs Landau-Lifshitz macrospin, and the device schematic
  cw_spheroid_spectra.png      alias table check, device variants against cw.dat, fit and held-out window
  cw_magnetized_spheroid.json  all numbers quoted in the .md
    python3 cw_magnetized_spheroid.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon  # noqa: E402

HERE = Path(__file__).resolve().parent
CW_DAT = HERE.parent / "cw.dat"
YEAR = 365.2422
MONTHS = {"draconic": 27.212221, "tropical": 27.321582, "anomalistic": 27.554550, "synodic": 29.530589}
T_FREE = 432.3 / YEAR                  # free (Kittel / Chandler) period, years
Q = 85.0
ALPHA = 1 / (2 * Q)                    # Gilbert damping
OMEGA = 2 * np.pi / T_FREE
LAM = ALPHA * OMEGA                    # amplitude decay rate of the linearized Kittel mode
DT = 0.005
T0_SIM = 1800.0                        # spin-up start, so transients have decayed (Q/pi*T ~ 32 yr e-fold)
TG = np.arange(T0_SIM, 2019.0, DT)
TRAIN, HOLD = (1935.0, 1990.0), (1990.0, 2018.7)
rad = np.radians


# ---------------------------------------------------------------- 1. Euler top == Landau-Lifshitz macrospin
def euler_rhs(_, L, Iinv):
    return np.cross(L, Iinv * L)       # body frame: dL/dt = L x (I^-1 L)


def ll_rhs(_, m, gMsN):
    return np.cross(m, gMsN * m)       # dm/dt = -gamma m x H_eff, H_eff = -Ms N m  ->  gamma Ms m x (N m)


def equivalence():
    A, B, C = 0.80, 0.90, 1.00         # exaggerated triaxial body so the check runs in a few periods
    Iinv = 1 / np.array([A, B, C])
    Lmag = 2 * np.pi
    L0 = Lmag * np.array([np.sin(0.01), 0.0, np.cos(0.01)])
    span, te = (0, 300), np.linspace(0, 300, 30001)
    e = solve_ivp(euler_rhs, span, L0, t_eval=te, args=(Iinv,), rtol=1e-11, atol=1e-12)
    gMsN = Lmag * Iinv                 # the mapping: gamma Ms N = |L| I^-1
    m = solve_ivp(ll_rhs, span, L0 / Lmag, t_eval=te, args=(gMsN,), rtol=1e-11, atol=1e-12)
    diff = np.max(np.abs(e.y / Lmag - m.y))
    x = m.y[0] - m.y[0].mean()
    f = np.fft.rfftfreq(len(x) * 16, te[1]); P = np.abs(np.fft.rfft(x, len(x) * 16))
    f_num = f[np.argmax(P[1:]) + 1]
    Nx, Ny, Nz = gMsN
    f_kittel = np.sqrt((Nx - Nz) * (Ny - Nz)) / (2 * np.pi)          # Kittel, zero applied field
    f_euler = Lmag / C * np.sqrt((C - A) * (C - B) / (A * B)) / (2 * np.pi)  # Euler, Omega = |L|/C
    sid = 0.99726958                                                   # sidereal day, days
    earth = {"H_dynamical_ellipticity": 0.0032738, "euler_period_days": sid / 0.0032738,
             "observed_period_days": 432.3, "lengthening_factor": 432.3 / (sid / 0.0032738)}
    return e, m, te, dict(max_abs_diff=float(diff), f_numerical=float(f_num), f_kittel=float(f_kittel),
                          f_euler=float(f_euler), earth=earth)


# ---------------------------------------------------------------- 2. alias algebra
def alias(f_line, comb_alternating):
    """Nearest-to-zero line when a signal at f_line (cycles/yr) is sampled by a semiannual comb.
    Same-polarity comb: harmonics at even integers; alternating polarity: at odd integers."""
    grid = np.arange(1, 60, 2) if comb_alternating else np.arange(0, 60, 2)
    return float(np.min(np.abs(f_line - grid)))


def alias_table():
    rows = []
    for name, P in MONTHS.items():
        f = YEAR / P
        for power in (1, 2):
            for alt in (False, True):
                rows.append(dict(month=name, power=power, alternating=alt, line=alias(power * f, alt)))
    return rows


# ---------------------------------------------------------------- 3. astronomy (Meeus mean elements, as in QBO)
def moon(t):
    T = (t - 2000.0) * 365.25 / 36525.0
    Lp, D, M, Mp, F = (rad(v) for v in (218.3164477 + 481267.88123421 * T, 297.8501921 + 445267.1114034 * T,
                                         357.5291092 + 35999.0502909 * T, 134.9633964 + 477198.8675055 * T,
                                         93.2720950 + 483202.0175233 * T))
    lam = Lp + rad(6.289) * np.sin(Mp) + rad(1.274) * np.sin(2 * D - Mp) + rad(0.658) * np.sin(2 * D)
    beta = rad(5.128) * np.sin(F) + rad(0.281) * np.sin(Mp + F) + rad(0.278) * np.sin(Mp - F) + rad(0.173) * np.sin(2 * D - F)
    eps = rad(23.439)
    sdec = np.sin(beta) * np.cos(eps) + np.cos(beta) * np.sin(eps) * np.sin(lam)
    return np.sin(beta) / np.sin(rad(5.145)), sdec / np.sin(rad(23.44 + 5.145))   # both normalized to ~unit peak


# ---------------------------------------------------------------- 4. the device
def comb_times(t0):
    years = np.arange(np.floor(TG[0]), np.ceil(TG[-1]) + 1)
    tn = np.concatenate([years + t0, years + t0 + 0.5]); order = np.argsort(tn); tn = tn[order]
    north = np.concatenate([np.ones_like(years), np.zeros_like(years)])[order].astype(bool)
    keep = (tn >= TG[0]) & (tn < TG[-1])
    return tn[keep], north[keep]


def field(t, which):
    sb, sd = moon(t)
    return {"beta": sb, "decl": sd}[which]


def kicks(t0, a, which="beta", power=2, alternating=False):
    tn, north = comb_times(t0)
    B = field(tn, which)
    w = np.where(north, 1 + a, 1 - a)
    if alternating:
        w = w * np.where(north, 1.0, -1.0)
    return tn, w * B ** power


def response_linear(tn, k, continuous=None):
    """Linearized Kittel mode with Gilbert damping, p = m_x + i m_y: dp/dt = (i w - lam) p + forcing.
    Each kick rotates m about y, i.e. adds a real step to p. Solved by convolution with the exact Green's function."""
    drive = np.zeros_like(TG)
    if continuous is None:
        np.add.at(drive, np.minimum(np.round((tn - TG[0]) / DT).astype(int), len(TG) - 1), k)
    else:
        drive = continuous * DT
    g = np.exp((1j * OMEGA - LAM) * np.arange(0, 6 / LAM, DT))
    return fftconvolve(drive, g)[: len(TG)]


def response_parametric(tn, k, eps, north):
    """As response_linear, plus the inertia pulse itself: while the shape is pulsed the Kittel (Euler) frequency
    shifts, so each pulse advances the precession phase by eps_n (north and south pulses unequal)."""
    idx = np.minimum(np.round((tn - TG[0]) / DT).astype(int), len(TG) - 1)
    step = np.exp((1j * OMEGA - LAM) * DT)
    kick = np.zeros(len(TG), complex); rot = np.ones(len(TG), complex)
    np.add.at(kick, idx, k); rot[idx] = np.exp(1j * eps)
    p = np.empty(len(TG), complex); z = 0j
    for i in range(len(TG)):
        z = z * rot[i] + kick[i]; p[i] = z; z *= step
    return p


def response_llg(tn, k, scale):
    """Full nonlinear Landau-Lifshitz-Gilbert macrospin with uniaxial shape anisotropy, kicks as finite rotations."""
    def rhs(m):
        h = np.array([0.0, 0.0, OMEGA * m[2]])            # gamma Ms (N_perp - N_z) m_z along z: Kittel freq OMEGA
        pre = -np.cross(m, h)
        return (pre + ALPHA * np.cross(m, pre)) / (1 + ALPHA ** 2)
    m = np.array([0.0, 0.0, 1.0]); out = np.empty((len(TG), 2)); j = 0
    for i, t in enumerate(TG):
        while j < len(tn) and tn[j] <= t + DT / 2:
            th = scale * k[j]; c, s = np.cos(th), np.sin(th)
            m = np.array([c * m[0] + s * m[2], m[1], -s * m[0] + c * m[2]]); j += 1
        out[i] = m[0], m[1]
        k1 = rhs(m); k2 = rhs(m + DT / 2 * k1); k3 = rhs(m + DT / 2 * k2); k4 = rhs(m + DT * k3)
        m = m + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); m /= np.linalg.norm(m)
    return out[:, 0] + 1j * out[:, 1]


# ---------------------------------------------------------------- 5. data, spectra, fit
D = np.loadtxt(CW_DAT); TD, XD = D[:, 0], D[:, 1]


def on_data(p):
    return np.interp(TD, TG, p.real), np.interp(TD, TG, p.imag)


def spectrum(x, dt=0.05, pad=8, norm=None):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(len(x) * pad, dt); P = np.abs(np.fft.rfft(x, len(x) * pad)) ** 2
    return f, P / (P[(f > 0.3) & (f < 1.6)].max() if norm is None else norm)


def peak(f, P, lo, hi):
    m = (f > lo) & (f < hi); i = np.argmax(P[m]); return float(f[m][i]), float(P[m][i])


def design(pr, pi_, t):
    return np.column_stack([pr, pi_, np.ones_like(t), np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])


def score(pr, pi_, win_fit, win_eval=None):
    mf = (TD >= win_fit[0]) & (TD < win_fit[1])
    c = np.linalg.lstsq(design(pr[mf], pi_[mf], TD[mf]), XD[mf], rcond=None)[0]
    out = []
    for w in (win_fit, win_eval) if win_eval else (win_fit,):
        mm = (TD >= w[0]) & (TD < w[1]); y = design(pr[mm], pi_[mm], TD[mm]) @ c
        out.append(float(np.corrcoef(y, XD[mm])[0, 1]))
    return out, c


def device_xy(x, which="beta"):
    t0, a = x
    tn, k = kicks(t0, a, which)
    return on_data(response_linear(tn, k))


def fit(which="beta"):
    res = differential_evolution(lambda x: -score(*device_xy(x, which), TRAIN)[0][0],
                                 [(0.0, 0.5), (-1.0, 1.0)], seed=3, popsize=20, maxiter=120, tol=1e-9)
    return res.x


def main():
    out = {}
    e, m, te, eq = equivalence(); out["equivalence"] = eq
    out["alias_table"] = alias_table()

    fD, PD = spectrum(XD)
    obs = {"cw": peak(fD, PD, 0.7, 1.0), "sat_low": peak(fD, PD, 0.12, 0.20), "sat_high": peak(fD, PD, 1.80, 1.89),
           "annual": peak(fD, PD, 0.95, 1.05)}
    out["observed_peaks"] = obs

    xbest = fit("beta"); t0, a = xbest
    out["fit"] = {"t0_years": float(t0), "trigger_day_of_year": float(t0 * YEAR),
                  "second_trigger_day": float((t0 + 0.5) * YEAR), "asymmetry_a": float(a)}

    win = (TD >= 1935.0)
    variants = {}
    tn, kb = kicks(t0, 0.0, "beta")
    variants["continuous B^2 (no comb)"] = response_linear(None, None, continuous=field(TG, "beta") ** 2)
    variants["comb x B^2, symmetric"] = response_linear(tn, kb)
    tn, k = kicks(t0, a, "beta"); variants["comb x B^2, pear (fitted a)"] = response_linear(tn, k)
    tn, k = kicks(t0, 0.0, "decl"); variants["comb x B^2, declination field"] = response_linear(tn, k)
    tn, k = kicks(t0, 0.0, "beta", power=1); variants["comb x B, same polarity"] = response_linear(tn, k)
    tn, k = kicks(t0, 0.0, "beta", power=1, alternating=True); variants["comb x B, alternating (QBO rule)"] = response_linear(tn, k)

    # parametric variant: phase steps eps_n = eps0 * (1 +- a_p); eps0 is absorbed into the mean frequency,
    # so only the alternating part eps0 * a_p matters; it is set from the observed 1 - CW satellite power
    tn, north = comb_times(t0); _, k = kicks(t0, a, "beta")
    best = None
    for ea in np.linspace(0.0, 0.6, 31):
        pp = response_parametric(tn, k, np.where(north, ea, -ea), north)
        ff, PP = spectrum(on_data(pp)[0][win])
        sat = 0.5 * (peak(ff, PP, 0.12, 0.20)[1] + peak(ff, PP, 1.80, 1.89)[1])
        err = abs(np.log(sat / (0.5 * (obs["sat_low"][1] + obs["sat_high"][1]))))
        if best is None or err < best[0]:
            best = (err, ea, pp)
    out["parametric"] = {"alternating_phase_step_rad": float(best[1])}
    variants["comb x B^2, pear + inertia phase steps"] = best[2]
    vrows, specs = {}, {}
    f, P = spectrum(on_data(variants["comb x B^2, symmetric"])[0][win]); ref = None
    xs = on_data(variants["comb x B^2, symmetric"])[0][win]; xs = (xs - xs.mean()) * np.hanning(len(xs))
    ref = (np.abs(np.fft.rfft(xs, len(xs) * 8)) ** 2).max()   # continuous drive is scaled against the comb
    logobs = np.log10(PD[(fD > 0.05) & (fD < 2.0)] + 1e-8)
    for name, p in variants.items():
        pr, pi_ = on_data(p)
        (ctr, cho), c = score(pr, pi_, TRAIN, HOLD)
        f, P = spectrum(pr[win], norm=ref if "pure B^2" in name or "continuous" in name else None)
        main_line = peak(f, P, 0.45, 0.95)
        lo, hi = peak(f, P, 0.12, 0.20), peak(f, P, 1.80, 1.89)
        lr = np.corrcoef(np.log10(P[(f > 0.05) & (f < 2.0)] + 1e-8), logobs)[0, 1]
        vrows[name] = dict(strongest_line_0p45_0p95=main_line[0], sat_low=lo, sat_high=hi, train_r=ctr, holdout_r=cho,
                           logspec_r=float(lr))
        specs[name] = (f, P)
    out["variants"] = vrows

    # baseline: a pure sinusoid at the predicted alias line, same regression (2 + annual + const)
    fl = alias(2 * YEAR / MONTHS["draconic"], False)
    ps = np.exp(2j * np.pi * fl * TG)
    (btr, bho), _ = score(*on_data(ps), TRAIN, HOLD)
    out["baseline_sinusoid"] = {"freq": fl, "train_r": btr, "holdout_r": bho}

    # reverse split, to see whether the window choice matters
    rev_fit = differential_evolution(lambda x: -score(*device_xy(x), HOLD)[0][0], [(0.0, 0.5), (-1.0, 1.0)],
                                     seed=3, popsize=20, maxiter=120, tol=1e-9).x
    (rtr, rho), _ = score(*device_xy(rev_fit), HOLD, TRAIN)
    (sbtr, sbho), _ = score(*on_data(ps), HOLD, TRAIN)
    out["reverse_split"] = {"fit_window": HOLD, "eval_window": TRAIN, "t0_days": float(rev_fit[0] * YEAR),
                            "a": float(rev_fit[1]), "device": [rtr, rho], "sinusoid": [sbtr, sbho]}

    # nonlinear LLG agrees with the linearized propagator at realistic amplitude (0.2 arcsec ~ 1e-6 rad)
    tn, k = kicks(t0, a, "beta")
    plin = response_linear(tn, k)
    scale = 1e-6 / np.std(plin.real[TG > 1935])
    pnl = response_llg(tn, k, scale)
    mm = TG > 1935
    out["llg_check"] = {"rel_rms_diff": float(np.std(pnl.real[mm] - scale * plin.real[mm]) / np.std(scale * plin.real[mm])),
                        "rms_angle_rad": float(np.std(pnl.real[mm]))}

    # held-out time series of the fitted device
    pr, pi_ = on_data(variants["comb x B^2, pear (fitted a)"])
    _, c = score(pr, pi_, TRAIN, HOLD); ymod = design(pr, pi_, TD) @ c
    ysin = design(*on_data(ps), TD) @ score(*on_data(ps), TRAIN, HOLD)[1]

    (HERE / "cw_magnetized_spheroid.json").write_text(json.dumps(out, indent=1, default=float))
    plot_equivalence(e, m, te, eq)
    plot_spectra(fD, PD, specs, vrows, obs, ymod, ysin, out)
    report(out)


# ---------------------------------------------------------------- figures
def plot_equivalence(e, m, te, eq):
    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_axes([0.02, 0.06, 0.46, 0.86]); ax.set_xlim(-5, 6); ax.set_ylim(-4.2, 4.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) The device: pear-shaped, polar-magnetized spheroid beside a pulsed coaxial electromagnet", fontsize=10)
    tilt = 23.4
    th = np.linspace(0, 2 * np.pi, 400); r = 1.55 * (1 + 0.16 * np.cos(th))      # pear (degree-3) outline
    x, y = r * np.sin(th) * 1.08, r * np.cos(th)
    c, s = np.cos(rad(tilt)), np.sin(rad(tilt))
    ax.add_patch(Polygon(np.column_stack([c * x + s * y, -s * x + c * y]), closed=True, fc="#c9d8ea", ec="#2b4c7e", lw=1.5))
    ax.annotate("", xy=(2.7 * s, 2.7 * c), xytext=(-2.7 * s, -2.7 * c), arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#b22222"))
    ax.text(2.9 * s + 0.1, 2.9 * c, "spin = polar\nmagnetization m\n(angular momentum L)", fontsize=8, color="#b22222")
    ax.add_patch(Ellipse((2.7 * s, 2.7 * c), 1.1, 0.35, angle=-tilt, fill=False, ls="--", ec="#b22222"))
    ax.text(-2.1, 2.9, "free precession =\nKittel mode =\nChandler wobble", fontsize=8, ha="center")
    ax.text(0, -0.35, "N (shape)\n= I$^{-1}$", fontsize=8, ha="center")
    ax.text(-0.05, -2.45, "pear: north and south\npulses unequal (1 ± a)", fontsize=8, ha="center")
    for i, yy in enumerate(np.linspace(-1.6, 1.6, 9)):                        # coaxial coil, axis vertical
        ax.add_patch(Ellipse((4.6, yy), 1.3, 0.32, fill=False, ec="#8a5a00", lw=1.4))
    ax.plot([4.6, 4.6], [-2.6, 2.6], ":", color="#8a5a00")
    ax.text(4.6, 2.2, "electromagnet\nm = 0 (coaxial)", fontsize=8, ha="center", color="#8a5a00")
    tt = np.linspace(0, 1, 300)
    ax.plot(3.6 + 2.0 * tt, -3.0 + 0.3 * np.sin(2 * np.pi * 4 * tt), color="#8a5a00", lw=1)
    ax.text(4.6, -3.55, "I(t) ∝ sin β$_{moon}$: draconic\ntorque ∝ B² (sign-blind)", fontsize=8, ha="center", color="#8a5a00")
    for k in range(6):
        h = 0.8 if k % 2 == 0 else 0.55
        ax.plot([-4.6 + 0.45 * k] * 2, [-3.9, -3.9 + h], color="#2b4c7e", lw=2)
    ax.plot([-4.8, -2.0], [-3.9, -3.9], color="#2b4c7e", lw=0.8)
    ax.text(-3.4, -3.0, "inertia comb (carrier):\nshape pulsed twice a year", fontsize=8, ha="center", color="#2b4c7e")
    ax.add_patch(FancyArrowPatch((3.7, 0.3), (1.9, 0.3), arrowstyle="-|>", mutation_scale=14, color="#8a5a00"))

    ax2 = fig.add_axes([0.55, 0.56, 0.42, 0.36])
    ax2.plot(te, e.y[0] / np.linalg.norm(e.y[:, 0]), lw=2.5, color="#c9d8ea", label="Euler top: L$_x$/|L|  (I = 0.8, 0.9, 1.0)")
    ax2.plot(te, m.y[0], "--", lw=1, color="#b22222", label="Landau–Lifshitz macrospin: m$_x$  (γM$_s$N = |L| I$^{-1}$)")
    ax2.set_title(f"(b) Same equation: max |difference| = {eq['max_abs_diff']:.1e}", fontsize=10)
    ax2.legend(fontsize=7, loc="upper right"); ax2.set_xlabel("time (units of 2π/|L|·C)")
    ax3 = fig.add_axes([0.55, 0.08, 0.42, 0.33]); ax3.axis("off")
    E = eq["earth"]
    txt = ("(c) Frequencies\n"
           f"  numerical: {eq['f_numerical']:.4f}   Kittel √((Nx−Nz)(Ny−Nz))/2π: {eq['f_kittel']:.4f}   "
           f"Euler: {eq['f_euler']:.4f}\n\n"
           f"  Earth: H = (C−A)/A = {E['H_dynamical_ellipticity']}  →  Euler period {E['euler_period_days']:.1f} d;\n"
           f"  observed 432.3 d (×{E['lengthening_factor']:.3f}: elastic mantle, oceans, fluid core;\n"
           "  in the magnet, a softened effective anisotropy N$_\\perp$−N$_z$).\n"
           f"  Q = 85  ↔  Gilbert α = 1/(2Q) = {1 / (2 * Q):.4f}  (the range of permalloy).")
    ax3.text(0, 0.95, txt, fontsize=9, va="top", family="DejaVu Sans")
    fig.savefig(HERE / "cw_spheroid_equivalence.png", dpi=130); plt.close(fig)


def plot_spectra(fD, PD, specs, vrows, obs, ymod, ysin, out):
    fig, axs = plt.subplots(3, 1, figsize=(13, 13), gridspec_kw=dict(height_ratios=[1.25, 1.25, 1]))
    fl = alias(2 * YEAR / MONTHS["draconic"], False)
    marks = [(fl, "CW alias 0.844"), (1 - fl, "1 − CW"), (1 + fl, "1 + CW"), (2 - fl, "2 − CW")]
    ax = axs[0]
    ax.semilogy(fD, PD, color="k", lw=1.2, label="cw.dat (1935–2018)")
    for name, col in [("comb x B^2, symmetric", "#4a90d9"), ("comb x B^2, pear (fitted a)", "#d94a4a"),
                      ("comb x B^2, pear + inertia phase steps", "#2e8b57"), ("continuous B^2 (no comb)", "#999999")]:
        f, P = specs[name]; ax.semilogy(f, P, lw=1, color=col, label=name)
    for v, lab in marks:
        ax.axvline(v, ls=":", color="g"); ax.text(v, 2, lab, rotation=90, fontsize=7, va="bottom", ha="right", color="g")
    ax.set_xlim(0.02, 2.0); ax.set_ylim(1e-6, 30); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) Carrier × modulator: the inertia comb carries the B² drive to the CW line; the pear shape adds the "
                 "1 ∓ CW satellites", fontsize=10)
    ax.set_ylabel("power / CW peak")
    ax = axs[1]
    ax.semilogy(fD, PD, color="k", lw=1.2, label="cw.dat")
    fT = YEAR / MONTHS["tropical"]; fDr = YEAR / MONTHS["draconic"]
    for name, col, fa in [("comb x B^2, declination field", "#9b59b6", alias(2 * fT, False)),
                          ("comb x B, same polarity", "#e67e22", alias(fDr, False)),
                          ("comb x B, alternating (QBO rule)", "#16a085", alias(fDr, True))]:
        f, P = specs[name]
        ax.semilogy(f, P, lw=1, color=col, label=f"{name}: alias {fa:.3f}/yr, fit r = {vrows[name]['train_r']:.2f}")
        ax.axvline(fa, ls="--", lw=0.8, color=col)
    ax.axvline(fl, ls=":", color="g"); ax.set_xlim(0.02, 2.0); ax.set_ylim(1e-6, 30); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(b) The wrong drives: declination (tropical month), sign-sensitive B, or the QBO's alternating comb", fontsize=10)
    ax.set_ylabel("power / CW peak"); ax.set_xlabel("frequency (cycles/yr)")
    ax = axs[2]
    ax.plot(TD, XD, color="k", lw=1, label="cw.dat")
    ax.plot(TD, ymod, color="#d94a4a", lw=1, label="device (pear comb × B²), fit 1935–1990")
    ax.plot(TD, ysin, color="#4a90d9", lw=0.8, alpha=0.7, label="sinusoid at 0.8438/yr, same fit")
    ax.axvspan(*HOLD, color="0.92"); ax.text(2004, ax.get_ylim()[1] * 0.85, "held out", ha="center", fontsize=8)
    v = vrows["comb x B^2, pear (fitted a)"]; b = out["baseline_sinusoid"]
    ax.set_title(f"(c) Time series: device r = {v['train_r']:.2f} train / {v['holdout_r']:.2f} held out; "
                 f"sinusoid {b['train_r']:.2f} / {b['holdout_r']:.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="lower left"); ax.set_xlabel("year")
    fig.tight_layout(); fig.savefig(HERE / "cw_spheroid_spectra.png", dpi=120); plt.close(fig)


def report(out):
    eq = out["equivalence"]
    print(f"Euler vs LL max diff {eq['max_abs_diff']:.2e}; f num/Kittel/Euler {eq['f_numerical']:.4f} "
          f"{eq['f_kittel']:.4f} {eq['f_euler']:.4f}")
    print("alias table (line nearest zero, cycles/yr):")
    for r in out["alias_table"]:
        print(f"  {r['month']:12s} B^{r['power']} {'alternating' if r['alternating'] else 'same       '} "
              f"{r['line']:.4f}  ({YEAR / r['line'] if r['line'] > 0 else np.inf:.1f} d)")
    print("observed peaks:", {k: (round(v[0], 4), round(v[1], 5)) for k, v in out["observed_peaks"].items()})
    print("fit:", out["fit"])
    for k, v in out["variants"].items():
        print(f"  {k:36s} line {v['strongest_line_0p45_0p95']:.4f}  sat {v['sat_low'][0]:.4f}:{v['sat_low'][1]:.1e} "
              f"{v['sat_high'][0]:.4f}:{v['sat_high'][1]:.1e}  r {v['train_r']:.3f}/{v['holdout_r']:.3f}  logspec r {v['logspec_r']:.2f}")
    print("baseline sinusoid:", out["baseline_sinusoid"])
    print("reverse split:", out["reverse_split"])
    print("LLG check:", out["llg_check"])


if __name__ == "__main__":
    main()
