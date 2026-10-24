"""Chandler wobble forecast engines. Each takes binned complex polar motion p = x - i y (arcsec) on times t (decimal
years), an origin and a horizon, uses only samples at or before the origin, and returns a projection on the bin grid
after the origin. See targets.yml for what each engine represents.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
YEAR = 365.2422
F_CW = 2 * YEAR / 27.212221 - 26                       # draconic alias, 0.84398 /yr
F_FM = 1 - (YEAR / 27.212221 - YEAR / 27.554550)       # F - M on the annual harmonic, 0.83325 /yr
rad = np.radians


def fingerprint(engine: str, cfg: dict) -> str:
    h = hashlib.sha256()
    h.update((HERE / "engines.py").read_bytes())
    h.update(json.dumps({"engine": engine, "cfg": cfg}, sort_keys=True).encode())
    return h.hexdigest()


def future_grid(t, origin, horizon, step):
    """Bin centres after the origin, continuing the observation grid."""
    k0 = np.floor((origin - t[0]) / step + 1e-9)
    return t[0] + (k0 + np.arange(1, int(round(horizon / step)) + 1)) * step


def _lstsq(G, y):
    c, *_ = np.linalg.lstsq(G, y, rcond=None)
    return c


def _seasonal(t):
    return [np.exp(2j * np.pi * f * t) for f in (1.0, -1.0, 2.0, -2.0)]


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------------------------- alias_cw (gem-lte port)
def alias_cw(t, p, origin, horizon, cfg, step):
    r = 365.242 / 27.2122; pcw = 365.242 / (r - np.floor(r)) / 2.0
    m = (t <= origin) & (t > origin - cfg["train_years"])
    def X(tt):
        return np.column_stack([np.ones_like(tt), tt - origin] +
                               [np.exp(1j * 2 * np.pi * 365.2425 / P * tt) for P in (pcw, 365.242)]).astype(complex)
    c = _lstsq(X(t[m]), p[m]); g = future_grid(t, origin, horizon, step)
    return dict(grid=g, p=X(g) @ c, meta=dict(chandler_period_days=pcw, cw_amp_mas=float(abs(c[2]) * 1e3),
                                             r_train=_corr(X(t[m]) @ c, p[m])))


# --------------------------------------------------------------------------------------------- ls_free (benchmark)
def ls_free(t, p, origin, horizon, cfg, step):
    m = (t <= origin) & (t > origin - cfg["train_years"]); tt = t[m]; tc = tt - origin
    def X(x, f):
        xc = x - origin
        return np.column_stack([np.ones_like(x), xc, xc ** 2, np.exp(2j * np.pi * f * x)] + _seasonal(x)).astype(complex)
    best = None
    for f in np.linspace(cfg["f_scan"][0], cfg["f_scan"][1], 201):
        c = _lstsq(X(tt, f), p[m]); e = np.linalg.norm(p[m] - X(tt, f) @ c)
        if best is None or e < best[0]:
            best = (e, f, c)
    _, f, c = best; g = future_grid(t, origin, horizon, step)
    return dict(grid=g, p=X(g, f) @ c, meta=dict(cw_freq=float(f), cw_period_days=float(YEAR / f), cw_amp_mas=float(abs(c[3]) * 1e3),
                                                r_train=_corr(X(tt, f) @ c, p[m])))


# --------------------------------------------------------------------------------------------- free_decay (consensus)
def free_decay(t, p, origin, horizon, cfg, step):
    f0 = YEAR / cfg["period_days"]; lam = np.pi * f0 / cfg["Q"]
    m = (t <= origin) & (t > origin - cfg["state_years"]); tt = t[m]
    def X(x):
        return np.column_stack([np.ones_like(x), x - origin, np.exp(2j * np.pi * f0 * x)] + _seasonal(x)).astype(complex)
    c = _lstsq(X(tt), p[m]); g = future_grid(t, origin, horizon, step)
    Xg = X(g); Xg[:, 2] *= np.exp(-lam * (g - origin))           # the CW decays freely after the origin
    return dict(grid=g, p=Xg @ c, meta=dict(f0=f0, Q=cfg["Q"], cw_amp_mas_at_origin=float(abs(c[2]) * 1e3),
                                           r_train=_corr(X(tt) @ c, p[m])))


# --------------------------------------------------------------------------------------------- device (gated lunar)
_DEVICE_CACHE: dict = {}


def _moon(tt):
    T = (tt - 2000.0) * 365.25 / 36525.0
    D, Mp, F = (rad(v) for v in (297.8501921 + 445267.1114034 * T, 134.9633964 + 477198.8675055 * T, 93.2720950 + 483202.0175233 * T))
    beta = rad(5.128) * np.sin(F) + rad(0.281) * np.sin(Mp + F) + rad(0.278) * np.sin(Mp - F) + rad(0.173) * np.sin(2 * D - F)
    r = 385000.56 - 20905.355 * np.cos(Mp) - 3699.111 * np.cos(2 * D - Mp) - 2955.968 * np.cos(2 * D) - 569.925 * np.cos(2 * Mp)
    return (np.sin(beta) / np.sin(rad(5.145))) ** 2 * (385000.56 / r) ** 3


def _device_responses(cfg, t_end):
    """The 183 comb-day responses, computed once per settings to a fixed end and kept on a 0.01-yr grid (~60 MB).
    The responses are causal, so one set serves every origin; keying on the origin held ~280 MB per origin."""
    key = json.dumps(cfg, sort_keys=True)
    if key in _DEVICE_CACHE and _DEVICE_CACHE[key][0][-1] >= t_end:
        return _DEVICE_CACHE[key]
    _DEVICE_CACHE.clear()
    t_end = max(t_end, np.ceil(t_end / 10.0) * 10.0)              # one computation covers the decade
    dt = 0.002; tg = np.arange(1820.0, t_end + 1.0, dt)
    tau = cfg["tau_days"] / YEAR; s = np.arange(-6 * tau, 6 * tau + dt, dt)
    g = np.exp(-0.5 * (s / tau) ** 2); g /= g.sum(); dg = -s / tau ** 2 * g
    pulse = (1 - cfg["push_pull"]) * g + cfg["push_pull"] * dg / (2 * np.pi * 14); lag = len(s) // 2
    lam = cfg["f0"] / (2 * cfg["Q"])
    h = np.exp((2j * np.pi * cfg["f0"] - 2 * np.pi * lam) * np.arange(0, 8 * cfg["Q"] / cfg["f0"], dt)) * dt
    lunar = _moon(tg); out = {}
    years = np.arange(np.floor(tg[0]), np.ceil(tg[-1]))
    for d in range(0, 183):
        t0 = d / YEAR
        tn = np.sort(np.concatenate([years + t0, years + t0 + 0.5])); tn = tn[(tn > tg[0]) & (tn < tg[-1])]
        north = np.isclose((tn - t0) % 1.0, 0.0, atol=1e-6) | np.isclose((tn - t0) % 1.0, 1.0, atol=1e-6)
        imp = np.zeros(len(tg)); np.add.at(imp, np.round((tn - tg[0]) / dt).astype(int), np.where(north, 1 + cfg["asymmetry"], 1 - cfg["asymmetry"]))
        gate = fftconvolve(imp, pulse)[lag: lag + len(tg)]
        out[d] = fftconvolve(lunar * gate, h)[: len(tg)][::5].astype(np.complex64)
    _DEVICE_CACHE[key] = (tg[::5], out)
    return _DEVICE_CACHE[key]


def device(t, p, origin, horizon, cfg, step):
    g = future_grid(t, origin, horizon, step)
    tg, resp = _device_responses({k: cfg[k] for k in ("f0", "Q", "tau_days", "push_pull", "asymmetry")}, g[-1])
    m = (t <= origin) & (t > origin - cfg["train_years"]); tt = t[m]
    def X(x, z):
        xc = x - origin
        return np.column_stack([z, np.ones_like(x), xc, xc ** 2] + _seasonal(x)).astype(complex)
    def at(z, x):
        return np.interp(x, tg, z.real) + 1j * np.interp(x, tg, z.imag)
    best = None
    for d, z in resp.items():
        zt = at(z, tt); c = _lstsq(X(tt, zt), p[m]); e = np.linalg.norm(p[m] - X(tt, zt) @ c)
        if best is None or e < best[0]:
            best = (e, d, c, zt)
    _, d, c, zt = best
    return dict(grid=g, p=X(g, at(resp[d], g)) @ c, meta=dict(comb_day=int(d), r_train=_corr(X(tt, zt) @ c, p[m]), **cfg))


# --------------------------------------------------------------------------------------------- two_line_fm
def _hats(x, knots):
    """Piecewise-linear (hat) basis on the knots; constant beyond the end knots."""
    cols = []
    for i, k in enumerate(knots):
        h = np.zeros_like(x)
        if i > 0:
            a = knots[i - 1]; mm = (x >= a) & (x <= k); h[mm] = (x[mm] - a) / (k - a)
        else:
            h[x <= k] = 1.0
        if i < len(knots) - 1:
            b = knots[i + 1]; mm = (x >= k) & (x <= b); h[mm] = (b - x[mm]) / (b - k)
        else:
            h[x >= k] = 1.0
        cols.append(h)
    return cols


def two_line_fm(t, p, origin, horizon, cfg, step):
    """Steady forced lines at F_CW and F_FM over the long record; seasonal lines per segment; piecewise-linear drift.
    Projection: both lines, the last segment's seasonal lines, and the drift continued at its last-interval slope."""
    m = (t >= cfg["train_start"]) & (t <= origin); tt = t[m]
    edges = np.arange(origin, cfg["train_start"] - 1e-9, -cfg["season_segment_years"])[::-1]
    if edges[0] > cfg["train_start"] + 1e-9:
        edges = np.r_[cfg["train_start"], edges]
    knots = np.arange(origin, cfg["train_start"] - 1e-9, -cfg["drift_knot_years"])[::-1]
    if knots[0] > cfg["train_start"] + 1e-9:
        knots = np.r_[cfg["train_start"], knots]
    nseg = len(edges) - 1

    def parts(x):
        seg = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nseg - 1)
        lines = [np.exp(2j * np.pi * F_CW * x), np.exp(2j * np.pi * F_FM * x)]
        seas = [(seg == s_).astype(float) * e for s_ in range(nseg) for e in _seasonal(x)]
        drift = _hats(np.minimum(x, origin), list(knots))
        return lines, seas, drift

    def X(x):
        a, b, c_ = parts(x)
        return np.column_stack(a + b + c_).astype(complex)

    c = _lstsq(X(tt), p[m]); fit = X(tt) @ c
    nd = len(knots); dco = c[-nd:]                                   # drift coefficients at the knots
    slope = (dco[-1] - dco[-2]) / (knots[-1] - knots[-2])
    g = future_grid(t, origin, horizon, step)
    proj = X(g) @ c + slope * (g - origin)                           # hats hold the origin value; add the slope
    return dict(grid=g, p=proj, meta=dict(cw_line_mas=float(abs(c[0]) * 1e3), fm_line_mas=float(abs(c[1]) * 1e3),
                                         beat_period_years=float(1 / abs(F_CW - F_FM)), r_train=_corr(fit, p[m]),
                                         train_start=cfg["train_start"], n_segments=int(nseg), n_knots=int(nd)))


ENGINES = dict(alias_cw=alias_cw, ls_free=ls_free, free_decay=free_decay, device=device, two_line_fm=two_line_fm)
