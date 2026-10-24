#!/usr/bin/env python3
"""The post-2010 collapse of the Chandler wobble read as a counter-forcing episode (2026-10-24).
Discussion: cw_suppression.md. Needs data/finals2000A.all (fetch_eop.py).

1. Envelope: raw complex polar motion with trend, annual, semiannual and Mm removed, demodulated at the CW alias
   (0.8440 /yr) and smoothed (Gaussian, sigma 1 yr): amplitude and phase through the collapse.
2. Required forcing: in the frame rotating at the alias, dz/dt = a z + F(t), a = i 2 pi (f0 - f_CW) - pi f0 / Q. F is
   fitted as constant over epochs (exact solution of the ODE, linear least squares with the initial state). Compared:
   constant F; F switched off in 2010 (free decay); F per epoch. For f0, Q from cw_bandpass.py and the consensus.
3. Mm across the episode: amplitude and phase in windows; a comb-timing shift dt rotates the CW alias forcing by
   360 * 26 * dt and the Mm forcing by 360 * 14 * dt (deg, dt in yr).
4. Steady-line alternatives: the CW line plus one more steady forced line, as a beat instead of an episode.
   2F+M on the odd harmonic 41 (0.9008 /yr) and F-M on the odd harmonic 1 (0.8332 /yr) both need an annual impulse
   (odd comb harmonics exist only with unequal semiannual pulses). Also a free scan of the second line, the beat
   minima it implies, and an out-of-sample check of the F-M beat against the cw.dat envelope, 1938-1972.
5. The 6-year cycle in length of day: LOD (finals2000A columns 80-86) with trend, seasonal and 18.6/9.3-yr terms
   removed, band-passed to 4.5-8.5 yr, and regressed in windows on the astronomical perigee-node angle F - M'
   (Meeus): a lunar F - M origin keeps a fixed phase relative to that angle. Significance: annual-mean LOD
   periodogram against a power-law background and AR(1)/AR(2) Monte Carlo; and the same band-pass + regression on
   AR(2) noise, to see how often noise mimics the "lock".
    python3 cw_suppression.py  -> cw_suppression.png, cw_suppression.json
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

import cw_eop_closure as E

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
YEAR = 365.2422
F_CW, F_MM = 2 * YEAR / 27.212221 - 26, 14 - YEAR / 27.554550
EPOCHS = [1973, 1990, 2005, 2010, 2013, 2016, 2019, 2027]
CASES = [(YEAR / 432.3, 85.0, "consensus 432 d, Q 85"), (0.835, 63.5, "fitted f0 0.835, Q 64"),
         (0.8432, 31.6, "fitted f0 0.843, Q 32"), (0.8432, 20.0, "f0 0.843, Q 20")]


def rotating_residual():
    tb, xb, yb = E.load_finals(); p = xb - 1j * yb
    tc = (tb - tb.mean()) / 50
    G = np.column_stack([tc ** k for k in range(3)] + [np.exp(2j * np.pi * f * tb) for f in (1.0, -1.0, 2.0, -2.0, F_MM)])
    r = (p - G @ np.linalg.lstsq(G, p, rcond=None)[0]) * 1e3            # mas
    return tb, p, r * np.exp(-2j * np.pi * F_CW * tb)


def smooth(t, w, s):
    k = np.arange(-int(4 * s / 0.05), int(4 * s / 0.05) + 1) * 0.05; g = np.exp(-0.5 * (k / s) ** 2); g /= g.sum()
    return np.convolve(w, g, "same"), (t > t[0] + 3 * s) & (t < t[-1] - 3 * s)


def design(t, f0, q, edges):
    a = 1j * 2 * np.pi * (f0 - F_CW) - np.pi * f0 / q
    cols = [np.exp(a * (t - t[0]))]
    for e0, e1 in zip(edges[:-1], edges[1:]):
        on = np.where(t > e0, (np.exp(a * (t - e0)) - 1) / a, 0)
        off = np.where(t > e1, (np.exp(a * (t - e1)) - 1) / a, 0)
        cols.append(on - off)
    return np.column_stack(cols)


def fit(t, z, m, f0, q, edges):
    D = design(t, f0, q, edges); c, *_ = np.linalg.lstsq(D[m], z[m], rcond=None)
    return c, D @ c, float(np.sqrt(np.mean(np.abs(z[m] - D[m] @ c) ** 2)))


Y = YEAR
F_2FpM = 41 - (2 * Y / 27.212221 + Y / 27.554550)        # 2F+M on harmonic 41
F_FmM = 1 - (Y / 27.212221 - Y / 27.554550)              # F-M on harmonic 1


def two_line(t, f0, q, f2, t0=1973.0):
    a = 1j * 2 * np.pi * (f0 - F_CW) - np.pi * f0 / q; b = 1j * 2 * np.pi * (f2 - F_CW)
    return np.column_stack([np.exp(a * (t - t0)), (np.exp(a * (t - t0)) - 1) / a,
                            (np.exp(b * (t - t0)) - np.exp(a * (t - t0))) / (b - a)]), a, b


def cwdat_envelope():
    cw = np.loadtxt(HERE.parent / "cw.dat"); td, xd = cw[:, 0], cw[:, 1]
    tc = (td - td.mean()) / 50
    G = np.column_stack([tc ** k for k in range(3)] + [f(2 * np.pi * fr * td) for fr in (1.0, 2.0, F_MM) for f in (np.cos, np.sin)])
    r = (xd - G @ np.linalg.lstsq(G, xd, rcond=None)[0]) / (-0.334) * 1e3      # cw.dat = -0.334 x over 1973-2018
    env, ok = smooth(td, r * np.exp(-2j * np.pi * F_CW * td), 1.0)
    return td, 2 * np.abs(env), ok                          # a real cosine carries half the circular amplitude


def beats(t, z, m):
    td, envc, okc = cwdat_envelope(); out = {}
    for f0, q, label in CASES[:3] + [(0.8432, 50.0, "f0 0.843, Q 50 (Yamaguchi & Furuya)")]:
        row = {}
        for name, f2 in (("2F+M (0.9008)", F_2FpM), ("F-M (0.8332)", F_FmM)):
            D, a, b = two_line(t, f0, q, f2); c, *_ = np.linalg.lstsq(D[m], z[m], rcond=None)
            rms = float(np.sqrt(np.mean(np.abs(z[m] - D[m] @ c) ** 2)))
            steady = lambda tt: np.abs(-c[1] / a + c[2] / (b - a) * np.exp(b * (tt - 1973)))
            tt = np.arange(1880, 2120, 0.1); e = steady(tt)
            mins = [float(round(tt[i])) for i in range(1, len(tt) - 1) if e[i] < e[i - 1] and e[i] < e[i + 1]]
            mo = (td >= 1938) & (td < 1972.5)
            tt2 = np.arange(1880, 2060, 0.05); zz = -c[1] / a + c[2] / (b - a) * np.exp(b * (tt2 - 1973))
            ph = np.degrees(np.unwrap(np.angle(zz))); jumps = []
            for y0, y1 in ((1905, 1945), (1995, 2040)):
                mm = (tt2 >= y0) & (tt2 < y1); i = np.where(mm)[0][np.argmin(np.abs(zz[mm]))]; k = int(6 / 0.05)
                jumps.append([float(round(tt2[i])), float(ph[i + k] - ph[i - k])])
            row[name] = dict(rms=rms, cw_line_mas=float(abs(c[1] / a)), second_line_mas=float(abs(c[2] / (b - a))), beat_minima=mins,
                             phase_jump_at_minima=jumps,
                             oos_r_1938_1972=float(np.corrcoef(steady(td[mo]), envc[mo])[0, 1]),
                             oos_samples={str(y): [float(steady(np.array([y]))[0]), float(envc[np.argmin(abs(td - y))])] for y in (1940, 1950, 1960, 1967)})
        best = None
        for f2 in np.linspace(0.70, 1.00, 301):
            D, a, b = two_line(t, f0, q, f2); c, *_ = np.linalg.lstsq(D[m], z[m], rcond=None)
            r = float(np.sqrt(np.mean(np.abs(z[m] - D[m] @ c) ** 2)))
            if best is None or r < best[0]:
                best = (r, float(f2))
        row["best second line"] = dict(rms=best[0], f=best[1])
        out[label] = row
    return out


def lod_six_year():
    t, l = [], []
    for line in E.FINALS.read_text().splitlines():
        if len(line) < 86 or line[57] != "I" or not line[79:86].strip():
            continue
        t.append(2000.0 + (float(line[7:15]) - 51544.5) / 365.25); l.append(float(line[79:86]))
    t, l = np.array(t), np.array(l)
    e = np.arange(np.floor(t[0] / 0.05) * 0.05, t[-1] + 0.05, 0.05); k = np.digitize(t, e) - 1
    n = np.bincount(k, minlength=len(e)); ok = (n >= 5)[:-1]
    tb = (e[:-1] + 0.025)[ok]; lb = (np.bincount(k, l, len(e)) / np.maximum(n, 1))[:-1][ok]
    tc = tb - tb.mean()
    G = np.column_stack([np.ones_like(tc), tc] + [f(2 * np.pi * fr * tb) for fr in (1.0, 2.0, 1 / 18.613, 2 / 18.613) for f in (np.cos, np.sin)])
    r = lb - G @ np.linalg.lstsq(G, lb, rcond=None)[0]
    x = np.concatenate([r[::-1], r, r[::-1]]); X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 0.05)
    X[(f < 1 / 8.5) | (f > 1 / 4.5)] = 0; bp = np.fft.irfft(X, len(x))[len(r): 2 * len(r)]
    T = (tb - 2000) * 365.25 / 36525
    th = np.radians((93.2720950 + 483202.0175233 * T) - (134.9633964 + 477198.8675055 * T))   # F - M'
    zc = tb[1:][np.diff(np.sign(bp)) > 0]
    out = dict(rms_ms=float(bp.std()), mean_period_zero_crossings=float(np.mean(np.diff(zc))), windows=[])
    for a, b in ((1973, 1991), (1991, 2009), (2009, 2027), (1973, 2027)):
        m = (tb >= a + 1) & (tb < b - 1); D = np.column_stack([np.cos(th[m]), np.sin(th[m])])
        c = np.linalg.lstsq(D, bp[m], rcond=None)[0]
        out["windows"].append(dict(window=[a, b], amp_ms=float(np.hypot(*c)), phase_rel_FM_deg=float(np.degrees(np.arctan2(-c[1], c[0]))),
                                   r=float(np.corrcoef(bp[m], D @ c)[0, 1])))
    return out


def seven_sixths(tb, p):
    """F - M (~1/6 /yr) folded by the annual comb also lands at 1 + (F - M) = 7/6 /yr; compare with 2 - CW (1.156)."""
    fFM = YEAR / 27.212221 - YEAR / 27.554550; out = {}
    for name, f in (("1+(F-M)", 1 + fFM), ("2-CW", 2 - F_CW)):
        rows = []
        for a, b in ((1973, 1986.5), (1986.5, 2000), (2000, 2013.5), (2013.5, 2027)):
            m = (tb >= a) & (tb < b); c, _, _ = E.fit_lines(tb[m], p[m], [f, F_CW, F_MM, 1.0, -1.0, 2.0])
            rows.append([float(abs(c[f]) * 1e3), float(np.degrees(np.angle(c[f])))])
        ph = np.radians([r[1] for r in rows]); mean = np.angle(np.mean(np.exp(1j * ph)))
        out[name] = dict(f=float(f), windows=rows, spread_deg=float(np.degrees(np.max(np.abs(np.angle(np.exp(1j * (ph - mean))))))))
    grid = np.linspace(1.14, 1.19, 101)
    out["best_f_1.14_1.19"] = float(grid[int(np.argmax([abs(E.fit_lines(tb, p, [g, F_CW, F_MM, 1.0, -1.0, 2.0])[0][g]) for g in grid]))])
    return out


def lod_significance(n_mc=1000, seed=0):
    t, l = [], []
    for line in E.FINALS.read_text().splitlines():
        if len(line) < 86 or line[57] != "I" or not line[79:86].strip():
            continue
        t.append(2000.0 + (float(line[7:15]) - 51544.5) / 365.25); l.append(float(line[79:86]))
    t, l = np.array(t), np.array(l); rng = np.random.default_rng(seed)
    yrs = np.arange(1973, 2026); ly = np.array([l[(t >= y) & (t < y + 1)].mean() for y in yrs]); ty = yrs + 0.5
    tc = (ty - ty.mean()) / 26
    G = np.column_stack([tc ** k for k in range(4)] + [np.cos(2 * np.pi * ty / 18.613), np.sin(2 * np.pi * ty / 18.613)])
    r = ly - G @ np.linalg.lstsq(G, ly, rcond=None)[0]

    def pgram(x):
        x = (x - x.mean()) * np.hanning(len(x)); n = len(x) * 8
        return np.fft.rfftfreq(n, 1.0), np.abs(np.fft.rfft(x, n)) ** 2
    f, P = pgram(r); band = (f >= 1 / 8.5) & (f <= 1 / 4.5); fit = (f > 0.06) & (f < 0.5) & ~band
    fp = f[band][np.argmax(P[band])]; c = np.polyfit(np.log(f[fit]), np.log(P[fit]), 1)
    ratio = P[band].max() / np.exp(np.polyval(c, np.log(fp))); nind = int(0.44 * len(r))
    out = dict(band_peak_period=float(1 / fp), peak_over_background=float(ratio), p_single=float(np.exp(-ratio)),
               p_global=float(1 - (1 - np.exp(-ratio)) ** nind),
               power_6yr_rel_strongest_3_15yr=float(P[np.argmin(abs(f - 1 / 5.97))] / P[(f > 1 / 15) & (f < 1 / 3)].max()))
    for p in (1, 2):
        X = np.column_stack([r[p - k - 1: len(r) - k - 1] for k in range(p)]); a = np.linalg.lstsq(X, r[p:], rcond=None)[0]
        sd = np.std(r[p:] - X @ a); ex = 0
        for _ in range(n_mc):
            x = np.zeros(len(r) + 200); e = rng.normal(0, sd, len(x))
            for i in range(p, len(x)):
                x[i] = sum(a[k] * x[i - k - 1] for k in range(p)) + e[i]
            x = x[200:]; x = x - G @ np.linalg.lstsq(G, x, rcond=None)[0]; _, Ps = pgram(x); ex += Ps[band].max() >= P[band].max()
        out[f"p_AR{p}_band_peak"] = ex / n_mc
    return out


def lock_null(obs_windows, n_mc=1000, seed=1):
    """Band-passed AR(2) noise, regressed on the F - M angle in the same windows: how often does noise 'lock'?"""
    t, l = [], []
    for line in E.FINALS.read_text().splitlines():
        if len(line) < 86 or line[57] != "I" or not line[79:86].strip():
            continue
        t.append(2000.0 + (float(line[7:15]) - 51544.5) / 365.25); l.append(float(line[79:86]))
    t, l = np.array(t), np.array(l)
    e = np.arange(np.floor(t[0] / 0.05) * 0.05, t[-1] + 0.05, 0.05); k = np.digitize(t, e) - 1
    n = np.bincount(k, minlength=len(e)); ok = (n >= 5)[:-1]
    tb = (e[:-1] + 0.025)[ok]; lb = (np.bincount(k, l, len(e)) / np.maximum(n, 1))[:-1][ok]; tc = tb - tb.mean()
    G = np.column_stack([np.ones_like(tc), tc] + [f(2 * np.pi * fr * tb) for fr in (1.0, 2.0, 1 / 18.613, 2 / 18.613) for f in (np.cos, np.sin)])
    r = lb - G @ np.linalg.lstsq(G, lb, rcond=None)[0]
    X = np.column_stack([r[1:-1], r[:-2]]); a = np.linalg.lstsq(X, r[2:], rcond=None)[0]; sd = np.std(r[2:] - X @ a)
    T = (tb - 2000) * 365.25 / 36525
    th = np.radians((93.2720950 + 483202.0175233 * T) - (134.9633964 + 477198.8675055 * T))
    rng = np.random.default_rng(seed); rs, close = [], 0
    minr = min(w["r"] for w in obs_windows[:3])
    d_obs = abs(((obs_windows[0]["phase_rel_FM_deg"] - obs_windows[1]["phase_rel_FM_deg"]) + 180) % 360 - 180)
    for _ in range(n_mc):
        x = np.zeros(len(r) + 400); ee = rng.normal(0, sd, len(x))
        for i in range(2, len(x)):
            x[i] = a[0] * x[i - 1] + a[1] * x[i - 2] + ee[i]
        x = x[400:]; x = x - G @ np.linalg.lstsq(G, x, rcond=None)[0]
        z = np.concatenate([x[::-1], x, x[::-1]]); F = np.fft.rfft(z); f = np.fft.rfftfreq(len(z), 0.05)
        F[(f < 1 / 8.5) | (f > 1 / 4.5)] = 0; bp = np.fft.irfft(F, len(z))[len(x): 2 * len(x)]
        rr, ph = [], []
        for a_, b_ in ((1973, 1991), (1991, 2009), (2009, 2027)):
            m = (tb >= a_ + 1) & (tb < b_ - 1); D = np.column_stack([np.cos(th[m]), np.sin(th[m])]); c = np.linalg.lstsq(D, bp[m], rcond=None)[0]
            rr.append(np.corrcoef(bp[m], D @ c)[0, 1]); ph.append(np.degrees(np.arctan2(-c[1], c[0])))
        rs.append(rr); close += abs(((ph[0] - ph[1]) + 180) % 360 - 180) <= d_obs
    rs = np.array(rs)
    return dict(median_window_r_noise=[float(v) for v in np.median(rs, 0)], p_all_r_ge_obs=float(np.mean(rs.min(1) >= minr)),
                p_two_windows_within_obs_phase=float(close / n_mc), obs_phase_diff=float(d_obs))


def mm_windows(tb, p):
    out = []
    for a, b in ((1973, 1990), (1990, 2005), (1996, 2010), (2005, 2015), (2010, 2020), (2015, 2027), (2016, 2027)):
        m = (tb >= a) & (tb < b); c, _, _ = E.fit_lines(tb[m], p[m], [F_MM, F_CW, 1.0, -1.0, 2.0, -2.0])
        out.append(dict(window=[a, b], mm_mas=float(abs(c[F_MM]) * 1e3), mm_phase=float(np.degrees(np.angle(c[F_MM]))),
                        cw_mas=float(abs(c[F_CW]) * 1e3)))
    return out


def main():
    t, p, w = rotating_residual()
    env, ok = smooth(t, w, 1.0)
    z, m = smooth(t, w, 0.7)
    out = {"envelope": [dict(year=y, amp_mas=float(abs(env[np.argmin(abs(t - y))])),
                             phase_deg=float(np.degrees(np.angle(env[np.argmin(abs(t - y))] / env[np.argmin(abs(t - 1990))]))))
                        for y in range(1978, 2023) if ok[np.argmin(abs(t - y))]], "cases": {}}
    # free decay check: e-fold time of the 2010 -> 2014 drop, and the Q it would need without forcing
    a10, a14 = abs(env[np.argmin(abs(t - 2010))]), abs(env[np.argmin(abs(t - 2014))])
    tau = 4 / np.log(a10 / a14); out["drop_2010_2014"] = dict(from_mas=float(a10), to_mas=float(a14), efold_yr=float(tau),
                                                            free_decay_Q=float(np.pi * F_CW * tau))
    recon = {}
    for f0, q, label in CASES:
        c0, r0, e0 = fit(t, z, m, f0, q, [1973, 2027])
        c1, r1, e1 = fit(t, z, m, f0, q, [1973, 2010])                  # forcing off from 2010: free decay
        c2, r2, e2 = fit(t, z, m, f0, q, EPOCHS)
        F = c2[1:]
        out["cases"][label] = dict(f0=f0, Q=q, free_efold_yr=float(q / (np.pi * f0)), rms_constant=e0, rms_switch_off=e1,
                                   rms_epochs=e2,
                                   forcing=[dict(epoch=[e_0, e_1], rel_amp=float(abs(f / F[0])), rel_phase=float(np.degrees(np.angle(f / F[0]))))
                                            for e_0, e_1, f in zip(EPOCHS[:-1], EPOCHS[1:], F)])
        recon[label] = (r0, r1, r2)
    out["mm_windows"] = mm_windows(t, p)
    out["comb_shift_7d"] = dict(cw_deg=360 * 26 * 7 / YEAR, mm_deg=360 * 14 * 7 / YEAR, qbo_deg=360 * 13 * 7 / YEAR)
    out["two_line"] = beats(t, z, m)
    out["lod_six_year"] = lod_six_year()
    out["seven_sixths"] = seven_sixths(t, p)
    out["lod_significance"] = lod_significance()
    out["lod_lock_null"] = lock_null(out["lod_six_year"]["windows"])
    fFM = YEAR / 27.212221 - YEAR / 27.554550
    out["six_year_arithmetic"] = dict(f_FM=fFM, period_FM=1 / fFM, f_annual_minus_cw=1 - F_CW, period_beat=1 / (1 - F_CW),
                                      envelope_period=1 / (fFM - (1 - F_CW)))
    (HERE / "cw_suppression.json").write_text(json.dumps(out, indent=1))
    report(out)
    plot(t, env, ok, z, m, recon, out)


def report(out):
    d = out["drop_2010_2014"]
    print(f"2010 -> 2014: {d['from_mas']:.0f} -> {d['to_mas']:.0f} mas, e-fold {d['efold_yr']:.1f} yr, free-decay Q {d['free_decay_Q']:.0f}")
    for k, v in out["cases"].items():
        print(f"{k}: free e-fold {v['free_efold_yr']:.1f} yr; rms constant {v['rms_constant']:.1f}, switch-off {v['rms_switch_off']:.1f}, "
              f"epochs {v['rms_epochs']:.1f} mas")
        print("   ", [(f"{e['epoch'][0]}-{e['epoch'][1]}", round(e['rel_amp'], 2), round(e['rel_phase'])) for e in v["forcing"]])
    print("Mm windows:", [(f"{w['window'][0]}-{w['window'][1]}", round(w['mm_mas'], 1), round(w['mm_phase']), round(w['cw_mas'])) for w in out["mm_windows"]])
    print("7-day comb shift rotates CW forcing by %.0f deg, Mm by %.0f deg, QBO (F on harmonic 13) by %.0f deg"
          % (out["comb_shift_7d"]["cw_deg"], out["comb_shift_7d"]["mm_deg"], out["comb_shift_7d"]["qbo_deg"]))
    a6 = out["six_year_arithmetic"]; L6 = out["lod_six_year"]
    print(f"F-M {a6['period_FM']:.3f} yr vs annual-CW beat {a6['period_beat']:.2f} yr -> envelope {a6['envelope_period']:.1f} yr")
    print(f"LOD 6-yr band rms {L6['rms_ms']:.3f} ms, mean period {L6['mean_period_zero_crossings']:.2f} yr;",
          [(f"{w['window'][0]}-{w['window'][1]}", round(w['amp_ms'], 3), round(w['phase_rel_FM_deg']), round(w['r'], 2)) for w in L6["windows"]])
    print("LOD 6-yr significance:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["lod_significance"].items()})
    print("LOD lock null:", out["lod_lock_null"])
    ss = out["seven_sixths"]
    print("7/6 test:", {k: (f"spread {v['spread_deg']:.0f}", [round(r[0], 1) for r in v["windows"]]) for k, v in ss.items() if k != "best_f_1.14_1.19"},
          "best", round(ss["best_f_1.14_1.19"], 4))
    for k, v in out["two_line"].items():
        print(f"two-line, {k}: best second line {v['best second line']['f']:.4f} rms {v['best second line']['rms']:.1f};",
              {n: f"rms {d['rms']:.1f}, CW {d['cw_line_mas']:.0f} / 2nd {d['second_line_mas']:.0f} mas, minima {d['beat_minima']}, "
                  f"out-of-sample r {d['oos_r_1938_1972']:.2f}" for n, d in v.items() if n != "best second line"})


def plot(t, env, ok, z, m, recon, out):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0.06, 0.58, 0.55, 0.36])
    ax.plot(t[m], np.abs(z[m]), color="k", lw=1.4, label="observed CW envelope (raw IERS, demodulated at 0.8440 /yr)")
    lab = "fitted f0 0.843, Q 32"
    r0, r1, r2 = recon[lab]
    ax.plot(t[m], np.abs(r0[m]), color="0.6", lw=1, label="constant forcing")
    ax.plot(t[m], np.abs(r1[m]), color="#4a90d9", lw=1, ls="--", label="forcing switched off in 2010 (free decay, Q 32)")
    ax.plot(t[m], np.abs(r2[m]), color="#d94a4a", lw=1.2, label="forcing per epoch (counter-forcing 2010–16)")
    for e in EPOCHS[1:-1]:
        ax.axvline(e, color="0.85", lw=0.8)
    ax.set_xlim(1975, 2025); ax.set_ylabel("amplitude (mas)"); ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("(a) The collapse: too fast for free decay; reproduced by an opposing forcing episode", fontsize=10)

    ax = fig.add_axes([0.68, 0.58, 0.29, 0.36])
    ax.plot(t[ok], np.degrees(np.unwrap(np.angle(env[ok] / env[np.argmin(abs(t - 1990))]))), color="k")
    ax.axvspan(2010, 2016, color="#f6d5d5"); ax.set_xlim(1975, 2025); ax.set_ylabel("CW phase rel. 1990 (deg)")
    ax.set_title("(b) Phase excursion at the minimum", fontsize=10)

    ax = fig.add_axes([0.06, 0.08, 0.55, 0.38])
    cols = {"consensus 432 d, Q 85": "0.5", "fitted f0 0.835, Q 64": "#e67e22", "fitted f0 0.843, Q 32": "#d94a4a", "f0 0.843, Q 20": "#8e44ad"}
    for k, v in out["cases"].items():
        x = [0.5 * (e["epoch"][0] + e["epoch"][1]) for e in v["forcing"]]
        ph = [e["rel_phase"] for e in v["forcing"]]; amp = [e["rel_amp"] for e in v["forcing"]]
        ax.scatter(x, ph, s=[40 + 60 * min(a, 2.5) for a in amp], color=cols[k], label=f"{k} (size ∝ |F|)", alpha=0.8)
    ax.axhspan(150, 210, color="#f6d5d5"); ax.axhspan(-210, -150, color="#f6d5d5")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-200, 200); ax.set_xlim(1975, 2025)
    ax.set_ylabel("forcing phase rel. 1973–90 (deg)"); ax.set_xlabel("epoch midpoint"); ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(c) Required forcing at the CW alias, per epoch: rotated ~180–240° from 2005–10 to 2010–16, back near 0° after 2019", fontsize=9)

    ax = fig.add_axes([0.68, 0.08, 0.29, 0.38])
    for w in out["mm_windows"]:
        x = 0.5 * (w["window"][0] + w["window"][1])
        ax.errorbar(x, w["mm_phase"], xerr=0.5 * (w["window"][1] - w["window"][0]), fmt="o", color="#8e44ad", capsize=2)
    base = np.mean([w["mm_phase"] for w in out["mm_windows"] if w["window"][1] <= 2010])
    ax.axhline(base, color="k", lw=0.8); ax.text(1976, base + 4, "before 2010", fontsize=7)
    for sgn in (-1, 1):
        ax.axhline(base + sgn * out["comb_shift_7d"]["mm_deg"], color="#8e44ad", ls="--", lw=0.8)
    ax.text(1976, base - out["comb_shift_7d"]["mm_deg"] + 4, "± 7-day comb shift", fontsize=7, color="#8e44ad")
    ax.set_xlim(1975, 2030); ax.set_ylim(-60, 230); ax.set_ylabel("Mm phase (deg)")
    ax.set_title("(d) Mm phase by window, against the shift a\n7-day comb change would cause (dashed)", fontsize=10)
    fig.savefig(HERE / "cw_suppression.png", dpi=115); plt.close(fig)


if __name__ == "__main__":
    main()
