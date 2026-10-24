#!/usr/bin/env python3
"""Is the annual amplitude modulation of the Chandler wobble in the Earth, or in how cw.dat was made? (2026-10-23)
Discussion: cw_data_tests.md, section 2.

Data: IERS Rapid Service daily polar motion (USNO finals2000A.all, measured rows only, 1973-now; fetch_eop.py),
averaged into 0.05-yr bins like cw.dat.

Two tests of the 1 -/+ CW sidebands:
  phase closure  D = phi(f+1) + phi(f-1) - 2 phi(f): 0 for amplitude modulation, 180 deg for phase modulation;
                 run on x and y separately (as for cw.dat) and on complex polar motion p = x - i y (CW prograde)
  sense test     in p, a modulation of the wobble itself rescales a circular motion, so its sidebands are one-sided:
                 +(f+1) = +1.843 and +(f-1) = -0.157 cycles/yr only. A modulation (or processing step) acting on one
                 component alone is a real-signal product and puts equal power at the mirror lines -1.843 and +0.157.
                 R = mirror-sense / body-sense sideband power: 0 for the body, 1 for a single-component artifact.
    python3 cw_eop_closure.py  -> cw_eop_closure.png, cw_eop_closure.json
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FINALS = HERE / "data" / "finals2000A.all"
YEAR = 365.2422
T_REF = 2000.0
DT = 0.05
WINDOWS = {"1973-2026": (1973.0, 2026.9), "1973-1990": (1973.0, 1990.0), "1990-2026": (1990.0, 2026.9),
           "1973-2018.6 (cw.dat overlap)": (1973.0, 2018.6)}


def load_finals():
    t, x, y = [], [], []
    for line in FINALS.read_text().splitlines():
        if len(line) < 46 or line[16] != "I":
            continue
        mjd = float(line[7:15]); t.append(2000.0 + (mjd - 51544.5) / 365.25)
        x.append(float(line[18:27])); y.append(float(line[37:46]))
    t, x, y = map(np.array, (t, x, y))
    edges = np.arange(np.floor(t[0] / DT) * DT, t[-1] + DT, DT)
    k = np.digitize(t, edges) - 1
    n = np.bincount(k, minlength=len(edges)); ok = n >= 5
    tb = (edges[:-1] + DT / 2)[ok[:-1]]
    xb = (np.bincount(k, x, len(edges)) / np.maximum(n, 1))[:-1][ok[:-1]]
    yb = (np.bincount(k, y, len(edges)) / np.maximum(n, 1))[:-1][ok[:-1]]
    return tb, xb, yb


def fit_lines(t, z, freqs, poly=2):
    """Least squares z ~ poly(t) + sum c_nu exp(i 2 pi nu (t - T_REF)); complex z, signed frequencies.
    For real z, pass pairs of cos/sin by fitting the real and imaginary parts of c separately (real=True)."""
    tc = (t - t.mean()) / 50.0
    cols = [tc ** k for k in range(poly + 1)] + [np.exp(2j * np.pi * nu * (t - T_REF)) for nu in freqs]
    G = np.column_stack(cols).astype(complex)
    c, *_ = np.linalg.lstsq(G, z.astype(complex), rcond=None)
    return dict(zip(freqs, c[poly + 1:])), z - G @ c, G @ c


def fit_real(t, x, freqs, poly=2):
    """Real series: cos/sin at each positive frequency; returns phasors A e^{i phi} for x = A cos(2 pi f t + phi)."""
    tc = (t - t.mean()) / 50.0
    cols = [tc ** k for k in range(poly + 1)]
    for f in freqs:
        ph = 2 * np.pi * f * (t - T_REF); cols += [np.cos(ph), np.sin(ph)]
    G = np.column_stack(cols); c, *_ = np.linalg.lstsq(G, x, rcond=None)
    out = {f: complex(c[poly + 1 + 2 * i], -c[poly + 2 + 2 * i]) for i, f in enumerate(freqs)}
    return out, x - G @ c, G @ c


def closure(c, fc):
    D = np.angle(c[fc + 1] * c[fc - 1] / c[fc] ** 2)
    z = c[fc] / c[fc + 1] / abs(c[fc] / c[fc + 1]) + c[fc - 1] / c[fc] / abs(c[fc - 1] / c[fc])
    return float(D), float((np.angle(z) / (2 * np.pi)) % 1.0 * YEAR)


def complex_freqs(fc):
    return [fc, -fc, 1.0, -1.0, 2.0, -2.0, fc + 1, fc - 1, -(fc + 1), -(fc - 1), fc + 2, fc - 2]


def real_freqs(fc):
    return [fc, fc + 1, fc - 1, fc + 2, fc - 2, 1.0, 2.0]   # fc - 1, fc - 2 negative: the folded images


def best_fc(t, z, cplx):
    grid = np.linspace(0.835, 0.852, 171)
    if cplx:
        return float(grid[np.argmax([abs(fit_lines(t, z, [f, 1.0, -1.0])[0][f]) for f in grid])])
    return float(grid[np.argmax([abs(fit_real(t, z, [f, 1.0])[0][f]) for f in grid])])


def analyse_complex(t, p, nboot=300, seed=0):
    fc = best_fc(t, p, True)
    c, resid, model = fit_lines(t, p, complex_freqs(fc))
    D, day = closure(c, fc)
    A = abs(c[fc])
    body = abs(c[fc + 1]) ** 2 + abs(c[fc - 1]) ** 2
    mirror = abs(c[-(fc + 1)]) ** 2 + abs(c[-(fc - 1)]) ** 2
    r = dict(fc=fc, cw_amp_mas=1e3 * A, ratio_u=abs(c[fc + 1]) / A, ratio_l=abs(c[fc - 1]) / A,
             mirror_u=abs(c[-(fc + 1)]) / A, mirror_l=abs(c[-(fc - 1)]) / A, R=float(mirror / body),
             retro_cw=abs(c[-fc]) / A, D=D, am_peak_day=day)
    rng = np.random.default_rng(seed); blk = 40; n = len(p); Ds, days, Rs = [], [], []
    for _ in range(nboot):
        st = rng.integers(0, n - blk, n // blk + 1)
        pb = model + np.concatenate([resid[s:s + blk] for s in st])[:n]
        cb, _, _ = fit_lines(t, pb, complex_freqs(fc)); d, dd = closure(cb, fc); Ds.append(d); days.append(dd)
        Rs.append((abs(cb[-(fc + 1)]) ** 2 + abs(cb[-(fc - 1)]) ** 2) / (abs(cb[fc + 1]) ** 2 + abs(cb[fc - 1]) ** 2))
    circ = lambda a: float(np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * np.array(a))))))))
    r["D_sd_deg"] = circ(Ds); r["am_peak_day_sd"] = circ(2 * np.pi * np.array(days) / YEAR) / 360 * YEAR
    r["R_p16_p84"] = [float(np.percentile(Rs, 16)), float(np.percentile(Rs, 84))]
    return r


def analyse_real(t, x, nboot=300, seed=0):
    fc = best_fc(t, x, False)
    c, resid, model = fit_real(t, x, real_freqs(fc))
    D, day = closure(c, fc); A = abs(c[fc])
    r = dict(fc=fc, ratio_u=abs(c[fc + 1]) / A, ratio_l=abs(c[fc - 1]) / A, D=D, am_peak_day=day)
    rng = np.random.default_rng(seed); blk = 40; n = len(x); Ds = []
    for _ in range(nboot):
        st = rng.integers(0, n - blk, n // blk + 1)
        cb, _, _ = fit_real(t, model + np.concatenate([resid[s:s + blk] for s in st])[:n], real_freqs(fc))
        Ds.append(closure(cb, fc)[0])
    r["D_sd_deg"] = float(np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * np.array(Ds))))))))
    return r


def which_component(tb, xb, yb, td, xd):
    """Correlate cw.dat with x and y after removing a quadratic, over the overlap (both on the 0.05-yr grid)."""
    m = (td >= tb[0]) & (td <= tb[-1])
    out = {}
    for name, s in (("x", xb), ("y", yb)):
        si = np.interp(td[m], tb, s)
        tc = td[m] - td[m].mean(); P = np.polyval(np.polyfit(tc, si, 2), tc)
        out[name] = float(np.corrcoef(si - P, xd[m])[0, 1])
    k = np.polyfit(np.interp(td[m], tb, xb if abs(out["x"]) > abs(out["y"]) else yb) -
                   np.polyval(np.polyfit(td[m] - td[m].mean(), np.interp(td[m], tb, xb if abs(out["x"]) > abs(out["y"]) else yb), 2),
                              td[m] - td[m].mean()), xd[m], 1)[0]
    out["scale_cwdat_per_arcsec"] = float(k)
    return out


def cwdat_vs_raw(tb, xb, yb, td, xd, a=1973.0, b=2018.6):
    """How cw.dat relates to raw x, y: per-line gains, what is left after regressing on x and y, and its season."""
    m = (td >= a) & (td <= b); t, c = td[m], xd[m]
    x, y = np.interp(t, tb, xb), np.interp(t, tb, yb)
    fc = 0.8427
    fr = [fc, fc + 1, fc - 1, fc + 2, fc - 2, 1.0, 2.0, 1.5, 3.0]
    cc, _, _ = fit_real(t, c, fr); cx, _, _ = fit_real(t, x, fr)
    k0 = abs(cc[fc] / cx[fc])
    gains = {f"{f:.4f}": dict(cwdat=abs(cc[f]), x_scaled=abs(cx[f]) * k0, gain=abs(cc[f] / cx[f]) / k0) for f in fr}
    tc = t - t.mean()
    G = np.column_stack([x, y, np.ones_like(t), tc, tc ** 2]); k, *_ = np.linalg.lstsq(G, c, rcond=None); r = c - G @ k
    month = ((t % 1) * 12).astype(int)
    seg = {}
    for s0, s1 in ((1973, 1985), (1985, 1997), (1997, 2009), (2009, 2018.6)):
        mm = (t >= s0) & (t < s1); Gs = np.column_stack([x[mm], np.ones(mm.sum()), tc[mm]])
        seg[f"{s0}-{s1}"] = float(np.linalg.lstsq(Gs, c[mm], rcond=None)[0][0])
    def pk(sig, lo, hi):
        s = (sig - sig.mean()) * np.hanning(len(sig)); f = np.fft.rfftfreq(len(s) * 8, DT)
        P = np.abs(np.fft.rfft(s, len(s) * 8)) ** 2; return P[(f > lo) & (f < hi)].max()
    ref = pk(c, 0.8, 0.9)
    d = np.diff(td); brk = [float(td[j + 1]) for j in np.where(abs(d - DT) > 1e-4)[0]]
    return dict(window=[a, b], fc=fc, x_coef=float(k[0]), y_coef=float(k[1]), resid_over_cwdat_std=float(r.std() / c.std()),
                gains=gains, gain_by_segment=seg, resid_rms_by_month=[float(r[month == j].std()) for j in range(12)],
                sat_power_cwdat=[pk(c, 0.12, 0.2) / ref, pk(c, 1.8, 1.9) / ref],
                sat_power_resid=[pk(r, 0.12, 0.2) / ref, pk(r, 1.8, 1.9) / ref], grid_breaks=brk)


def main():
    if not FINALS.exists():
        raise SystemExit("data/finals2000A.all missing: run fetch_eop.py first")
    tb, xb, yb = load_finals()
    p = xb - 1j * yb                                   # prograde convention: the CW turns at +0.843 cycles/yr
    cw = np.loadtxt(HERE.parent / "cw.dat"); td, xd = cw[:, 0], cw[:, 1]
    out = {"span": [float(tb[0]), float(tb[-1])], "n": int(len(tb)), "component": which_component(tb, xb, yb, td, xd),
           "cwdat_vs_raw": cwdat_vs_raw(tb, xb, yb, td, xd), "windows": {}}
    for name, (a, b) in WINDOWS.items():
        m = (tb >= a) & (tb < b)
        w = {"complex p": analyse_complex(tb[m], p[m]), "x": analyse_real(tb[m], xb[m]), "y": analyse_real(tb[m], yb[m])}
        md = (td >= a) & (td < b)
        if md.sum() > 200:
            w["cw.dat"] = analyse_real(td[md], xd[md])
        out["windows"][name] = w
    (HERE / "cw_eop_closure.json").write_text(json.dumps(out, indent=1))
    report(out)
    plot(tb, p, out)


def report(out):
    print(f"finals2000A: {out['span'][0]:.2f}-{out['span'][1]:.2f}, {out['n']} bins; cw.dat vs x/y: {out['component']}")
    v = out["cwdat_vs_raw"]
    print(f"cw.dat = {v['x_coef']:.4f} x + {v['y_coef']:.4f} y + ...; resid/cw.dat std {v['resid_over_cwdat_std']:.3f}; "
          f"grid breaks {v['grid_breaks']}; gain by segment {v['gain_by_segment']}")
    for f, g in v["gains"].items():
        print(f"  line {f}: cw.dat {g['cwdat']:.5f}  scaled x {g['x_scaled']:.5f}  gain {g['gain']:.2f}")
    print("  satellite power (0.157, 1.843) / CW: cw.dat", [f"{q:.1e}" for q in v["sat_power_cwdat"]],
          " residual", [f"{q:.1e}" for q in v["sat_power_resid"]])
    print("  residual rms by month (x1e3):", [round(1e3 * q, 1) for q in v["resid_rms_by_month"]])
    for wn, w in out["windows"].items():
        print(f"== {wn}")
        for k, r in w.items():
            s = (f"  {k:10s} fc {r['fc']:.4f}  sidebands u/l {r['ratio_u']:.3f}/{r['ratio_l']:.3f}  "
                 f"D {np.degrees(r['D']):6.1f} +- {r['D_sd_deg']:4.1f}  AM peak day {r['am_peak_day']:.0f}")
            if "R" in r:
                s += (f"  mirror u/l {r['mirror_u']:.3f}/{r['mirror_l']:.3f}  R {r['R']:.3f} "
                      f"[{r['R_p16_p84'][0]:.3f}, {r['R_p16_p84'][1]:.3f}]  day sd {r['am_peak_day_sd']:.0f}  "
                      f"retro CW {r['retro_cw']:.3f}")
            print(s)


def plot(tb, p, out):
    fig = plt.figure(figsize=(16, 9.5))
    ax = fig.add_axes([0.06, 0.56, 0.9, 0.38])
    m = tb >= 1990
    tt, pp = tb[m], p[m]
    tc = (tt - tt.mean()) / 50; G = np.column_stack([tc ** k for k in range(3)])
    pp = pp - G @ np.linalg.lstsq(G, pp, rcond=None)[0]
    w = np.hanning(len(pp)); nf = len(pp) * 8
    P = np.abs(np.fft.fftshift(np.fft.fft(pp * w, nf))) ** 2; f = np.fft.fftshift(np.fft.fftfreq(nf, DT))
    P /= P[(f > 0.8) & (f < 0.9)].max()
    ax.semilogy(f, P, color="k", lw=1)
    fc = out["windows"]["1990-2026"]["complex p"]["fc"]
    for v, lab, col in [(fc, "CW", "g"), (1, "annual", "0.5"), (-1, "annual\n(retro)", "0.5"),
                        (fc + 1, "f+1\n(body)", "#4a90d9"), (fc - 1, "f−1\n(body)", "#4a90d9"),
                        (-(fc + 1), "−(f+1)\n(mirror)", "#d94a4a"), (-(fc - 1), "−(f−1)\n(mirror)", "#d94a4a")]:
        ax.axvline(v, color=col, ls=":", lw=1.2); ax.text(v + 0.02, 3, lab, color=col, fontsize=8, va="top")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(1e-6, 10); ax.set_xlabel("frequency (cycles/yr; + prograde, − retrograde)")
    ax.set_ylabel("power / CW peak")
    r = out["windows"]["1990-2026"]["complex p"]
    ax.set_title(f"(a) Complex polar motion p = x − iy, IERS daily 1990–2026 (space geodesy): body-sense sidebands "
                 f"{r['ratio_u']:.3f} / {r['ratio_l']:.3f} of the CW, mirror-sense {r['mirror_u']:.3f} / {r['mirror_l']:.3f}",
                 fontsize=10)

    ax = fig.add_axes([0.06, 0.07, 0.42, 0.38])  # (b)
    names = list(out["windows"]); series = ["complex p", "x", "y", "cw.dat"]
    cols = {"complex p": "k", "x": "#4a90d9", "y": "#2e8b57", "cw.dat": "#d94a4a"}
    for i, wn in enumerate(names):
        for j, s in enumerate(series):
            if s not in out["windows"][wn]:
                continue
            r = out["windows"][wn][s]; D = np.degrees(r["D"]); D = D - 360 if D > 260 else D
            ax.errorbar(i + (j - 1.5) * 0.15, D, yerr=r["D_sd_deg"], fmt="o", color=cols[s], capsize=3,
                        label=s if i == 0 else None)
    ax.axhspan(-10, 10, color="#4a90d9", alpha=0.1); ax.axhline(180, color="#2e8b57", ls="--", lw=0.8)
    ax.text(len(names) - 0.6, 3, "AM", fontsize=8); ax.text(len(names) - 0.6, 183, "PM", fontsize=8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(-200, 280)
    ax.set_ylabel("phase closure D (deg)"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(b) Phase closure: raw x, y and complex p against cw.dat", fontsize=10)

    v = out["cwdat_vs_raw"]
    ax = fig.add_axes([0.56, 0.07, 0.2, 0.38])
    labels = {"0.8427": "CW", "1.0000": "annual", "2.0000": "semi-\nannual", "-0.1573": "1−CW", "1.8427": "1+CW",
              "2.8427": "2+CW", "-1.1573": "2−CW", "1.5000": "1.5", "3.0000": "3.0"}
    keys = ["0.8427", "1.0000", "2.0000", "-1.1573", "-0.1573", "1.8427", "2.8427", "1.5000", "3.0000"]
    g = [v["gains"][k]["gain"] for k in keys]
    ax.bar(range(len(keys)), g, color=["#2e8b57" if 0.8 < q < 1.25 else "#d94a4a" for q in g])
    ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log"); ax.set_ylim(0.05, 10)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([labels[k] for k in keys], fontsize=7, rotation=60)
    ax.set_ylabel("|cw.dat| / |scaled raw x| per line")
    ax.set_title("(c) cw.dat against raw x, 1973–2018", fontsize=10)
    ax = fig.add_axes([0.81, 0.07, 0.16, 0.38])
    ax.bar(range(1, 13), 1e3 * np.array(v["resid_rms_by_month"]), color="0.5")
    ax.set_xlabel("month"); ax.set_ylabel("rms of cw.dat − (a·x + b·y + trend)  (×10⁻³)")
    ax.set_title("(d) Seasonal residual", fontsize=10)
    fig.savefig(HERE / "cw_eop_closure.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
