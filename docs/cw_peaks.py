#!/usr/bin/env python3
"""Every peak of the Chandler band, in cw.dat and in raw IERS polar motion (2026-10-23).
Discussion: cw_data_tests.md, sections 2-4. Needs data/finals2000A.all (fetch_eop.py).

  catalog       local maxima of the cw.dat periodogram (1935-2018.6) and of raw complex polar motion p = x - i y
                (1973-2026, both senses), each matched to combinations n1 f_CW + n2 (annual) and to the lunar alias lines
  artifact      over the 1973-2018.6 overlap, how much of each cw.dat line survives regression on raw x, y, and on
                multiplicative models: instantaneous products (x^2, y^2, xy), lagged products, a seasonal (calendar) gain
  stability     amplitude and absolute phase of fixed-frequency lines in successive windows; a forced line keeps its
                phase, a sidelobe of a changing CW does not. Monte Carlo: how often a stochastically excited free
                wobble (Q = 85) holds its phase as well as the observed CW does
    python3 cw_peaks.py  -> cw_peaks.png, cw_peaks.json
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
F_D, F_T = YEAR / 27.212221, YEAR / 27.321582
F_A, F_S = YEAR / 27.554550, YEAR / 29.530589
F_CW = 2 * F_D - 26                      # 0.8440, draconic B^2 alias
F_2N = 2 * (F_D - F_T)                   # 0.1075, twice the nodal frequency
NAMED = {"Mm = 14 − A (anomalistic B, same comb)": 14 - F_A, "tropical B² alias = CW − 2N": 2 * F_T - 26,
         "Mm2 = 2A − 26 (anomalistic B²)": 2 * F_A - 26, "Drac2 = 28 − 2D (next comb harmonic)": 28 - 2 * F_D,
         "tropical B alias 14 − T": 14 - F_T, "F+L alias": F_D + F_T - 26, "CW + 2N": F_CW + F_2N,
         "draconic B alias (same comb)": abs(F_D - 14), "QBO rule (B, alternating)": F_D - 13,
         "nodal 18.6 yr": F_D - F_T, "2N (9.3 yr)": F_2N}
BLOG = {"Mm2 0.5105": 2 * F_A - 26, "T 0.6317": 14 - F_T, "T B² 0.7365": 2 * F_T - 26, "Mm 0.7448": 14 - F_A,
        "CW 0.8440": F_CW, "Drac2 1.1560": 28 - 2 * F_D, "1.1848": 1.1848, "semiannual 2.0": 2.0}


def identify(f, tol=0.012):
    cands = []
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            v = n1 * F_CW + n2
            if (n1 or n2) and abs(abs(v) - f) < tol:
                if v < 0:
                    n1, n2 = -n1, -n2
                terms = [(n2, ""), (n1, "CW")] if n2 > 0 else [(n1, "CW"), (n2, "")]
                lab = ""
                for n, sym in terms:
                    if not n:
                        continue
                    mag = (str(abs(n)) if (abs(n) != 1 or not sym) else "") + sym
                    lab += ("−" if n < 0 else ("+" if lab else "")) + mag
                cands.append((abs(abs(v) - f), lab, abs(n1) + abs(n2)))
    for k, v in NAMED.items():
        if abs(v - f) < tol:
            cands.append((abs(v - f), k, 0))
    cands.sort(key=lambda c: (c[2] if c[2] else 0, c[0]))
    return list(dict.fromkeys(c[1] for c in cands))[:3]


def periodogram(x, dt=0.05, pad=8, two_sided=False):
    x = (x - x.mean()) * np.hanning(len(x))
    if two_sided:
        f = np.fft.fftshift(np.fft.fftfreq(len(x) * pad, dt)); P = np.abs(np.fft.fftshift(np.fft.fft(x, len(x) * pad))) ** 2
    else:
        f = np.fft.rfftfreq(len(x) * pad, dt); P = np.abs(np.fft.rfft(x, len(x) * pad)) ** 2
    return f, P / P[(f > 0.8) & (f < 0.9)].max()


def peaks(f, P, lo, hi, thresh=3e-4, half=0.03):
    out = []
    for i in np.where((np.abs(f) > lo) & (np.abs(f) < hi))[0]:
        w = np.abs(f - f[i]) < half
        if P[i] >= thresh and P[i] == P[w].max():
            out.append((float(f[i]), float(P[i])))
    return out


def detrend(t, s):
    tc = (t - t.mean()) / 50; G = np.column_stack([tc ** k for k in range(3)])
    return s - G @ np.linalg.lstsq(G, s, rcond=None)[0]


def artifact_models(t, c, X, Y):
    tc = (t - t.mean()) / 50
    base = [np.ones_like(t), tc, tc ** 2, X, Y]
    C1, S1, C2, S2 = (fn(k * 2 * np.pi * t) for k in (1, 2) for fn in (np.cos, np.sin))
    lag = [X * np.roll(X, L) for L in (5, 10, 15)] + [Y * np.roll(Y, L) for L in (5, 10, 15)] + \
          [X * np.roll(Y, L) for L in (5, 10, 15)] + [Y * np.roll(X, L) for L in (5, 10, 15)]
    models = {"linear a·x + b·y": [], "+ x², y², xy (squared amplitude)": [X * X, Y * Y, X * Y],
              "+ lagged products (0.25–0.75 yr)": [X * X, Y * Y, X * Y] + lag,
              "+ seasonal gain on x, y (annual)": [X * C1, X * S1, Y * C1, Y * S1],
              "+ seasonal gain (annual + semiannual)": [X * C1, X * S1, Y * C1, Y * S1, X * C2, X * S2, Y * C2, Y * S2]}
    s = (c - c.mean()) * np.hanning(len(c)); f = np.fft.rfftfreq(len(c) * 8, 0.05)
    ref = (np.abs(np.fft.rfft(s, len(c) * 8)) ** 2)[(f > 0.8) & (f < 0.9)].max()
    lines = {"0.157": 0.157, "0.737": 0.7365, "0.951": 0.9514, "1.157": 1.157, "1.5": 1.5, "1.843": 1.843,
             "2.843": 2.843, "3.0": 3.0}
    def lp(r, f0):
        s = (r - r.mean()) * np.hanning(len(r)); P = np.abs(np.fft.rfft(s, len(r) * 8)) ** 2 / ref
        return float(P[np.abs(f - f0) < 0.02].max())
    out = {"cw.dat": {k: lp(c, v) for k, v in lines.items()}}
    for name, extra in models.items():
        G = np.column_stack(base + extra); k, *_ = np.linalg.lstsq(G, c, rcond=None); r = c - G @ k
        out[name] = {k2: lp(r, v) for k2, v in lines.items()}
        out[name]["resid_rms_frac"] = float(r.std() / c.std())
        if name == "+ seasonal gain on x, y (annual)":
            out[name]["x_gain_depth"] = float(np.hypot(k[5], k[6]) / abs(k[3]))
    return out


def window_lines(t, z, wins, freqs, cplx):
    res = {}
    for a, b in wins:
        m = (t >= a) & (t < b)
        c, _, _ = E.fit_lines(t[m], z[m], freqs + [1.0, -1.0, 2.0]) if cplx else E.fit_real(t[m], z[m], freqs + [1.0, 2.0])
        A = abs(c[F_CW])
        res[f"{a:.0f}-{b:.1f}"] = {f"{f:.4f}": (float(abs(c[f]) / A), float(np.degrees(np.angle(c[f])))) for f in freqs}
    return res


def blog_lines(tb, p, td, xd):
    """Each labelled line of the 2026-03-06 blog spectrum: amplitude (mas) and absolute phase per window, raw p
    (prograde) in four 13.5-yr windows, and the chance that random phases would be as steady."""
    rng = np.random.default_rng(0)
    ph = rng.uniform(-np.pi, np.pi, (100000, 4)); mean = np.angle(np.exp(1j * ph).mean(1))
    null = np.degrees(np.abs(np.angle(np.exp(1j * (ph - mean[:, None])))).max(1))
    wins = [(1973, 1986.5), (1986.5, 2000), (2000, 2013.5), (2013.5, 2027)]
    out = {}
    for name, f in BLOG.items():
        rows = []
        for a, b in wins:
            m = (tb >= a) & (tb < b)
            fr = list(dict.fromkeys([f, F_CW, 1.0, -1.0, 2.0, -2.0]))
            c, _, _ = E.fit_lines(tb[m], p[m], fr)
            rows.append((float(abs(c[f]) * 1e3), float(abs(c[F_CW]) * 1e3), float(np.degrees(np.angle(c[f])))))
        spread = phase_spread([r[2] for r in rows])
        grid = np.linspace(f - 0.02, f + 0.02, 81) if name != "semiannual 2.0" else [2.0]
        best = max(grid, key=lambda g: abs(E.fit_lines(tb, p, list(dict.fromkeys([g, F_CW, 1.0, -1.0, 2.0])))[0][g]))
        out[name] = dict(f=f, windows=rows, spread_deg=spread, p_random=float(np.mean(null <= spread)), best_f=float(best))
    return out


def feature_0817(td, xd):
    """The cw.dat peak at 0.817 /yr, once a candidate for the free mode: amplitude, phase and closure with the CW and
    0.871 in windows long enough to separate it from the CW (>= 28 yr; 0.027 /yr apart)."""
    fr = [0.817, F_CW, 0.871, 1.0, 2.0]; out = {}
    for wins in ([(1935, 1977), (1977, 2018.7)], [(1935, 1963), (1963, 1991), (1991, 2018.7)]):
        rows = []
        for a, b in wins:
            m = (td >= a) & (td < b); c, _, _ = E.fit_real(td[m], xd[m], fr); A = abs(c[F_CW])
            rows.append(dict(window=[a, b], amp_0817=float(abs(c[0.817]) / A), phase_0817=float(np.degrees(np.angle(c[0.817]))),
                             amp_0871=float(abs(c[0.871]) / A), phase_0871=float(np.degrees(np.angle(c[0.871]))),
                             closure=float(np.degrees(np.angle(c[0.817] * c[0.871] / c[F_CW] ** 2)))))
        out[f"{len(wins)} windows"] = dict(rows=rows, spread_0817=phase_spread([r["phase_0817"] for r in rows]),
                                           spread_0871=phase_spread([r["phase_0871"] for r in rows]))
    return out


def phase_spread(ph):
    ph = np.radians(ph); m = np.angle(np.mean(np.exp(1j * ph)))
    return float(np.degrees(np.max(np.abs(np.angle(np.exp(1j * (ph - m)))))))


def monte_carlo(n_years, wins, nsim=2000, q=85.0, f0=1 / (432.3 / YEAR), seed=1):
    """Free wobble excited by white noise: how often is its phase at F_CW as steady as observed across the windows?"""
    rng = np.random.default_rng(seed); dt = 0.05; n = int(n_years / dt) + int(200 / dt)
    step = np.exp((2j * np.pi * f0 - np.pi * f0 / q) * dt)
    z = np.empty((n, nsim), complex); acc = np.zeros(nsim, complex)
    for i in range(n):                                       # all realizations advance together
        acc = acc * step + rng.normal(size=nsim) + 1j * rng.normal(size=nsim); z[i] = acc
    z = z[-int(n_years / dt):]; t = np.arange(len(z)) * dt
    ph = []
    for a, b in wins:
        m = (t >= a) & (t < b); tt = t[m]
        G = np.column_stack([np.exp(2j * np.pi * F_CW * tt), np.ones_like(tt)])
        ph.append(np.degrees(np.angle(np.linalg.lstsq(G, z[m], rcond=None)[0][0])))
    ph = np.radians(np.array(ph)); mean = np.angle(np.mean(np.exp(1j * ph), axis=0))
    return np.degrees(np.max(np.abs(np.angle(np.exp(1j * (ph - mean)))), axis=0))


def main():
    tb, xb, yb = E.load_finals(); p = xb - 1j * yb
    cw = np.loadtxt(HERE.parent / "cw.dat"); td, xd = cw[:, 0], cw[:, 1]
    out = {"F_CW": F_CW, "F_2N": F_2N, "named": NAMED}

    # catalog
    fC, PC = periodogram(xd)
    m = (tb >= 1973) & (tb < 2027); pp = detrend(tb[m], p[m])
    fR, PR = periodogram(pp, two_sided=True)
    cat = []
    mo = (td >= 1973) & (td <= 2018.6); t, c = td[mo], xd[mo]
    X, Y = detrend(t, np.interp(t, tb, xb)), detrend(t, np.interp(t, tb, yb))
    tc = (t - t.mean()) / 50; G = np.column_stack([np.ones_like(t), tc, tc ** 2, X, Y])
    r_lin = c - G @ np.linalg.lstsq(G, c, rcond=None)[0]
    fo, Pc_o = periodogram(c)
    s = (c - c.mean()) * np.hanning(len(c)); ref_o = (np.abs(np.fft.rfft(s, len(c) * 8)) ** 2)[(fo > 0.8) & (fo < 0.9)].max()
    sr = (r_lin - r_lin.mean()) * np.hanning(len(r_lin)); Pr_o = np.abs(np.fft.rfft(sr, len(c) * 8)) ** 2 / ref_o
    for f0, P0 in peaks(fC, PC, 0.03, 5.0):
        near = lambda f, P, g: float(P[np.abs(f - g) < 0.012].max())
        cat.append(dict(f=f0, power=P0, ids=identify(f0), raw_pro=near(fR, PR, f0), raw_retro=near(fR, PR, -f0),
                        overlap_power=near(fo, Pc_o, f0), overlap_resid=near(fo, Pr_o, f0)))
    out["catalog_cwdat"] = cat
    out["catalog_raw"] = [dict(f=f0, power=P0, ids=identify(abs(f0))) for f0, P0 in peaks(fR, PR, 0.03, 5.0)]

    out["artifact"] = artifact_models(t, c, X, Y)

    freqs = [14 - F_A, NAMED["tropical B² alias = CW − 2N"], NAMED["F+L alias"], F_CW, NAMED["CW + 2N"], F_CW - 1, F_CW + 1]
    out["stability_raw"] = window_lines(tb, p, [(1973, 1991), (1991, 2009), (2009, 2027)],
                                        [14 - F_A, NAMED["tropical B² alias = CW − 2N"], NAMED["F+L alias"], F_CW,
                                         NAMED["CW + 2N"], F_CW - 1, F_CW + 1], True)
    out["stability_cwdat"] = window_lines(td, xd, [(1935, 1963), (1963, 1991), (1991, 2018.7)], freqs, False)
    obs_raw = phase_spread([v[f"{F_CW:.4f}"][1] for v in out["stability_raw"].values()])
    obs_cw = phase_spread([v[f"{F_CW:.4f}"][1] for v in out["stability_cwdat"].values()])
    mc_raw = monte_carlo(54, [(0, 18), (18, 36), (36, 54)])
    mc_cw = monte_carlo(84, [(0, 28), (28, 56), (56, 84)])
    out["phase_lock"] = {"raw_spread_deg": obs_raw, "raw_p_value": float(np.mean(mc_raw <= obs_raw)),
                         "cwdat_spread_deg": obs_cw, "cwdat_p_value": float(np.mean(mc_cw <= obs_cw)),
                         "mc_median_spread_raw": float(np.median(mc_raw)), "mc_median_spread_cwdat": float(np.median(mc_cw))}
    out["blog_lines"] = blog_lines(tb, p, td, xd)
    out["feature_0817"] = feature_0817(td, xd)
    (HERE / "cw_peaks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    report(out)
    plot(fC, PC, fR, PR, out, mc_cw, mc_raw)


def report(out):
    print(f"F_CW {F_CW:.4f}  2N {F_2N:.4f}")
    print("cw.dat peaks (1935-2018): f, power/CW | raw p prograde, retro | overlap: cw.dat power, residual after x,y | ids")
    for r in out["catalog_cwdat"]:
        print(f"  {r['f']:.4f} {r['power']:.1e} | {r['raw_pro']:.1e} {r['raw_retro']:.1e} | {r['overlap_power']:.1e} "
              f"{r['overlap_resid']:.1e} | {', '.join(r['ids'])}")
    print("raw p peaks (1973-2026):")
    for r in out["catalog_raw"]:
        print(f"  {r['f']:+.4f} {r['power']:.1e} | {', '.join(r['ids'])}")
    print("artifact models (line power / CW left in residual):")
    for k, v in out["artifact"].items():
        print(f"  {k:40s}", {kk: (f"{vv:.1e}" if kk not in ("resid_rms_frac", "x_gain_depth") else round(vv, 3)) for kk, vv in v.items()})
    for nm in ("stability_raw", "stability_cwdat"):
        print(nm)
        for w, v in out[nm].items():
            print("  ", w, {k: f"{a:.3f}@{ph:.0f}" for k, (a, ph) in v.items()})
    print("phase lock:", out["phase_lock"])
    for k, v in out["feature_0817"].items():
        print(f"0.817 feature, {k}: amp {[round(r['amp_0817'], 2) for r in v['rows']]} phase {[round(r['phase_0817']) for r in v['rows']]} "
              f"(spread {v['spread_0817']:.0f}); 0.871 phase {[round(r['phase_0871']) for r in v['rows']]} (spread {v['spread_0871']:.0f}); "
              f"closure {[round(r['closure']) for r in v['rows']]}")
    print("blog lines, raw p 4 windows: |line| mas / |CW| mas @ phase; spread; p(random); best f near line")
    for k, v in out["blog_lines"].items():
        print(f"  {k:15s}", " ".join(f"{a:5.1f}/{c:4.0f}@{ph:5.0f}" for a, c, ph in v["windows"]),
              f"spread {v['spread_deg']:4.0f}  p {v['p_random']:.3f}  best {v['best_f']:.4f}")


def plot(fC, PC, fR, PR, out, mc_cw, mc_raw):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_axes([0.06, 0.69, 0.9, 0.27])
    ax.semilogy(fC, PC, color="k", lw=0.9)
    for r in out["catalog_cwdat"]:
        if r["f"] > 3.25:
            continue
        processed = r["overlap_resid"] > 0.5 * r["overlap_power"]
        early = r["overlap_power"] < 0.3 * r["power"]
        col = "0.55" if early else ("#d94a4a" if processed else "#2e8b57")
        ax.plot(r["f"], r["power"], "v", color=col, ms=6)
        if r["power"] > 4e-4:
            ax.text(r["f"], r["power"] * 1.6, (r["ids"][0] if r["ids"] else f"{r['f']:.3f}"), fontsize=6.5, rotation=90,
                    ha="center", va="bottom", color=col)
    ax.set_xlim(0.02, 3.3); ax.set_ylim(1e-5, 30); ax.set_ylabel("power / CW")
    ax.set_title("(a) cw.dat 1935–2018, every peak. Green: in raw polar motion (1973–2018 overlap). "
                 "Red: mostly in the part raw x, y do not explain (processing).\nGrey: weak in 1973–2018, so carried by the "
                 "pre-1973 record (not checkable against raw data)", fontsize=10)

    ax = fig.add_axes([0.06, 0.37, 0.9, 0.24])
    ax.semilogy(fR, PR, color="k", lw=0.9)
    for r in out["catalog_raw"]:
        if r["power"] > 4e-4:
            ax.plot(r["f"], r["power"], "v", color="#4a90d9", ms=5)
            ax.text(r["f"], r["power"] * 1.6, (r["ids"][0] if r["ids"] else f"{r['f']:+.3f}"), fontsize=6.5, rotation=90,
                    ha="center", va="bottom", color="#4a90d9")
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(1e-6, 30); ax.set_ylabel("power / CW")
    ax.set_xlabel("frequency (cycles/yr; + prograde, − retrograde)")
    ax.set_title("(b) Raw complex polar motion p = x − iy, IERS daily 1973–2026", fontsize=10)

    ax = fig.add_axes([0.06, 0.05, 0.42, 0.25])
    lines = ["0.157", "1.843", "1.157", "2.843", "3.0", "1.5", "0.737", "0.951"]
    models = list(out["artifact"])
    w = 0.8 / len(models)
    for j, mname in enumerate(models):
        ax.bar(np.arange(len(lines)) + (j - len(models) / 2 + 0.5) * w, [out["artifact"][mname][k] for k in lines], w,
               label=mname)
    ax.set_yscale("log"); ax.set_xticks(range(len(lines))); ax.set_xticklabels(lines)
    ax.set_ylabel("line power / CW left unexplained"); ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax.set_title("(c) cw.dat 1973–2018: what each processing model explains", fontsize=10)

    ax = fig.add_axes([0.56, 0.05, 0.4, 0.25])
    labs = {f"{NAMED['tropical B² alias = CW − 2N']:.4f}": "0.7365 trop. B²", f"{F_CW:.4f}": "0.8440 CW",
            f"{14 - F_A:.4f}": "0.7448 Mm",
            f"{NAMED['CW + 2N']:.4f}": "0.9514 CW+2N", f"{NAMED['F+L alias']:.4f}": "0.7903 F+L"}
    for nm, mk in (("stability_raw", "o"), ("stability_cwdat", "s")):
        for i, (wname, v) in enumerate(out[nm].items()):
            for k, lab in labs.items():
                if k in v:
                    ph = v[k][1] + (180 if nm == "stability_cwdat" else 0)       # cw.dat is -x: shift by 180 deg
                    ph = (ph + 180) % 360 - 180
                    col = {"0.8440 CW": "k", "0.7365 trop. B²": "#d94a4a", "0.9514 CW+2N": "#4a90d9", "0.7903 F+L": "#e67e22",
                           "0.7448 Mm": "#8e44ad"}[lab]
                    ax.plot(i + (0.15 if nm == "stability_cwdat" else 0), ph, mk, color=col,
                            label=f"{lab}" if (i == 0 and nm == "stability_raw") else None, ms=6)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["window 1", "window 2", "window 3"])
    pl = out["phase_lock"]
    ax.set_ylabel("absolute phase (deg)"); ax.set_ylim(-200, 200); ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.set_title(f"(d) Phase in successive windows (o raw 1973–2027, □ cw.dat 1935–2018).\nCW spread "
                 f"{pl['raw_spread_deg']:.0f}° / {pl['cwdat_spread_deg']:.0f}°; free noise-driven wobble: p = "
                 f"{pl['raw_p_value']:.2f} / {pl['cwdat_p_value']:.2f}", fontsize=9)
    fig.savefig(HERE / "cw_peaks.png", dpi=115); plt.close(fig)


if __name__ == "__main__":
    main()
