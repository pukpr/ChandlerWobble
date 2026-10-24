#!/usr/bin/env python3
"""Long-record phase coherence of the Chandler wobble against the draconic alias, IERS EOP C01 1846-2026 (2026-10-24).
Discussion: cw_longrecord.md. Needs data/eopc01.iau2000.1846-now (fetch_eop.py).

1. Phase track: complex polar motion p = x - i y in sliding 12-yr windows (1-yr step), each fitted with a prograde
   line at the fixed draconic alias f_CW = 2 * 365.2422 / 27.212221 - 26 = 0.84398 /yr, prograde and retrograde
   annual and semiannual lines, and a quadratic drift. A forced line at f_CW keeps a flat phase; a free mode driven
   by noise wanders.
2. Plateaus: the tightest 70-yr span (max deviation from its circular mean), and the tightest separate 40-yr span.
3. Null: a free wobble (432.3 d or exactly f_CW; Q = 50, 85, 150) driven by white noise, simulated on the same time
   grid and analysed identically, 1500 runs each (the search over all spans is included).
4. Coherent frequency: the single prograde frequency that best fits 1926-2026 (and 1846-2026).
5. The F - M beat out of sample: the CW + F - M two-line forcing fitted on 1973-2026 (as in cw_suppression.py),
   extrapolated back and compared with the C01 envelope 1890-1972.
    python3 cw_longrecord.py  -> cw_longrecord.png, cw_longrecord.json
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
C01 = HERE / "data" / "eopc01.iau2000.1846-now"
YEAR = 365.2422
F_CW = 2 * YEAR / 27.212221 - 26
F_FM_ALIAS = 1 - (YEAR / 27.212221 - YEAR / 27.554550)
WIN, STEP = 12.0, 1.0
NULLS = [(YEAR / 432.3, 85.0), (F_CW, 85.0), (YEAR / 432.3, 50.0), (F_CW, 50.0), (YEAR / 432.3, 150.0)]


def load():
    d = np.array([[float(v) for v in l.split()[:3]] for l in C01.read_text().splitlines()
                  if l.strip() and not l.startswith("#")])
    t = 2000.0 + (d[:, 0] - 51544.5) / 365.25
    return t, d[:, 1] - 1j * d[:, 2]


def window_rows(t):
    cents = np.arange(t[0] + WIN / 2, t[-1] - WIN / 2 + 1e-9, STEP); rows = []
    for c in cents:
        m = (t >= c - WIN / 2) & (t < c + WIN / 2); tt = t[m]; tc = tt - c
        G = np.column_stack([np.exp(2j * np.pi * fr * tt) for fr in (F_CW, 1.0, -1.0, 2.0, -2.0)] + [np.ones_like(tt), tc, tc ** 2])
        rows.append((m, np.linalg.pinv(G)[0]))
    return cents, rows


def spans(ph, k):
    dev, mean = [], []
    for i in range(ph.shape[0] - k + 1):
        seg = ph[i:i + k]; mu = np.angle(np.exp(1j * seg).mean(0))
        dev.append(np.degrees(np.abs(np.angle(np.exp(1j * (seg - mu)))).max(0))); mean.append(mu)
    return np.array(dev), np.array(mean)


def plateaus(ph):
    """Tightest 70-yr span; tightest non-overlapping 40-yr span; separation of their mean phases (deg)."""
    d70, m70 = spans(ph, 70); d40, m40 = spans(ph, 40)
    single = ph.ndim == 1
    if single:
        ph = ph[:, None]; d70, m70, d40, m40 = d70[:, None], m70[:, None], d40[:, None], m40[:, None]
    i70 = np.argmin(d70, axis=0); o70, o40, sep, j40 = [], [], [], []
    for s in range(ph.shape[1]):
        a = i70[s]; ok = [j for j in range(d40.shape[0]) if j + 40 <= a or j >= a + 70]
        if not ok:
            o70.append(d70[a, s]); o40.append(999.0); sep.append(0.0); j40.append(-1); continue
        j = ok[int(np.argmin(d40[ok, s]))]
        o70.append(d70[a, s]); o40.append(d40[j, s]); j40.append(j)
        sep.append(abs(np.degrees(np.angle(np.exp(1j * (m40[j, s] - m70[a, s]))))))
    return np.array(o70), np.array(o40), np.array(sep), i70, np.array(j40)


def coherent_freq(t, p, t0):
    m = t >= t0; tt = t[m]; tc = (tt - tt.mean()) / 50; grid = np.linspace(0.835, 0.853, 361); a = []
    for f in grid:
        G = np.column_stack([np.exp(2j * np.pi * fr * tt) for fr in (f, 1.0, -1.0, 2.0, -2.0)] + [np.ones_like(tt), tc, tc ** 2, tc ** 3])
        a.append(abs(np.linalg.pinv(G)[0] @ p[m]))
    return float(grid[int(np.argmax(a))])


def null_runs(t, rows, f0, q, nsim=1500, seed=3):
    rng = np.random.default_rng(seed + int(q) + int(1e4 * f0))
    tg = np.arange(t[0] - 200, t[-1] + 0.05, 0.05); step = np.exp((2j * np.pi * f0 - np.pi * f0 / q) * 0.05)
    acc = np.zeros(nsim, complex); Z = np.empty((len(tg), nsim), complex)
    for j in range(len(tg)):
        acc = acc * step + rng.normal(size=nsim) + 1j * rng.normal(size=nsim); Z[j] = acc
    Zs = Z[np.clip(np.round((t - tg[0]) / 0.05).astype(int), 0, len(tg) - 1)]
    return plateaus(np.angle(np.array([r @ Zs[m] for m, r in rows])))


def fm_beat_out_of_sample(t, p):
    """CW + F-M two-line forcing fitted on 1973-2026 in the alias-rotating frame; envelope compared with C01 1890-1972."""
    tc = (t - t.mean()) / 50
    G = np.column_stack([tc ** k for k in range(3)] + [np.exp(2j * np.pi * f * t) for f in (1.0, -1.0, 2.0, -2.0, 14 - YEAR / 27.554550)])
    r = (p - G @ np.linalg.lstsq(G, p, rcond=None)[0]) * 1e3
    w = r * np.exp(-2j * np.pi * F_CW * t)
    k = np.arange(-int(4 / 0.05), int(4 / 0.05) + 1) * 0.05; g = np.exp(-0.5 * (k / 1.0) ** 2); g /= g.sum()
    tu = np.arange(t[0], t[-1], 0.05); wu = np.interp(tu, t, w.real) + 1j * np.interp(tu, t, w.imag)
    env = np.convolve(wu, g, "same"); ok = (tu > tu[0] + 3) & (tu < tu[-1] - 3)
    out = {}; curves = {}
    for f0, q, label in ((YEAR / 432.3, 85.0, "Q 85"), (0.8432, 50.0, "Q 50"), (0.8432, 31.6, "Q 32")):
        a = 1j * 2 * np.pi * (f0 - F_CW) - np.pi * f0 / q; b = 1j * 2 * np.pi * (F_FM_ALIAS - F_CW)
        m = ok & (tu >= 1973)
        D = np.column_stack([np.exp(a * (tu - 1973)), (np.exp(a * (tu - 1973)) - 1) / a,
                             (np.exp(b * (tu - 1973)) - np.exp(a * (tu - 1973))) / (b - a)])
        c, *_ = np.linalg.lstsq(D[m], env[m], rcond=None)
        steady = np.abs(-c[1] / a + c[2] / (b - a) * np.exp(b * (tu - 1973)))
        mo = ok & (tu >= 1890) & (tu < 1972.5)
        tm = tu[ok & (tu > 1900) & (tu < 1950)][np.argmin(steady[ok & (tu > 1900) & (tu < 1950)])]
        curves[label] = steady
        out[label] = dict(r_1890_1972=float(np.corrcoef(steady[mo], np.abs(env[mo]))[0, 1]), predicted_min_1900_1950=float(tm),
                          observed_min_1900_1950=float(tu[ok & (tu > 1900) & (tu < 1950)][np.argmin(np.abs(env[ok & (tu > 1900) & (tu < 1950)]))]))
    return out, tu, env, ok, curves


def main():
    t, p = load()
    cents, rows = window_rows(t)
    trk = np.array([r @ p[m] for m, r in rows]); ph = np.angle(trk); amp = np.abs(trk) * 1e3
    o70, o40, osep, i70, j40 = plateaus(ph)
    out = dict(span=[float(t[0]), float(t[-1])], f_cw=F_CW, window_years=WIN,
               track=[dict(year=float(c), amp_mas=float(a), phase_deg=float(np.degrees(q))) for c, a, q in zip(cents, amp, ph)],
               plateau70=dict(start=float(cents[i70[0]]), end=float(cents[i70[0] + 69]), max_dev_deg=float(o70[0])),
               plateau40=dict(start=float(cents[j40[0]]), end=float(cents[j40[0] + 39]), max_dev_deg=float(o40[0])),
               plateau_separation_deg=float(osep[0]),
               coherent_freq_1926=coherent_freq(t, p, 1926.0), coherent_freq_1846=coherent_freq(t, p, 1846.0),
               malkin_miller_period_yr=1.185, nulls={})
    nulls = {}
    for f0, q in NULLS:
        s70, s40, ssep, _, _ = null_runs(t, rows, f0, q)
        both = (s70 <= o70[0]) & (s40 <= o40[0])
        nulls[(f0, q)] = s70
        out["nulls"][f"{YEAR / f0:.1f} d, Q {q:.0f}"] = dict(median_70=float(np.median(s70)), p_70=float(np.mean(s70 <= o70[0])),
                                                            p_both=float(np.mean(both)), p_both_reversed=float(np.mean(both & (ssep >= 150))),
                                                            n=int(len(s70)))
    beat, tu, env, ok, curves = fm_beat_out_of_sample(t, p)
    out["fm_beat_out_of_sample"] = beat
    (HERE / "cw_longrecord.json").write_text(json.dumps(out, indent=1))
    report(out)
    plot(cents, amp, ph, out, nulls, o70[0], tu, env, ok, curves)


def report(out):
    print(f"C01 {out['span'][0]:.1f}-{out['span'][1]:.1f}; f_CW {out['f_cw']:.5f}")
    print("70-yr plateau:", out["plateau70"], "| 40-yr plateau:", out["plateau40"], "| separation", round(out["plateau_separation_deg"]))
    print("coherent frequency 1926-2026:", out["coherent_freq_1926"], " 1846-2026:", out["coherent_freq_1846"])
    for k, v in out["nulls"].items():
        print(f"null {k}: median 70-yr spread {v['median_70']:.0f}; P(70) {v['p_70']:.4f}; P(both) {v['p_both']:.4f}; "
              f"P(both, reversed >= 150 deg) {v['p_both_reversed']:.4f} (n {v['n']})")
    print("F-M beat out of sample:", out["fm_beat_out_of_sample"])


def plot(cents, amp, ph, out, nulls, o70, tu, env, ok, curves):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0.06, 0.70, 0.9, 0.25])
    ax.plot(cents, amp, color="k", lw=1.3, label="CW amplitude at the draconic alias (12-yr windows, C01)")
    for y, lab in ((1850, "1850s"), (1925, "1920s reversal"), (2015, "post-2015 absence")):
        ax.axvline(y, color="#d94a4a", ls=":", lw=1); ax.text(y + 1, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 250, lab, fontsize=8, color="#d94a4a")
    ax.set_xlim(1846, 2027); ax.set_ylabel("mas"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) Chandler wobble amplitude, 1846–2026 (minima: Malkin & Miller 2010; Yamaguchi & Furuya 2024)", fontsize=10)
    ax = fig.add_axes([0.06, 0.38, 0.9, 0.25])
    ref = np.degrees(np.angle(np.exp(1j * ph[np.argmin(abs(cents - 1990))])))
    phd = (np.degrees(ph) - ref + 90) % 360 - 90                 # wrap to [-90, 270) so neither plateau splits
    ax.plot(cents, phd, ".", color="k", ms=4)
    p70, p40 = out["plateau70"], out["plateau40"]
    for pl, col in ((p70, "#d6eaf8"), (p40, "#fdebd0")):
        ax.axvspan(pl["start"], pl["end"], color=col, zorder=0)
    ax.text(p70["start"] + 2, 230, f"window centres {p70['start']:.0f}–{p70['end']:.0f}: within ±{p70['max_dev_deg']:.0f}°", fontsize=8)
    ax.text(p40["start"] + 2, 230, f"{p40['start']:.0f}–{p40['end']:.0f}: ±{p40['max_dev_deg']:.0f}°", fontsize=8)
    ax.set_xlim(1846, 2027); ax.set_ylim(-90, 270); ax.set_ylabel("phase vs 0.84398 /yr (deg, rel. 1990)")
    ax.set_title(f"(b) Phase relative to the draconic alias: two plateaus {out['plateau_separation_deg']:.0f}° apart, joined by the 1920s reversal", fontsize=10)
    ax = fig.add_axes([0.06, 0.06, 0.42, 0.24])
    for (f0, q), col in zip(nulls, ("0.5", "0.7", "#4a90d9", "#2e8b57", "#e67e22")):
        ax.hist(nulls[(f0, q)], bins=np.arange(0, 181, 5), histtype="step", color=col, lw=1.2, label=f"free {YEAR / f0:.1f} d, Q {q:.0f}")
    ax.axvline(o70, color="#d94a4a", lw=2); ax.text(o70 + 2, ax.get_ylim()[1] * 0.85, "observed", color="#d94a4a", fontsize=8)
    ax.set_xlabel("tightest 70-yr span: max phase deviation (deg)"); ax.set_ylabel("null runs"); ax.legend(fontsize=7)
    ax.set_title("(c) Noise-driven free wobble: how tight a 70-yr phase lock gets", fontsize=10)
    ax = fig.add_axes([0.56, 0.06, 0.4, 0.24])
    ax.plot(tu[ok], np.abs(env[ok]), color="k", lw=1.2, label="C01 envelope at the alias")
    ax.plot(tu[ok], curves["Q 50"][ok], color="#d94a4a", lw=1.4, label="CW + F − M beat, fitted 1973–2026 (Q 50)")
    ax.axvspan(1973, 2027, color="0.93", zorder=0); ax.text(1990, 10, "fitted", fontsize=8)
    b = out["fm_beat_out_of_sample"]["Q 50"]
    ax.set_xlim(1846, 2027); ax.set_ylabel("mas"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"(d) F − M beat fitted on 1973–2026 predicts its earlier minimum at {b['predicted_min_1900_1950']:.0f}\n"
                 f"(observed {b['observed_min_1900_1950']:.0f}); envelope r 1890–1972 = {b['r_1890_1972']:.2f}", fontsize=9)
    fig.savefig(HERE / "cw_longrecord.png", dpi=115); plt.close(fig)


if __name__ == "__main__":
    main()
