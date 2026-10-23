#!/usr/bin/env python3
"""Flagship figure for docs/README.md: the essential Chandler wobble fit (2026-10-23).

The device of cw_magnetized_spheroid.md with the amplitude-step inertia pulses of section 5:
  drive     B^2, B proportional to the Moon's ecliptic latitude (draconic month, astronomical phase)
  carrier   inertia pulses twice a year; the torque reaches the Kittel/Chandler mode only through them
  pulses    rescale the wobble amplitude by 1.25 (enlarging pulse) and 1/1.25 (the other), from the sideband level
  timing    the enlarging pulse is searched only within days 230-262, the season implied by the sideband phases
One timing parameter is fitted (1935-1990); 1990-2018.6 is held out. Amplitude, projection, constant and the annual
wobble are set by linear regression, as in cw_magnetized_spheroid.py.
    python3 cw_flagship.py  -> cw_flagship.png
"""
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize_scalar

import cw_magnetized_spheroid as dev
import cw_sidebands as sb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
EPS = 0.25


def device(t0):
    tn, north = dev.comb_times(t0); _, k = dev.kicks(t0, 0.0, "beta")
    idx = np.minimum(np.round((tn - dev.TG[0]) / dev.DT).astype(int), len(dev.TG) - 1)
    step = np.exp((1j * dev.OMEGA - dev.LAM) * dev.DT)
    kick = np.zeros(len(dev.TG), complex); fac = np.ones(len(dev.TG))
    np.add.at(kick, idx, k); fac[idx] = np.where(north, 1 + EPS, 1 / (1 + EPS))
    p = np.empty(len(dev.TG), complex); z = 0j
    for i in range(len(dev.TG)):
        z = z * fac[i] + kick[i]; p[i] = z; z *= step
    return dev.on_data(p)


def main():
    res = minimize_scalar(lambda t0: -dev.score(*device(t0), dev.TRAIN)[0][0],
                          bounds=(230 / dev.YEAR, 262 / dev.YEAR), method="bounded", options=dict(xatol=1e-5))
    t0 = res.x
    pr, pi_ = device(t0)
    (rtr, rho), c = dev.score(pr, pi_, dev.TRAIN, dev.HOLD)
    y = dev.design(pr, pi_, dev.TD) @ c
    ps = np.exp(2j * np.pi * sb.FC_PRED * dev.TG)
    (str_, sho), cs = dev.score(*dev.on_data(ps), dev.TRAIN, dev.HOLD)
    clos = sb.summarize(dev.TD, y, noise_mc=0)
    obs = sb.summarize(dev.TD, dev.XD, noise_mc=0)
    print(f"enlarging pulse day {t0 * dev.YEAR:.1f}, reducing pulse day {(t0 + 0.5) * dev.YEAR % dev.YEAR:.1f}")
    print(f"device r train {rtr:.3f} held out {rho:.3f}; sinusoid {str_:.3f} / {sho:.3f}")
    print(f"model D1 {np.degrees(clos['D1']):.1f} deg, sidebands {clos['ratio_l1']:.3f}/{clos['ratio_u1']:.3f}, "
          f"AM peak day {clos['am_peak_day']:.0f}; data {np.degrees(obs['D1']):.1f}, "
          f"{obs['ratio_l1']:.3f}/{obs['ratio_u1']:.3f}, day {obs['am_peak_day']:.0f}")

    fig = plt.figure(figsize=(15, 8.2))
    ax = fig.add_axes([0.06, 0.56, 0.9, 0.38])
    ax.plot(dev.TD, dev.XD, color="k", lw=0.9, label="cw.dat (Chandler band of polar motion)")
    ax.plot(dev.TD, y, color="#d94a4a", lw=1.1,
            label="device: draconic B² × semiannual inertia comb, amplitude-step pulses (1 timing parameter)")
    ax.axvspan(*dev.HOLD, color="0.93", zorder=0)
    ax.text(2004, 0.125, "held out", ha="center", fontsize=9)
    ax.set_xlim(1934, 2019.5); ax.set_ylim(-0.16, 0.15); ax.set_xlabel("year")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.set_title(f"(a) Fit 1935–1990: r = {rtr:.2f}; held out 1990–2018: r = {rho:.2f}   "
                 f"(a plain sinusoid at the predicted 432.8 d: {str_:.2f} / {sho:.2f})", fontsize=10)

    ax = fig.add_axes([0.06, 0.07, 0.56, 0.38])
    fD, PD = dev.spectrum(dev.XD); fM, PM = dev.spectrum(y)
    ax.semilogy(fD, PD, color="k", lw=1.1, label="cw.dat")
    ax.semilogy(fM, PM, color="#d94a4a", lw=1.0, label="device")
    fc = sb.FC_PRED
    for v, lab, yy, ha in [(fc, "CW = |2f_D − 26| = 0.844\n(432.8 d, not fitted)", 3, "right"), (1 - fc, "1 − CW", 3, "left"),
                           (1 + fc, "1 + CW", 3, "right"), (1.0, "annual wobble\n(regression term)", 3, "left")]:
        ax.axvline(v, ls=":", color="g", lw=1)
        ax.text(v + (0.01 if ha == "left" else -0.01), yy, lab, fontsize=7.5, color="g", va="top", ha=ha)
    ax.set_xlim(0.02, 2.0); ax.set_ylim(1e-5, 5); ax.set_xlabel("frequency (cycles/yr)")
    ax.set_ylabel("power / CW peak"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(b) Spectrum: the carrier line and its annual mixing products", fontsize=10)

    ax = fig.add_axes([0.68, 0.07, 0.29, 0.38]); ax.axis("off")
    rows = [("", "data", "device"),
            ("CW line (cycles/yr)", f"{obs['fc']:.4f}", f"{clos['fc']:.4f}"),
            ("1 − CW / CW (amplitude)", f"{obs['ratio_l1']:.3f}", f"{clos['ratio_l1']:.3f}"),
            ("1 + CW / CW (amplitude)", f"{obs['ratio_u1']:.3f}", f"{clos['ratio_u1']:.3f}"),
            ("phase closure D₁", f"{np.degrees(obs['D1']):.0f}°", f"{np.degrees(clos['D1']):.0f}°"),
            ("largest-amplitude day", f"{obs['am_peak_day']:.0f}", f"{clos['am_peak_day']:.0f}")]
    tb = ax.table(cellText=rows, loc="upper center", cellLoc="center", colWidths=[0.55, 0.22, 0.22])
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
    ax.set_title("(c) What the device must reproduce", fontsize=10)
    ax.text(0.0, 0.08, f"Enlarging pulse: day {t0 * dev.YEAR:.0f};  reducing pulse: day "
                       f"{(t0 + 0.5) * dev.YEAR % dev.YEAR:.0f}.\nAmplitude step 1.25 set from the sideband level; "
                       "Q = 85, free period 432.3 d.", fontsize=8, transform=ax.transAxes)
    fig.savefig(HERE / "cw_flagship.png", dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
