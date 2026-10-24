#!/usr/bin/env python3
"""The counter-torque that would remove the Chandler wobble, and a pulsed controller that does it (2026-10-23).
Discussion: cw_synchronized_wobble.md, section 5. Needs data/finals2000A.all (fetch_eop.py).

1. Inverse plant. Polar motion m = x - i y obeys the Liouville equation  (i / sigma) dm/dt + m = psi,
   sigma = 2 pi f0 (1 + i / 2Q). Given the observed m, psi is the excitation that must have acted; -psi is the
   counter-excitation that would cancel it. A torque Gamma enters as psi = -i Gamma / (Omega^2 (C - A)).
2. Budget. Compare the counter-torque in each line with the lunar torque on the equatorial bulge.
3. Pulsed control. The device of cw_magnetized_spheroid.py (draconic B^2 kicks on a semiannual comb) with a sampled
   feedback controller that fires counter-pulses at the same comb instants (as spin-stabilized spacecraft damp
   nutation with pulsed thrusters). In steady state the controller's pulses equal the aliased lunar kicks.
    python3 cw_counter_torque.py  -> cw_counter_torque.png, cw_counter_torque.json
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

import cw_eop_closure as E
import cw_magnetized_spheroid as dev

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
YEAR = 365.2422
F0 = YEAR / 432.3                    # free Chandler frequency, cycles/yr
F_CW = 2 * YEAR / 27.212221 - 26     # draconic B^2 alias
F_MM = 14 - YEAR / 27.554550         # anomalistic alias
MAS = np.pi / 180 / 3600e3
OMEGA = 7.292115e-5                  # rad/s
C, H = 8.034e37, 0.0032738           # polar moment (kg m^2) and dynamical ellipticity (C - A)/C
C_MINUS_A = C * H
G, M_MOON, R_MOON = 6.674e-11, 7.346e22, 3.844e8
SEC_PER_YEAR = YEAR * 86400


def excitation(t, m, q):
    """psi = m + (i / sigma) dm/dt, applied exactly in the frequency domain: psi(f) = m(f) (1 - f / s), with
    s = F0 (1 + i / 2Q) in cycles/yr. (A finite-difference derivative errs by ~1% at the CW, and psi is a ~1%
    residual of a near-cancellation there, so it must not be used.)"""
    n = len(m); dt = float(np.median(np.diff(t)))
    f = np.fft.fftfreq(n, dt)
    return np.fft.ifft(np.fft.fft(m) * (1 - f / (F0 * (1 + 1j / (2 * q)))))


def transfer_inverse(f, q):
    return 1 - f / (F0 * (1 + 1j / (2 * q)))


def torque_per_mas():
    return OMEGA ** 2 * C_MINUS_A * MAS                    # N m per mas of excitation


def lunar_bulge_torque():
    """Peak lunar torque on the equatorial bulge, (3 G M / r^3)(C - A) sin(d) cos(d), at the 18.6-yr maximum d = 28.6 deg."""
    d = np.radians(28.6)
    return 3 * G * M_MOON / R_MOON ** 3 * C_MINUS_A * np.sin(d) * np.cos(d)


def lines(t, z, freqs):
    c, _, _ = E.fit_lines(t, z, freqs)
    return c


def pulsed_control(gain=0.35, on_year=1900.0, t0=0.469):
    """Device response with a sampled feedback controller acting at the comb instants after on_year."""
    tn, k = dev.kicks(t0, 0.0, "beta")
    k = k / np.abs(k).max()
    idx = np.minimum(np.round((tn - dev.TG[0]) / dev.DT).astype(int), len(dev.TG) - 1)
    step = np.exp((1j * dev.OMEGA - dev.LAM) * dev.DT)
    kick = np.zeros(len(dev.TG), complex); np.add.at(kick, idx, k)
    pulse_at = np.zeros(len(dev.TG), bool); pulse_at[idx] = True
    z = 0j; out = np.empty(len(dev.TG), complex); u_hist = []
    for i in range(len(dev.TG)):
        if pulse_at[i] and dev.TG[i] >= on_year:
            u = -gain * z                                # fire against the wobble measured just before the pulse
            u_hist.append((dev.TG[i], kick[i], u))
            z = z + kick[i] + u
        else:
            z = z + kick[i]
        out[i] = z; z *= step
    return out, np.array(u_hist)


def main():
    tb, xb, yb = E.load_finals()
    m = (xb - 1j * yb) * 1e3 / 1e3                          # arcsec
    tc = (tb - tb.mean()) / 50
    G_ = np.column_stack([tc ** k for k in range(3)])
    m = m - G_ @ np.linalg.lstsq(G_, m, rcond=None)[0]
    m_mas = m * 1e3
    out = {"F0": F0, "F_CW": F_CW, "F_MM": F_MM, "torque_per_mas_Nm": torque_per_mas(),
           "lunar_bulge_torque_Nm": lunar_bulge_torque()}
    sel = (tb >= 1973) & (tb < 2013.5)
    freqs = [F_CW, F_MM, 1.0, -1.0, 2.0, -2.0]
    cm = lines(tb[sel], m_mas[sel], freqs)
    out["lines"] = {}
    for q in (50, 85, 150):                                  # exact per line: psi = m (1 - f / s)
        out["lines"][f"Q={q}"] = {f"{f:+.4f}": dict(motion_mas=float(abs(cm[f])),
                                                    excitation_mas=float(abs(cm[f] * transfer_inverse(f, q))),
                                                    counter_torque_Nm=float(abs(cm[f] * transfer_inverse(f, q)) * torque_per_mas()))
                                  for f in freqs}
    # the same lines through the bandpasses fitted without consensus constraints (cw_bandpass.py)
    bp_path = HERE / "cw_bandpass.json"
    if bp_path.exists():
        bp = json.loads(bp_path.read_text()); out["fitted_bandpasses"] = {}
        sols = [(r["f0"], r["Q"], k) for k, r in bp["runs"].items() if not k.startswith("consensus")]
        q32 = min(bp["Q_profile"], key=lambda r: (r[1] > 0.01, r[0])); sols.append((q32[2], q32[0], "lowest Q that fits"))
        for f0, q, label in sols:
            s_ = f0 * (1 + 1j / (2 * q))
            row = {f"{f:+.4f}": float(abs(cm[f] * (1 - f / s_)) * torque_per_mas()) for f in (F_CW, F_MM, 1.0)}
            out["fitted_bandpasses"][label] = dict(f0=f0, Q=q, counter_torque_Nm=row,
                                                   cw_over_lunar=row[f"{F_CW:+.4f}"] / lunar_bulge_torque())
    # pulse budget: impulse per semiannual pulse that sustains the CW line against damping (Q = 85)
    A_cw = abs(cm[F_CW]) * MAS                               # rad
    per_pulse = A_cw * (1 - np.exp(-np.pi * F0 / 85 * 0.5))  # amplitude lost per half year, made good by one pulse
    dH = C * OMEGA * per_pulse                                # transverse angular momentum per pulse (N m s)
    out["pulse_budget"] = {"cw_amp_mas": abs(cm[F_CW]), "per_pulse_mas": per_pulse / MAS, "angular_impulse_Nms": dH,
                           "hours_of_peak_lunar_torque": dH / lunar_bulge_torque() / 3600,
                           "mean_torque_Nm": dH / (0.5 * SEC_PER_YEAR)}
    # pulsed controller demo
    p_free, _ = pulsed_control(gain=0.0)
    p_ctl, u = pulsed_control(gain=0.35, on_year=1960.0)
    late = u[u[:, 0].real > 1990]
    tl = late[:, 0].real
    ph = np.exp(-2j * np.pi * F_CW * tl)
    U, K = np.sum(late[:, 2] * ph), np.sum(late[:, 1] * ph)   # each pulse train's component at the CW alias
    out["control"] = {"gain": 0.35, "on_year": 1960.0,
                      "residual_rms_after_1990_over_uncontrolled": float(np.std(p_ctl[dev.TG > 1990]) / np.std(p_free[dev.TG > 1990])),
                      "cw_component_u_over_k": [float(abs(U / K)), float(np.degrees(np.angle(U / K)))],
                      "pulse_rms_u_over_k": float(np.sqrt(np.mean(np.abs(late[:, 2]) ** 2)) / np.sqrt(np.mean(np.abs(late[:, 1]) ** 2)))}
    (HERE / "cw_counter_torque.json").write_text(json.dumps(out, indent=1, default=float))
    report(out)
    plot(tb, m_mas, out, p_free, p_ctl, u)


def report(out):
    print(f"torque per mas of excitation: {out['torque_per_mas_Nm']:.2e} N m; peak lunar bulge torque {out['lunar_bulge_torque_Nm']:.2e} N m")
    for q, v in out["lines"].items():
        print(q, {k: f"m {d['motion_mas']:.1f} mas -> psi {d['excitation_mas']:.2f} mas = {d['counter_torque_Nm']:.1e} N m" for k, d in v.items()})
    print("pulse budget:", {k: f"{v:.3g}" for k, v in out["pulse_budget"].items()})
    for k, v in out.get("fitted_bandpasses", {}).items():
        print(f"fitted bandpass {k}: f0 {v['f0']:.4f} Q {v['Q']:.1f}", {kk: f"{vv:.1e}" for kk, vv in v["counter_torque_Nm"].items()},
              f"CW/lunar {v['cw_over_lunar']:.1e}")
    print("control:", out["control"])


def plot(tb, m_mas, out, p_free, p_ctl, u):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0.06, 0.58, 0.42, 0.36])
    for q, col in ((85, "#d94a4a"),):
        psi = excitation(tb, m_mas, q)
        for z, lab, c2 in ((m_mas, "polar motion m", "k"), (psi, f"required excitation ψ (Q = {q})", col)):
            zz = (z - z.mean()) * np.hanning(len(z)); nf = len(z) * 8
            f = np.fft.fftshift(np.fft.fftfreq(nf, 0.05)); P = np.abs(np.fft.fftshift(np.fft.fft(zz, nf))) / len(z) * 4
            ax.semilogy(f, P, color=c2, lw=1, label=lab)
    for v, lab in ((F_CW, "CW"), (F_MM, "Mm"), (1, "annual"), (-1, "retro\nannual"), (2, "semi")):
        ax.axvline(v, ls=":", color="g", lw=0.8); ax.text(v + 0.02, 300, lab, fontsize=7, color="g", va="top")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(1e-2, 400); ax.set_xlabel("frequency (cycles/yr; + prograde)")
    ax.set_ylabel("amplitude (mas)"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(a) Running the plant backwards: the excitation that produced the observed wobble", fontsize=10)

    ax = fig.add_axes([0.56, 0.58, 0.4, 0.36]); ax.axis("off")
    L = out["lines"]["Q=85"]; tpm = out["torque_per_mas_Nm"]; pb = out["pulse_budget"]
    rows = [("line", "motion (mas)", "excitation (mas)", "counter-torque (N m)")]
    for k, lab in ((f"{F_CW:+.4f}", "CW 0.8440"), (f"{F_MM:+.4f}", "Mm 0.7448"), ("+1.0000", "annual (pro)"),
                   ("-1.0000", "annual (retro)"), ("+2.0000", "semiannual")):
        d = L[k]; rows.append((lab, f"{d['motion_mas']:.1f}", f"{d['excitation_mas']:.2f}", f"{d['counter_torque_Nm']:.1e}"))
    tb_ = ax.table(cellText=rows, loc="upper center", cellLoc="center"); tb_.auto_set_font_size(False); tb_.set_fontsize(9); tb_.scale(1, 1.5)
    ax.text(0, 0.28, f"Peak lunar torque on the bulge: {out['lunar_bulge_torque_Nm']:.1e} N m\n"
                     f"CW counter-torque / lunar torque: {L[f'{F_CW:+.4f}']['counter_torque_Nm'] / out['lunar_bulge_torque_Nm']:.1e}\n"
                     f"As semiannual pulses: {pb['per_pulse_mas']:.1f} mas per pulse = {pb['angular_impulse_Nms']:.1e} N m s\n"
                     f"= {pb['hours_of_peak_lunar_torque']:.1f} h of peak lunar torque per half year", fontsize=9,
            transform=ax.transAxes, va="top")
    ax.set_title("(b) Counter-torque budget, raw IERS 1973–2013, Q = 85", fontsize=10)

    ax = fig.add_axes([0.06, 0.08, 0.55, 0.38])
    s = 1 / np.std(p_free[dev.TG > 1935].real)
    m = dev.TG > 1930
    ax.plot(dev.TG[m], p_free.real[m] * s, color="0.6", lw=0.8, label="device, no control")
    ax.plot(dev.TG[m], p_ctl.real[m] * s, color="#d94a4a", lw=1, label="pulsed counter-torque from 1960 (gain 0.35)")
    ax.axvline(1960, color="k", ls=":"); ax.legend(fontsize=8, loc="upper right")
    c = out["control"]
    ax.set_title(f"(c) A sampled controller at the comb instants removes the wobble (residual {100 * c['residual_rms_after_1990_over_uncontrolled']:.1f}% "
                 f"after 1990)", fontsize=10)
    ax.set_xlabel("year"); ax.set_ylabel("m_x (normalized)")

    ax = fig.add_axes([0.68, 0.08, 0.28, 0.38])
    late = u[u[:, 0].real > 1990]
    tl = late[:, 0].real
    def comp(x):
        fr = np.linspace(0.02, 1.0, 981); return fr, np.abs([np.sum(x * np.exp(-2j * np.pi * f * tl)) for f in fr]) / len(x)
    fr, Ak = comp(late[:, 1]); _, Au = comp(late[:, 2])
    ax.plot(fr, Ak, color="0.5", lw=1, label="lunar kicks (B² on the comb)")
    ax.plot(fr, Au, color="#4a90d9", lw=1, label="controller pulses")
    ax.axvline(F_CW, color="g", ls=":", lw=0.8); ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("frequency of the pulse train (cycles/yr, folded)"); ax.set_ylabel("component amplitude")
    r, ang = c["cw_component_u_over_k"]
    ax.set_title(f"(d) Steady state: at the CW alias the controller's pulse train\nequals the lunar one reversed "
                 f"(ratio {r:.2f} at {ang:.0f}°)", fontsize=10)
    fig.savefig(HERE / "cw_counter_torque.png", dpi=115); plt.close(fig)


if __name__ == "__main__":
    main()
