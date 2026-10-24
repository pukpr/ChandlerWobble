#!/usr/bin/env python3
"""Sidebands of the Chandler line: mixing with the carrier or resonance filtering? (2026-10-23)
Companion to cw_magnetized_spheroid.py; discussion in cw_data_tests.md, section 2 (the AM reading tested here applies to
cw.dat only: raw IERS polar motion does not show these sidebands).

Lines fitted jointly to cw.dat by least squares, with signed frequencies (a negative frequency is the folded image):
  carrier  f_c ~ 0.844        CW line
  f_c + 1, f_c - 1            annual sidebands   (observed at 1.844 and 0.156)
  f_c + 2, f_c - 2            semiannual sidebands (2.844 and 1.156)
  1, 2                        annual and semiannual wobble
Phase closure  D_k = phi(f_c + k) + phi(f_c - k) - 2 phi(f_c)  does not depend on the time origin, on the modulation
phase, or on a small error in f_c. Amplitude modulation of the carrier gives D = 0; phase (frequency) modulation gives
D = pi; additive forcing lines passed through the resonance give whatever the forcing and transfer phases make them.
    python3 cw_sidebands.py  -> cw_spheroid_sidebands.png, cw_sidebands.json
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

import cw_magnetized_spheroid as dev

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
T_REF = 2000.0
FC_PRED = dev.alias(2 * dev.YEAR / dev.MONTHS["draconic"], False)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def lines(t, x, fc):
    freqs = {"c": fc, "u1": fc + 1, "l1": fc - 1, "u2": fc + 2, "l2": fc - 2, "ann": 1.0, "semi": 2.0}
    cols = [np.ones_like(t)]
    for f in freqs.values():
        ph = 2 * np.pi * f * (t - T_REF); cols += [np.cos(ph), np.sin(ph)]
    G = np.column_stack(cols); c, *_ = np.linalg.lstsq(G, x, rcond=None)
    resid = x - G @ c
    out = {}
    for i, k in enumerate(freqs):
        a, b = c[1 + 2 * i], c[2 + 2 * i]               # a cos + b sin = A cos(ph + phi)
        out[k] = (float(np.hypot(a, b)), float(np.arctan2(-b, a)))
    D1 = wrap(out["u1"][1] + out["l1"][1] - 2 * out["c"][1])
    # AM: x = A cos(w t + phi)(1 + m cos 2pi(t - psi)), m > 0: phi_u = phi - 2pi psi, phi_l = phi + 2pi psi
    z = np.exp(1j * (out["c"][1] - out["u1"][1])) + np.exp(1j * (out["l1"][1] - out["c"][1]))
    out["psi"] = (float((np.angle(z) / (2 * np.pi)) % 1.0), 0.0)
    D2 = wrap(out["u2"][1] + out["l2"][1] - 2 * out["c"][1])
    return out, float(D1), float(D2), resid


def best_fc(t, x):
    grid = np.linspace(0.835, 0.852, 341)
    return float(grid[np.argmax([lines(t, x, f)[0]["c"][0] for f in grid])])


def summarize(t, x, fc=None, noise_mc=200, seed=0):
    fc = best_fc(t, x) if fc is None else fc
    L, D1, D2, resid = lines(t, x, fc)
    A = L["c"][0]
    r = dict(fc=fc, amp={k: v[0] for k, v in L.items()}, ratio_u1=L["u1"][0] / A, ratio_l1=L["l1"][0] / A,
             ratio_u2=L["u2"][0] / A, ratio_l2=L["l2"][0] / A, D1=D1, D2=D2, am_peak_day=L["psi"][0] * dev.YEAR)
    if noise_mc:                                           # phase-closure uncertainty from the residual (block bootstrap)
        rng = np.random.default_rng(seed); blk = 40; n = len(resid); fit = x - resid; d1, d2, dp = [], [], []
        for _ in range(noise_mc):
            starts = rng.integers(0, n - blk, n // blk + 1)
            rs = np.concatenate([resid[s:s + blk] for s in starts])[:n]
            Lb, a1, a2, _ = lines(t, fit + rs, fc); d1.append(a1); d2.append(a2); dp.append(2 * np.pi * Lb["psi"][0])
        circ = lambda d: float(np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * np.array(d))))))))
        r["D1_sd_deg"], r["D2_sd_deg"] = circ(d1), circ(d2); r["am_peak_day_sd"] = circ(dp) / 360 * dev.YEAR
    return r


# ---------------------------------------------------------------- mechanisms (all on the cw.dat sampling)
TD, XD = dev.TD, dev.XD
WIN = (dev.TG >= 1930)


def carrier_series():
    tn, k = dev.kicks(0.469, 0.0, "beta")
    p = dev.response_linear(tn, k)
    return p / np.std(p.real[dev.TG > 1935])


def sample(p):
    return np.interp(TD, dev.TG, p.real)


def mech_additive(a):
    tn, k = dev.kicks(0.469, a, "beta"); p = dev.response_linear(tn, k); return sample(p)


def mech_am(m, psi=0.3, m2=0.0):
    p = carrier_series(); t = dev.TG
    return sample(p * (1 + m * np.cos(2 * np.pi * (t - psi)) + m2 * np.cos(4 * np.pi * (t - psi))))


def mech_pm(beta, psi=0.3):
    p = carrier_series(); t = dev.TG
    return sample(p * np.exp(1j * beta * np.sin(2 * np.pi * (t - psi))))


def mech_phase_steps(eps):
    tn, north = dev.comb_times(0.469); _, k = dev.kicks(0.469, 0.0, "beta")
    return sample(dev.response_parametric(tn, k, np.where(north, eps, -eps), north))


def mech_amp_steps(eps):
    """Inertia pulses rescale the wobble amplitude instead of rotating its phase: p -> p (1 +- eps)."""
    tn, north = dev.comb_times(0.469); _, k = dev.kicks(0.469, 0.0, "beta")
    idx = np.minimum(np.round((tn - dev.TG[0]) / dev.DT).astype(int), len(dev.TG) - 1)
    step = np.exp((1j * dev.OMEGA - dev.LAM) * dev.DT)
    kick = np.zeros(len(dev.TG), complex); fac = np.ones(len(dev.TG))
    np.add.at(kick, idx, k); fac[idx] = np.where(north, 1 + eps, 1 / (1 + eps))
    p = np.empty(len(dev.TG), complex); z = 0j
    for i in range(len(dev.TG)):
        z = z * fac[i] + kick[i]; p[i] = z; z *= step
    return sample(p)


def mech_unfiltered(a):
    """The mixed forcing itself, kick train smoothed with a 0.25-yr Gaussian (no resonance)."""
    tn, k = dev.kicks(0.469, a, "beta")
    drive = np.zeros_like(dev.TG); np.add.at(drive, np.round((tn - dev.TG[0]) / dev.DT).astype(int), k)
    s = np.arange(-1, 1 + dev.DT, dev.DT); g = np.exp(-0.5 * (s / 0.25) ** 2)
    y = np.convolve(drive - drive.mean(), g / g.sum(), "same")
    return np.interp(TD, dev.TG, y)


def main():
    data = summarize(TD, XD, noise_mc=300)
    data_pred = summarize(TD, XD, fc=FC_PRED, noise_mc=0)
    halves = [summarize(TD[m], XD[m], noise_mc=0) for m in (TD < 1977, TD >= 1977)]
    obs_sb = 0.5 * (data["ratio_u1"] + data["ratio_l1"])

    # transfer function of the Kittel / Euler mode (one-sided, prograde) and of a real 2nd-order oscillator
    f0, Q = 1 / dev.T_FREE, dev.Q
    H1 = lambda f: 1 / (1j * (f - f0) + f0 / (2 * Q))
    H2 = lambda f: 1 / (f0 ** 2 - f ** 2 + 1j * f * f0 / Q)
    gain = {"kittel_sat_over_cw_amp": float(abs(H1(FC_PRED + 1)) / abs(H1(FC_PRED))),
            "kittel_lowsat_over_cw_amp": float(abs(H1(FC_PRED - 1)) / abs(H1(FC_PRED))),
            "osc2_upper_over_lower_power": float(abs(H2(FC_PRED + 1)) ** 2 / abs(H2(1 - FC_PRED)) ** 2)}

    mech = {}
    for a in (0.065, 0.13, 0.5, 1.0):
        mech[f"additive + resonance, a={a}"] = summarize(TD, mech_additive(a), noise_mc=0)
    for a in (0.065, 0.13):
        mech[f"mixed forcing, no resonance, a={a}"] = summarize(TD, mech_unfiltered(a), noise_mc=0)
    m_am = 2 * obs_sb
    mech[f"AM of CW, m={m_am:.3f}"] = summarize(TD, mech_am(m_am), noise_mc=0)
    mech[f"PM of CW, beta={m_am:.3f}"] = summarize(TD, mech_pm(m_am), noise_mc=0)
    mech["inertia phase steps +-0.2 rad"] = summarize(TD, mech_phase_steps(0.2), noise_mc=0)
    mech["inertia amplitude steps x(1.25)^+-1"] = summarize(TD, mech_amp_steps(0.25), noise_mc=0)
    for psi in (0.0, 0.25, 0.6):                          # closure is independent of the modulation phase
        mech[f"AM check psi={psi}"] = summarize(TD, mech_am(m_am, psi), noise_mc=0)

    # N/S balance implied by the observed sideband level under each reading
    balance = {
        "multiplicative (CW x (1 + a cos 2pi t)): a = 2 x sideband/carrier": 2 * obs_sb,
        "additive mixing seen without resonance: a = sideband/carrier": obs_sb,
        "additive mixing through the Q=85 resonance: a needed": obs_sb / gain["kittel_sat_over_cw_amp"],
    }
    out = dict(predicted_fc=FC_PRED, data=data, data_at_predicted_fc=data_pred,
               halves={"1935-1977": halves[0], "1977-2018": halves[1]}, transfer=gain, mechanisms=mech, balance=balance)
    (HERE / "cw_sidebands.json").write_text(json.dumps(out, indent=1))
    plot(out, m_am)
    rep = lambda r: (f"fc {r['fc']:.4f}  u1 {r['ratio_u1']:.4f} l1 {r['ratio_l1']:.4f}  u2 {r['ratio_u2']:.4f} "
                     f"l2 {r['ratio_l2']:.4f}  D1 {np.degrees(r['D1']):7.1f}  D2 {np.degrees(r['D2']):7.1f}"
                     + f"  AM peak day {r['am_peak_day']:.0f}"
                     + (f"  (sd {r['D1_sd_deg']:.0f}, {r['D2_sd_deg']:.0f}, day {r['am_peak_day_sd']:.0f})" if "D1_sd_deg" in r else ""))
    print("data            ", rep(data)); print("data @ predicted", rep(data_pred))
    for k, h in out["halves"].items(): print(f"{k:16s}", rep(h))
    print("transfer:", gain)
    for k, v in mech.items(): print(f"{k:40s}", rep(v))
    print("N/S balance:", {k: round(v, 3) for k, v in balance.items()})


def plot(out, m_am):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6))
    d = out["data"]
    ax = axs[0]
    rows = [("data", d, "k")] + [(k, v, c) for (k, v), c in zip(
        [(k, out["mechanisms"][k]) for k in (f"additive + resonance, a=1.0", "mixed forcing, no resonance, a=0.065",
                                             "inertia amplitude steps x(1.25)^+-1", "inertia phase steps +-0.2 rad")],
        ["#d94a4a", "#e67e22", "#4a90d9", "#2e8b57"])]
    labels = ["f−2\n(1.156)", "f−1\n(0.156)", "f\n(0.844)", "f+1\n(1.844)", "f+2\n(2.844)"]
    for j, (name, r, col) in enumerate(rows):
        v = [r["ratio_l2"], r["ratio_l1"], 1.0, r["ratio_u1"], r["ratio_u2"]]
        ax.semilogy(np.arange(5) + (j - 2) * 0.08, np.maximum(np.square(v), 1e-9), "o", color=col, ms=7 if j == 0 else 5,
                    label=name)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(1e-9, 3)
    ax.set_ylabel("line power / CW power"); ax.legend(fontsize=7, loc="lower center")
    ax.set_title("(a) Line powers: data against four mechanisms", fontsize=10)

    ax = axs[1]
    a = np.linspace(0, 1, 200); g = out["transfer"]["kittel_sat_over_cw_amp"]
    ax.semilogy(a, (a / 2) ** 2, color="#4a90d9", label="multiplicative: CW × (1 + a cos 2πt), (a/2)²")
    ax.semilogy(a, a ** 2, color="#e67e22", label="mixed forcing, no resonance: a²")
    ax.semilogy(a, (a * g) ** 2, color="#d94a4a", label=f"mixed forcing through Q = 85 resonance: (a·{g:.4f})²")
    sb = 0.5 * (d["ratio_u1"] ** 2 + d["ratio_l1"] ** 2)
    ax.axhline(sb, color="k", ls="--", lw=1); ax.text(0.02, sb * 1.3, f"observed {sb:.1e}", fontsize=8)
    for k, v in out["balance"].items():
        if v <= 1: ax.axvline(v, color="0.6", ls=":", lw=1)
    ax.set_xlabel("north/south imbalance a  (pulses 1 ± a)"); ax.set_ylabel("satellite power / CW power")
    ax.set_ylim(1e-9, 2); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(b) The satellites measure the N/S balance — if they are mixing products", fontsize=10)

    ax = axs[2]
    ax.axhspan(-15, 15, color="#4a90d9", alpha=0.12); ax.axhspan(165, 195, color="#2e8b57", alpha=0.12)
    ax.text(0.02, 2, "AM: D = 0", fontsize=8, color="#4a90d9", transform=ax.get_yaxis_transform())
    ax.text(0.02, 172, "PM: D = 180°", fontsize=8, color="#2e8b57", transform=ax.get_yaxis_transform())
    pts = [("data", d, "k")] + [(k, out["halves"][k], "0.5") for k in out["halves"]] + \
          [(k, out["mechanisms"][k], c) for k, c in [("additive + resonance, a=1.0", "#d94a4a"),
                                                       ("mixed forcing, no resonance, a=0.065", "#e67e22"),
                                                       (f"AM of CW, m={m_am:.3f}", "#4a90d9"),
                                                       (f"PM of CW, beta={m_am:.3f}", "#2e8b57"),
                                                       ("inertia phase steps +-0.2 rad", "#2e8b57"),
                                                       ("inertia amplitude steps x(1.25)^+-1", "#4a90d9")]]
    for i, (name, r, col) in enumerate(pts):
        D = np.degrees(r["D1"]) % 360
        D = D - 360 if D > 270 else D
        ax.errorbar(i, D, yerr=r.get("D1_sd_deg", 0), fmt="o", color=col, ms=7 if name == "data" else 5, capsize=3)
    ax.set_xticks(range(len(pts))); ax.set_xticklabels([p[0].replace(", ", "\n") for p in pts], rotation=60, fontsize=7, ha="right")
    ax.set_ylim(-90, 270); ax.set_ylabel("phase closure D₁ = φ(f+1) + φ(f−1) − 2φ(f)  (deg)")
    ax.set_title("(c) Phase closure of the annual sidebands", fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "cw_spheroid_sidebands.png", dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
