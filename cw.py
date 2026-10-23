from pathlib import Path

import argparse
import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import correlate
import matplotlib.pyplot as plt
from cw_physics import (
    CHANDLER_PERIOD_DAYS,
    DEFAULT_Q_FACTOR,
    DRACONIC_PERIOD_DAYS,
    IMPULSE_CADENCE_YEARS,
    TROPICAL_YEAR_DAYS,
    forced_response,
)

CW_DATA_FILENAME = "cw.dat"
SAMPLING_UNIFORMITY_RTOL = 1e-3
SAMPLING_UNIFORMITY_ATOL = 1e-9
DEFAULT_AR_ORDER = 30
STABILITY_MARGIN = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chandler wobble simulation with spectral estimators."
    )
    parser.add_argument(
        "--spectral-estimator",
        choices=["fft", "mem", "arz"],
        default="fft",
        help="Spectral estimator to use (fft, mem/Burg, or stabilized AR-Z).",
    )
    parser.add_argument(
        "--ar-order",
        type=int,
        default=DEFAULT_AR_ORDER,
        help="Autoregressive order for MEM/AR-Z estimators.",
    )
    parser.add_argument(
        "--phase-years",
        type=float,
        default=0.0,
        help="Draconic forcing phase in years.",
    )
    parser.add_argument(
        "--impulse-scale",
        type=float,
        default=0.1,
        help="Momentum increment that scales each normalized impulse.",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=DEFAULT_Q_FACTOR,
        help="Oscillator quality factor.",
    )
    parser.add_argument(
        "--amp-step",
        type=float,
        default=0.0,
        help="Inertia-pulse amplitude step: whole-year events scale the state by "
             "1+s, half-year events by 1/(1+s). 0.25 matches the observed sidebands.",
    )
    parser.add_argument(
        "--amp-step-swap",
        action="store_true",
        help="Make the half-year events the enlarging ones.",
    )
    return parser.parse_args()


def resolve_ar_order(order, sample_count):
    if sample_count < 2:
        raise RuntimeError(
            f"At least 2 samples required for spectral estimation, got {sample_count}."
        )
    if order < 1:
        raise ValueError("AR order must be at least 1.")
    return min(order, sample_count - 1)


def compute_fft_spectrum(signal, dt):
    signal = signal - np.mean(signal)
    freq = np.fft.rfftfreq(len(signal), dt)
    power = np.abs(np.fft.rfft(signal)) ** 2
    return freq, power


def burg_ar(signal, order):
    signal = np.asarray(signal, dtype=float)
    n_samples = len(signal)
    if order >= n_samples:
        raise ValueError("AR order must be less than the number of samples.")
    ef = signal[1:].copy()
    eb = signal[:-1].copy()
    ar_coeffs = np.zeros(order + 1, dtype=float)
    ar_coeffs[0] = 1.0
    error = np.dot(signal, signal) / n_samples
    for k in range(1, order + 1):
        num = -2.0 * np.dot(eb, ef)
        den = np.dot(ef, ef) + np.dot(eb, eb)
        if den <= 0:
            break
        reflection = num / den
        prev_coeffs = ar_coeffs[:k].copy()
        ar_coeffs[1:k] = prev_coeffs[1:k] + reflection * prev_coeffs[k - 1 : 0 : -1]
        ar_coeffs[k] = reflection
        ef_new = ef + reflection * eb
        eb_new = eb + reflection * ef
        ef = ef_new[1:]
        eb = eb_new[:-1]
        error *= 1.0 - reflection**2
        if error <= 0:
            error = np.finfo(float).eps
            break
    return ar_coeffs[1:], error


def yule_walker_ar(signal, order):
    signal = np.asarray(signal, dtype=float)
    n_samples = len(signal)
    if order >= n_samples:
        raise ValueError("AR order must be less than the number of samples.")
    autocorr = correlate(signal, signal, mode="full", method="fft")[
        n_samples - 1 : n_samples + order
    ]
    autocorr = autocorr / n_samples
    toeplitz_matrix = toeplitz(autocorr[:-1])
    ar_coeffs = np.linalg.solve(toeplitz_matrix, -autocorr[1:])
    noise_var = autocorr[0] + np.dot(autocorr[1:], ar_coeffs)
    return ar_coeffs, noise_var, autocorr


def stabilize_ar_coeffs(ar_coeffs):
    if ar_coeffs.size == 0:
        return ar_coeffs
    roots = np.roots(np.concatenate(([1.0], ar_coeffs)))
    threshold = 1.0 - STABILITY_MARGIN
    radius = np.abs(roots)
    stabilized = np.where(radius >= threshold, threshold * roots / radius, roots)
    stabilized_poly = np.poly(stabilized)
    return np.real_if_close(stabilized_poly[1:])


def ar_spectrum(ar_coeffs, noise_var, dt, n_samples):
    freq = np.fft.rfftfreq(n_samples, dt)
    omega = 2.0 * np.pi * freq * dt
    if ar_coeffs.size:
        k = np.arange(1, ar_coeffs.size + 1)
        exp_matrix = np.exp(-1j * np.outer(omega, k))
        denom = np.abs(1.0 + exp_matrix @ ar_coeffs) ** 2
    else:
        denom = np.ones_like(freq)
    denom = np.maximum(denom, np.finfo(float).eps)
    noise_var = max(noise_var, np.finfo(float).eps)
    return freq, noise_var / denom


def compute_spectrum(signal, dt, estimator, ar_order):
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    if estimator == "fft":
        return compute_fft_spectrum(signal, dt)
    order = resolve_ar_order(ar_order, len(signal))
    if estimator == "mem":
        ar_coeffs, noise_var = burg_ar(signal, order)
        return ar_spectrum(ar_coeffs, noise_var, dt, len(signal))
    if estimator == "arz":
        ar_coeffs, noise_var, autocorr = yule_walker_ar(signal, order)
        ar_coeffs = stabilize_ar_coeffs(ar_coeffs)
        stabilized_noise = autocorr[0] + np.dot(autocorr[1:], ar_coeffs)
        if stabilized_noise > 0:
            noise_var = stabilized_noise
        return ar_spectrum(ar_coeffs, noise_var, dt, len(signal))
    raise ValueError(f"Unknown spectral estimator: {estimator}")


args = parse_args()

# -----------------------------
# Newtonian impulse-map model
# -----------------------------

# The squared draconic signal has twice the draconic frequency. Sampling it
# twice yearly aliases its 26.84 cycles/year component to the CW frequency.
DRACONIC_CYCLES_PER_YEAR = TROPICAL_YEAR_DAYS / DRACONIC_PERIOD_DAYS
CW_Freq = abs(2 * DRACONIC_CYCLES_PER_YEAR - 26.0)
CW_Calculated = TROPICAL_YEAR_DAYS / CW_Freq
TMAX_YEARS = 130.0
OUTPUT_STEP_YEARS = 0.01
T = np.arange(0.0, TMAX_YEARS + OUTPUT_STEP_YEARS, OUTPUT_STEP_YEARS)
px = forced_response(
    T,
    args.phase_years,
    impulse_scale=args.impulse_scale,
    q_factor=args.q,
    amp_step=args.amp_step,
    amp_step_swap=args.amp_step_swap,
)

print(f"Forced alias period={CW_Calculated:.3f} days")
print(f"Oscillator period={CHANDLER_PERIOD_DAYS:.3f} days; "
      f"Q={args.q:.1f}; impulse cadence={IMPULSE_CADENCE_YEARS:.1f} years; "
      f"amplitude step={args.amp_step:.3f}")

# -----------------------------
# Diagnostics
# -----------------------------

r = np.abs(px)

# Remove initial transient
mask = T > 0.2 * TMAX_YEARS
#mask = T > 0.5 * tmax
T2 = T[mask]
r2 = r[mask]
px2 = px[mask]

cw_path = Path(__file__).with_name(CW_DATA_FILENAME)
try:
    cw_data = np.loadtxt(cw_path)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Chandler wobble data file not found at {cw_path}. "
        "Ensure cw.dat is available alongside cw.py."
    ) from exc
except PermissionError as exc:
    raise RuntimeError(f"Permission denied reading Chandler wobble data at {cw_path}.") from exc
except ValueError as exc:
    raise RuntimeError(
        f"Unable to parse Chandler wobble data from {cw_path}. Ensure it contains numeric columns."
    ) from exc
if cw_data.ndim != 2 or cw_data.shape[1] < 2:
    raise RuntimeError(f"Chandler wobble data in {cw_path} must have at least two columns.")
if cw_data.shape[0] < 2:
    raise RuntimeError(f"Chandler wobble data in {cw_path} must have at least two rows.")
if T2.size < 2:
    raise RuntimeError("Simulation did not produce enough samples for comparison.")
cw_time = cw_data[:, 0]
cw_amp = cw_data[:, 1]
cw_time_start = cw_time[0]
cw_time_diffs = np.diff(cw_time)
if np.any(cw_time_diffs <= 0):
    raise RuntimeError(f"Chandler wobble data in {cw_path} must have increasing times.")
cw_time_step = np.mean(cw_time_diffs)
cw_time_step_std = np.std(cw_time_diffs)
# Resample only if std exceeds both absolute (ATOL) and relative (RTOL * mean_step) thresholds.
if cw_time_step_std > SAMPLING_UNIFORMITY_ATOL and cw_time_step_std > SAMPLING_UNIFORMITY_RTOL * cw_time_step:
    cw_time_uniform = np.linspace(cw_time[0], cw_time[-1], len(cw_time))
    cw_amp = np.interp(cw_time_uniform, cw_time, cw_amp)
    cw_time = cw_time_uniform
    cw_time_diffs = np.diff(cw_time)
    cw_time_step = np.mean(cw_time_diffs)
# cw_time_aligned keeps the observational cadence within the simulation window.
cw_time_aligned = (cw_time - cw_time_start) + T2[0]
if cw_time_aligned[-1] > T2[-1]:
    raise RuntimeError(
        "Simulation time window is shorter than the observational data after transient removal."
    )

# uniform interpolation (match observational cadence for FFT comparison)
dt = cw_time_step
T_uniform = np.arange(T2[0], T2[-1], dt)
px2_uniform = np.interp(T_uniform, T2, px2)

# now compute spectrum
freq, spec = compute_spectrum(px2_uniform, dt, args.spectral_estimator, args.ar_order)
cw_freq, cw_spec = compute_spectrum(cw_amp, dt, args.spectral_estimator, args.ar_order)

# -----------------------------
# Plots
# -----------------------------

plt.figure(figsize=(16,12))

plt.subplot(2,1,1)
plt.plot(T2, px2, lw=0.8, color='r', label="simulation")
plt.plot(cw_time_aligned, cw_amp, lw=0.8, label="cw.dat")
plt.xlabel("Time")
plt.ylabel("px")
plt.title("Emergent Polar Motion Amplitude")
plt.legend()

mask = (freq >= 0.01) & (freq <= 2.0)
maskcw = (cw_freq >= 0.01) & (cw_freq <= 2.0)

plt.subplot(2,1,2)
plt.semilogy(freq[mask], spec[mask], color='r', label="simulation")
lines1, labels1 = plt.gca().get_legend_handles_labels()  # Primary axis
# plt.ylim(spec.min(), spec.max())
plt_twin = plt.twinx()
plt_twin.semilogy(cw_freq[maskcw], cw_spec[maskcw], alpha=0.7, label="cw.dat")
#plt_twin.set_ylim(np.percentile(cw_spec.min(),90), np.percentile(cw_spec.max(),95))
#plt_twin.margins(y=0)
plt.xlim(0.01, 2)
#plt.axvline(omega_m/(2*np.pi), color='b', label="draconic forcing")
plt.axvline(1.0, color='g', label="annual")
plt.axvline(0.0+CW_Freq, color='g', ls='--', label="CW")
plt.axvline(1.0-CW_Freq, color='g', ls=':', label="CW sideband (annual-CW)")
plt.axvline(2.0-CW_Freq, color='g', ls=':', label="CW sideband (2*annual-CW)")
plt.axvline(1.0+CW_Freq, color='g', ls='-.', label="CW sideband (annual+CW)")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Amplitude Spectrum  -- Estimator:" +  args.spectral_estimator.upper())
# Get lines and labels from BOTH axes
lines2, labels2 = plt_twin.get_legend_handles_labels()   # Twin axis
# Combine and create legend
plt.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
# plt.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.show()
