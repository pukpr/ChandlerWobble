from pathlib import Path

import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import toeplitz
from scipy.signal import correlate
import matplotlib.pyplot as plt

CW_DATA_FILENAME = "cw.dat"
SAMPLING_UNIFORMITY_RTOL = 1e-3
SAMPLING_UNIFORMITY_ATOL = 1e-9
DEFAULT_AR_ORDER = 30


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
    return parser.parse_args()


def resolve_ar_order(order, sample_count):
    if sample_count < 2:
        raise RuntimeError("Not enough samples for autoregressive spectral estimation.")
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
    threshold = 1.0 - 1e-8
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
# Parameters (nondimensional)
# -----------------------------

# Chandler wobble emerges from aliasing between lunar draconic forcing 
# and annual/semi-annual inertia impulses. The period is:
# T_CW = (1/2) * |1/(1/T_d - 13/T_y)| ≈ 433 days
# where the factor of 1/2 arises from quadratic (π-symmetric) coupling.

Nutation = 455.0  # 443 460
Calendar = 365.242
Draconic = 27.2122
CW_Freq = (Calendar/Draconic-13.0)*2.0  # Aliased difference frequency
CW_Calculated = Calendar/CW_Freq 

#omega0 = 1.0          # free nutation frequency (sets timescale)
omega0 = Calendar/Nutation*2*np.pi          # free nutation frequency (sets timescale)
#zeta   = 1e-3         # weak damping
#zeta   = 5e-1         # strong damping
zeta   = 0.15         # 0.05 strong damping

#omega_m = 30.0        # "draconic" forcing frequency (off-resonant)
omega_m = Calendar/Draconic * 2*np.pi        
eps_m   = 2e-2        # forcing amplitude
N       = 2 #40
M       = 2
Semi    = 1.0
Inertial_load = 0.8

#Ts = 2*np.pi * 13.0   # sampling period (annual analogue)
Ts = 1.0              # sampling period (annual analogue)
eps_s = 0.1          # impulse strength
#eps_s = 4e-6          # impulse strength

#tmax = 40000.0         # total integration time
tmax = 130.0         # total integration time
#dt_sample = 0.5       # output sampling
#dt_sample = 0.05       # output sampling

# -----------------------------
# ODE system between impulses
# -----------------------------

def rhs(t, y):
    px, py, vx, vy = y

    fx = eps_m * (np.cos(omega_m * t))**N * (1.0 + 0.0*(np.sin(np.pi * t / Ts))**M)
    fy = 0  # eps_m * (np.sin(omega_m * t))**N

    dpx = vx
    dpy = vy
    dvx = -2*zeta*vx - omega0**2 * px + fx
    dvy = -2*zeta*vy - omega0**2 * py + fy

    return [dpx, dpy, dvx, dvy]

# -----------------------------
# Impulse event
# -----------------------------

def impulse_event(t, y):
    #return np.sin(2*np.pi * t / Ts)
    return np.sin(np.pi * t / Ts)

impulse_event.terminal = True
impulse_event.direction = 0

# -----------------------------
# Apply impulse map
# -----------------------------
state = True
def apply_impulse(y,t):
    px, py, vx, vy = y
    global state
    if state:
       vx -= eps_s * omega0**2 * px * (1+Inertial_load*(np.cos(omega_m * t))**M)
       vy -= 0.0 # eps_s * omega0**2 * py * (1+Inertial_load*(np.cos(omega_m * t))**M)
    else:
       vx -= Semi*eps_s * omega0**2 * px * (1+Inertial_load*(np.cos(omega_m * t))**M)
       vy -= 0.0 # Semi*eps_s * omega0**2 * py * (1+Inertial_load*(np.cos(omega_m * t))**M)
    state = not state
    return np.array([px, py, vx, vy])

# -----------------------------
# Time integration loop
# -----------------------------

t = 0.0
y = np.array([1e-3, 0.0, 0.0, 0.0])  # small initial tilt

T_hist = []
P_hist = []

while t < tmax:
    sol = solve_ivp(
        rhs,
        (t + 1e-10, tmax),
        y,
        events=impulse_event,
        max_step=0.2,
        rtol=1e-9, # -9
        atol=1e-9
    )

    # store solution
    T_hist.append(sol.t)
    P_hist.append(sol.y)

    # advance time
    t = sol.t[-1]
    y = sol.y[:, -1]

    # apply impulse if event triggered
    if sol.status == 1:
        y = apply_impulse(y,t)
        print(f"[{y[0]}] in iteration {t}")

print(f"CW={CW_Calculated}")

# concatenate results
T = np.concatenate(T_hist)
P = np.hstack(P_hist)

px, py = P[0], P[1]

# -----------------------------
# Diagnostics
# -----------------------------

r = np.sqrt(px**2 + py**2)

# Remove initial transient
mask = T > 0.2 * tmax
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

plt.figure(figsize=(12,4))

plt.subplot(2,1,1)
plt.plot(T2-3.8, px2*100.0 - 0.02, lw=0.8, label="simulation")
plt.plot(cw_time_aligned, cw_amp, lw=0.8, alpha=0.6, label="cw.dat")
plt.xlabel("Time")
plt.ylabel("px")
plt.title("Emergent Polar Motion Amplitude")
plt.legend()

plt.subplot(2,1,2)
plt.semilogy(freq, spec, label="simulation")
plt.semilogy(cw_freq, cw_spec/10000.0, alpha=0.7, label="cw.dat")
#plt.plot(freq, spec)
plt.ylim(0.0000001, 0.1)
plt.xlim(0.01, 2)
plt.axvline(omega_m/(2*np.pi), color='b', label="draconic forcing")
plt.axvline(1.0, color='g', ls=':', label="annual")
plt.axvline(0.0+CW_Freq, color='r', ls='-.', label="CW")
plt.axvline(1.0-CW_Freq, color='r', ls='--', label="CW sideband (annual-CW)")
plt.axvline(2.0-CW_Freq, color='r', ls='--', label="CW sideband (2*annual-CW)")
plt.axvline(1.0+CW_Freq, color='r', ls=':', label="CW sideband (annual+CW)")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Spectrum")
plt.legend(fontsize=8, loc="lower center")

plt.tight_layout()
plt.show()
