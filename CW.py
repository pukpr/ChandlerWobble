"""
Minimal Chandler Wobble simulation demonstrating aliasing mechanism.

This is the reference implementation from dialog.pdf showing how the Chandler
wobble emerges from stroboscopic sampling of a continuous off-resonant lunar
(draconic) forcing by impulsive parametric modulation of the inertia tensor.

The emergent slow frequency is the aliased difference:
    f_CW ≈ (1/2) * |ω_m - k*ω_s|
where ω_m is the "monthly" forcing, ω_s is the sampling (annual) frequency,
and the factor of 1/2 comes from quadratic coupling (π-symmetry at poles).
"""

from pathlib import Path
import warnings

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

CW_DATA_FILENAME = "cw.dat"
MIN_STD_THRESHOLD = 1e-10  # avoid inflating nearly-constant observational data
SAMPLING_UNIFORMITY_RTOL = 1e-3
SAMPLING_UNIFORMITY_ATOL = 1e-9

# -----------------------------
# Parameters (nondimensional)
# -----------------------------

omega0 = 1.0          # free nutation frequency (sets timescale)
zeta = 1e-3           # weak damping

omega_m = 30.0        # "draconic" forcing frequency (off-resonant)
eps_m = 2e-3          # forcing amplitude

Ts = 2 * np.pi * 13.0  # sampling period (annual analogue)
eps_s = 3e-3           # impulse strength

tmax = 0.04  # total integration time (use 4000.0 for full run)
dt_sample = 0.5       # output sampling

# -----------------------------
# ODE system between impulses
# -----------------------------


def rhs(t, y):
    px, py, vx, vy = y

    fx = eps_m * np.cos(omega_m * t)
    fy = eps_m * np.sin(omega_m * t)

    dpx = vx
    dpy = vy
    dvx = -2 * zeta * vx - omega0**2 * px + fx
    dvy = -2 * zeta * vy - omega0**2 * py + fy

    return [dpx, dpy, dvx, dvy]


# -----------------------------
# Impulse event
# -----------------------------


def impulse_event(t, y):
    return np.sin(np.pi * t / Ts)


impulse_event.terminal = True
impulse_event.direction = 0  # trigger on both rising/falling crossings

# -----------------------------
# Apply impulse map
# -----------------------------


def apply_impulse(y):
    px, py, vx, vy = y
    vx -= eps_s * omega0**2 * px
    vy -= eps_s * omega0**2 * py
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
        (t, tmax),
        y,
        events=impulse_event,
        max_step=0.2,
        rtol=1e-9,
        atol=1e-9,
    )

    # store solution
    T_hist.append(sol.t)
    P_hist.append(sol.y)

    # advance time
    t = sol.t[-1]
    y = sol.y[:, -1]

    # apply impulse if event triggered
    if sol.status == 1:
        y = apply_impulse(y)
        t = np.nextafter(t, tmax)

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
T2 = T[mask]
r2 = r[mask]

# Load observational Chandler wobble data and scale to simulation window
cw_path = Path(__file__).with_name(CW_DATA_FILENAME)
try:
    cw_data = np.loadtxt(cw_path)
except FileNotFoundError as exc:
    raise RuntimeError(f"Chandler wobble data file not found at {cw_path}.") from exc
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
cw_time_diffs = np.diff(cw_time)
if np.any(cw_time_diffs <= 0):
    raise RuntimeError(f"Chandler wobble data in {cw_path} must have increasing times.")
cw_time_step = np.mean(cw_time_diffs)
cw_time_step_std = np.std(cw_time_diffs)
if cw_time_step_std > max(SAMPLING_UNIFORMITY_ATOL, SAMPLING_UNIFORMITY_RTOL * cw_time_step):
    # FFT assumes uniform sampling; resample to avoid distorted frequency components.
    cw_time_uniform = np.linspace(cw_time[0], cw_time[-1], len(cw_time))
    cw_amp = np.interp(cw_time_uniform, cw_time, cw_amp)
    cw_time = cw_time_uniform
cw_time_span = cw_time[-1] - cw_time[0]
if cw_time_span <= 0:
    raise RuntimeError(f"Chandler wobble data in {cw_path} must have a non-zero time span.")
sim_time_span = T2[-1] - T2[0]
cw_time_normalized = (cw_time - cw_time[0]) / cw_time_span
cw_time_scaled = cw_time_normalized * sim_time_span + T2[0]
cw_amp_centered = cw_amp - np.mean(cw_amp)
sim_amp_mean = np.mean(r2)
sim_amp_std = np.std(r2)
cw_amp_std = np.std(cw_amp_centered)
if cw_amp_std > MIN_STD_THRESHOLD:
    # Scale observational amplitude to match simulation variance for comparison.
    cw_scale = sim_amp_std / cw_amp_std
    cw_amp_scaled = cw_amp_centered * cw_scale + sim_amp_mean
else:
    # Preserve a baseline trace when variance is near zero to avoid amplifying noise.
    warnings.warn(
        "Observational cw.dat variance is near zero; plotting a baseline at the simulation mean.",
        RuntimeWarning,
    )
    cw_amp_scaled = np.full_like(cw_amp_centered, sim_amp_mean)

# FFT
dt = np.mean(np.diff(T2))
freq = np.fft.rfftfreq(len(r2), dt)
spec = np.abs(np.fft.rfft(r2 - np.mean(r2)))

cw_dt = np.mean(np.diff(cw_time_scaled))
cw_freq = np.fft.rfftfreq(len(cw_amp_scaled), cw_dt)
cw_spec = np.abs(np.fft.rfft(cw_amp_scaled - np.mean(cw_amp_scaled)))

# -----------------------------
# Plots
# -----------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(T2, r2, lw=0.8, label="simulation")
plt.plot(cw_time_scaled, cw_amp_scaled, lw=0.8, alpha=0.7, label="cw.dat")
plt.xlabel("Time")
plt.ylabel("|p|")
plt.title("Emergent Polar Motion Amplitude")
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(freq, spec, label="simulation")
plt.semilogy(cw_freq, cw_spec, alpha=0.7, label="cw.dat")
plt.axvline(omega_m / (2 * np.pi), color="r", ls="--", label="forcing")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Spectrum")
plt.legend()

plt.tight_layout()
plt.show()
