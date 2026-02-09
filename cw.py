import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

# uniform interpolation
# choose a finer dt
dt = 0.1  # for example

# create uniform time grid
T_uniform = np.arange(T2[0], T2[-1], dt)

# interpolate your solution onto this grid
px2_uniform = np.interp(T_uniform, T2, px2)

# now compute FFT
freq = np.fft.rfftfreq(len(px2_uniform), dt)
spec = np.abs(np.fft.rfft(px2_uniform - np.mean(px2_uniform)))**2


# FFT
#dt = np.mean(np.diff(T2))
#freq = np.fft.rfftfreq(len(px2), dt)
##spec = np.abs(np.fft.rfft(px2 - np.mean(px2)))
#spec = np.abs(np.fft.rfft(px2 - np.mean(px2))) ** 2


# -----------------------------
# Plots
# -----------------------------

plt.figure(figsize=(12,4))

plt.subplot(2,1,1)
plt.plot(T2, px2, lw=0.8)
plt.xlabel("Time")
plt.ylabel("px")
plt.title("Emergent Polar Motion Amplitude")

plt.subplot(2,1,2)
plt.semilogy(freq, spec)
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
plt.legend()

plt.tight_layout()
plt.show()
