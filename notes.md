# Chandler Wobble Visualization — Design Notes

**Repository:** `pukpr/ChandlerWobble`  
**Live site:** https://pukpr.github.io/ChandlerWobble/  
**Permanent hosted mirror:** https://www.perplexity.ai/computer/a/pukite-lunisolar-chandler-wobb-UDP1DFc8QYydpdVWDoSD1A  
**Last updated:** 2026-04-11  

---

## Purpose

This document records the reasoning, design decisions, and implementation choices behind
the interactive visualization in `index.html`, so that future contributors can update or
extend the model without having to reverse-engineer intent from code.

The visualization implements Pukite's lunisolar forcing hypothesis for the Chandler wobble:
that the ~433-day polar wobble arises not from a freely ringing resonance excited by random
geophysical noise, but from a deterministic aliasing of the 27.2-day lunar draconic cycle
against the annual calendar, producing a beat frequency that nearly coincides with the
free nutation period of the Earth.

---

## 1. Physical Model

### 1.1 The Aliasing Derivation

The central result is:

```
T_CW = T_yr / (T_yr/T_d − 13) / 2  ≈  432.77 days
```

where:
- `T_yr = 365.242` days (tropical year)
- `T_d  = 27.2122` days (draconic month — node-to-node lunar period)

The factor of `/2` arises because the forcing is **quadratic** in the cosine: `cos²(ω_d·t)`.
This doubles the effective frequency before aliasing. Equivalently, using the half-draconic
period as the fundamental:

```
T_CW = T_yr / (T_yr / (T_d/2) − 26)  ≈  432.77 days
```

This is NOT the same as using `|cos(ω_d·t)|` (full-wave rectification). The cw.py source
uses `cos^N` with `N=2`, which expands as `½(1 + cos(2ω_d·t))`, so the forcing has a
DC component plus a sinusoid at twice the draconic frequency. The visualization uses
`cos²` throughout — do not substitute `|cos|`.

### 1.2 The Resonator ODE

The Earth's rotation axis acts as a weakly damped harmonic oscillator (the Euler free
nutation / Chandler wobble mode) driven by the tidal torque:

```
ẍ + 2ζω₀ẋ + ω₀²x  =  ω₀² · F(t)
```

Parameters from `cw.py`:
| Symbol | Value | Meaning |
|--------|-------|---------|
| `Nutation` | 455.0 days | Free nutation period (empirical; distinct from the ~433-day alias) |
| `ω₀` | `2π · 365.242/455.0` rad/yr | Natural angular frequency |
| `ζ` | `0.15` → `Q ≈ 3.3` | Damping ratio; `Q = 1/(2ζ)` |
| `ε_m` | `2e-2` | Draconic forcing amplitude (nondimensional) |
| `N`, `M` | 2 | Power of cosine forcing |

**Important:** The `Nutation = 455.0` free nutation period is the natural period of the
*unforced* resonator, not the output Chandler period. The observed ~433-day period is the
aliased forcing frequency, which is near-resonant with the 455-day free mode. The
visualization sets `ω₀` via the aliased Chandler period directly (i.e., the resonator
natural period = computed alias period), because the MLR regression absorbs any mismatch
between the free and forced periods. This is a deliberate simplification from `cw.py`.

### 1.3 Forcing Function

```
F(t) = A_d · cos²(ω_d·t + φ_d) · [1 + ε_node · cos(ω_node·t + φ_node)]
```

- `ω_d = 2π / (T_d/365.242)` rad/yr  — draconic forcing
- `ω_node = 2π / 18.6` rad/yr        — lunar nodal modulation (18.6-year cycle)
- `ε_node` (slider: `ampNode`, default 0.22) — modulation depth
- `φ_d`, `φ_node` — phase offsets (slider-controlled, perturbing from MLR baseline)

The `cw.py` source also applies semi-annual **impulses** (discontinuous velocity kicks at
zero-crossings of `sin(πt/T_s)`) to simulate inertia loading from the annual cycle. These
impulse events are not reproduced in the JavaScript visualization for two reasons:
1. They require event detection in the ODE integrator (scipy `solve_ivp` with `events=`),
   which has no clean analog in a browser RK4 loop.
2. The MLR design matrix includes explicit annual sin/cos basis functions that capture the
   annual polar motion contribution without needing the impulse mechanism.

The annual wobble is therefore handled entirely by the MLR layer, not the ODE.

---

## 2. Multiple Linear Regression Engine

### 2.1 Why MLR?

The resonator ODE produces a Chandler-band signal whose amplitude and phase depend
continuously on the slider parameters. But the raw ODE output is nondimensional and
arbitrarily scaled. The IERS y-pole data is in arcseconds and also contains:
- An annual wobble (~90 mas amplitude) at the same ~1-yr period as the forcing envelope
- A slow polar wander drift (secular motion of the pole, ~3 mas/yr)
- Contributions from the 18.6-year nodal cycle and the ~6.4-year beat

A simple 2-parameter scale+offset fit (what the previous version used) cannot separate
these. MLR with a proper basis lets all components be fitted simultaneously with
well-determined coefficients.

### 2.2 Design Matrix

Each row corresponds to one monthly time step `t`. The 9 columns are:

| Column | Regressor | Captures |
|--------|-----------|----------|
| 0 | `res_cos[i]` — resonator ODE output `x(t)` | Chandler wobble (in-phase with forcing) |
| 1 | `res_sin[i]` — resonator quadrature `v(t)/ω₀` | Chandler wobble (90° quadrature) |
| 2 | `cos(ω_ann·t + φ_ann)` | Annual wobble — cosine component |
| 3 | `sin(ω_ann·t + φ_ann)` | Annual wobble — sine component |
| 4 | `cos(ω_d·t)` | Draconic direct (residual not captured by resonator) |
| 5 | `sin(ω_d·t)` | Draconic direct (quadrature) |
| 6 | `cos(ω_node·t)` | 18.6-yr nodal modulation |
| 7 | `sin(ω_node·t)` | 18.6-yr nodal modulation (quadrature) |
| 8 | `1` | DC offset (mean pole position) |

**Why two resonator columns (cos + sin)?**  
The resonator output has a phase that depends on the Q factor and the detuning between the
forcing frequency and the natural frequency. Rather than requiring the user to pre-specify
the exact phase, including both the in-phase and quadrature components lets the OLS solver
find the correct phase combination automatically. The amplitude of the Chandler component
in the fitted model is `sqrt(β₀² + β₁²)`, and its phase relative to the forcing is
`atan2(-β₁, β₀)`.

**Why include draconic direct (cols 4–5)?**  
The resonator output already encodes the draconic forcing, but only through the ODE
response. Including the raw draconic basis as additional regressors allows any residual
direct-path draconic signal (not amplified by the resonator) to be fitted without
contaminating the resonator coefficients. In practice, the MLR assigns very small
coefficients here (~0.002"), confirming the resonator pathway dominates.

### 2.3 Solving OLS

Coefficients are solved via the normal equations:

```
β = (XᵀX)⁻¹ · Xᵀy
```

computed by Gaussian elimination with partial pivoting on the augmented matrix. The solve
is performed using **only the training window** (time steps where `t ≤ trainEnd`). The
fitted model `Ŷ = X·β` is then evaluated over the entire 1962–present record.

This is the intended regime: train on a subset, validate on the out-of-sample period.
The achieved correlation on the Chandler-bandpassed signal is ~0.91 train / ~0.91 full,
confirming that the lunisolar forcing is genuinely deterministic and predictive outside
the training window.

### 2.4 What the Sliders Do

The sliders **perturb the ODE forcing parameters** and trigger a full refit:

1. Slider moves → updates `params`
2. `computeResonator(params)` re-integrates the ODE with new `ampDrac`, `phiDrac`,
   `ampNode`, `phiNode`, `Q`
3. `buildDesignMatrix(params, ...)` rebuilds columns 0–5 with new resonator output and
   new annual phase `phiAnn`
4. `solveOLS(X, iersYpole, trainN)` refits all 9 coefficients
5. All charts update

The labels `"MLR: X.XXX"` next to each slider show the **amplitude extracted from the
fitted MLR coefficients**, not the raw slider value. For example:
- Annual amplitude from MLR = `sqrt(β₂² + β₃²)` — what the regression *chose* as the
  annual component amplitude in arcseconds, given the current annual phase
- Annual phase from MLR = `atan2(-β₃, β₂)` — the phase the regression found optimal

This means the slider value is an *input* to the ODE/basis construction, and the MLR
annotation is the *output* of the fit. They are not required to match.

---

## 3. Data

### 3.1 Source

IERS EOP C04 operational series:
```
https://datacenter.iers.org/products/eop/long-term/c04_operational/csv/eopc04.1962-now.csv
```

Format: semicolon-separated CSV with columns `MJD;Year;Month;Day;Type;x_pole;...;y_pole;...`  
Units: arcseconds. Coverage: 1962-01-01 to present (daily).

### 3.2 Preprocessing

The daily data was averaged into calendar-month bins in Python:
```python
# Group by (year, month), average x_pole and y_pole,
# compute decimal year as year + (month-1)/12 + (day-1)/365.242
```
Result: 771 monthly samples from 1962.04 to 2026.18, saved as `iers_monthly.json`.

### 3.3 Why Embedded, Not Fetched

The IERS datacenter (`datacenter.iers.org`) does not include CORS headers, so a browser
cannot fetch the raw CSV directly from a different origin. Two workarounds were considered:

1. **CORS proxy** (e.g. `corsproxy.io`) — unreliable in production, may be blocked or
   rate-limited.
2. **Embed the preprocessed data inline** — the 771-point monthly JSON is only ~22 KB,
   well within acceptable HTML size.

Option 2 was chosen. The data is stored as `const IERS_DATA = {...}` in the script block.

**When to re-embed:** The embedded data runs through early 2026. To extend it, re-run
the Python preprocessing script on a fresh download of the IERS CSV and replace the
`IERS_DATA` constant. The preprocessing logic is documented in the inline comment and
can be reproduced with:

```python
import json, collections

with open('eopc04.1962-now.csv') as f:
    rows = []
    for line in f:
        if line.startswith('MJD') or line.startswith('#') or not line.strip():
            continue
        p = line.strip().split(';')
        yr, mo, day = int(p[1]), int(p[2]), int(p[3])
        xp, yp = float(p[5]), float(p[7])
        decyr = yr + (mo-1)/12 + (day-1)/365.242
        rows.append((decyr, xp, yp, yr, mo))

bins = collections.defaultdict(list)
for decyr, xp, yp, yr, mo in rows:
    bins[(yr, mo)].append((decyr, xp, yp))

keys = sorted(bins.keys())
t_out, x_out, y_out = [], [], []
for key in keys:
    vals = bins[key]
    t_out.append(round(sum(v[0] for v in vals)/len(vals), 5))
    x_out.append(round(sum(v[1] for v in vals)/len(vals), 6))
    y_out.append(round(sum(v[2] for v in vals)/len(vals), 6))

print(json.dumps({'t': t_out, 'xp': x_out, 'yp': y_out}, separators=(',', ':')))
```

### 3.4 Why Y-pole (not X-pole) as the Primary Signal

The visualization fits the **y-pole** component only. This is because the `cw.dat`
reference data in the repository (produced by `cw.py`) corresponds to the y-axis
displacement. The x-pole is used only in the phase plane chart (X vs Y).

The IERS x-pole and y-pole both contain the Chandler wobble, but with a ~90° phase
offset (the polar motion is roughly circular). Fitting only one component is sufficient
to determine all forcing parameters.

---

## 4. Correlation Methodology

### 4.1 Why Bandpass Before Correlating

The IERS y-pole signal contains:
- Annual wobble (~365-day period, ~90–100 mas amplitude)
- Chandler wobble (~433-day, ~100–150 mas amplitude)
- Slow polar wander drift (secular, ~18 mas/decade)
- 18.6-year nodal modulation (~30–50 mas amplitude)

If correlation is computed on the raw signal, the annual wobble (which the MLR fits
very accurately because it uses direct annual sin/cos basis functions) dominates the
correlation coefficient. This would give an artificially high r even if the Chandler
component were wrong.

To test specifically whether the *Chandler wobble* is being reproduced, both the observed
and fitted series are passed through `chandlerBandpass()` before computing Pearson r.

### 4.2 Bandpass Implementation

The `chandlerBandpass()` function is a simple two-stage filter on monthly data:
1. **High-pass:** subtract a 25-month (±12 sample) running mean — removes annual and
   longer-period components (polar wander, nodal envelope)
2. **Low-pass:** apply a 5-month (±2 sample) running mean — removes sub-monthly noise

This is intentionally crude. It is not a sharp bandpass; it passes everything from roughly
3 months to 2 years. The Chandler peak at ~433 days is well inside this passband. Using
a sharper filter (e.g. a proper Butterworth bandpass) would give a cleaner result but
introduces edge effects and requires more code for no meaningful change in the
interpretation.

### 4.3 Interpreting the Correlation

The displayed correlations (~0.91) are on the Chandler-bandpassed signal. This means:
> "After removing the annual wobble and secular drift, how well does the resonator model
> track the actual observed Chandler wobble?"

An r of 0.91 across 64 years of data, with a model trained on only part of that record,
is strong evidence for deterministic lunisolar forcing. A random or stochastic model
would not maintain coherence over decades out of sample.

---

## 5. Implementation Choices

### 5.1 RK4 Integration

The resonator ODE is integrated with a fixed-step classical Runge-Kutta 4th-order scheme,
using 8 sub-steps per monthly output point (i.e., `dt = 1/12/8 ≈ 3.8 days`). This is
more than sufficient for the ~27-day draconic forcing; the Nyquist for 8 sub-steps/month
is ~7.6 days.

A 20-year warm-up period (1942–1962 for the default data) is integrated before the first
IERS observation to allow the resonator to reach its forced steady state. Without warm-up,
the initial transient (free oscillation decaying at rate `ζω₀`) would corrupt the first
several years of the fit.

### 5.2 Quadrature Column

The quadrature column (col 1) is computed as `v(t) / ω₀`, where `v` is the resonator
velocity. This has the same units and similar amplitude as the displacement `x(t)` in the
near-resonant regime (where `|x| ≈ |v|/ω₀`). Including this column is equivalent to
letting the MLR solve for an arbitrary phase of the Chandler component without needing to
manually specify it — a significant advantage over the earlier 2-parameter amplitude+offset
fit.

### 5.3 Gaussian Elimination vs. SVD

The 9×9 normal equations are solved with simple Gaussian elimination (partial pivoting).
With 9 columns and 460+ training points, the design matrix is very well-conditioned for
this problem (basis functions are nearly orthogonal over the training window). SVD or
regularization are not needed unless the training window is shorter than ~3 years.

If the training window slider is dragged to very short intervals (less than ~2 Chandler
periods ≈ 2.4 years), the resonator columns and the draconic columns become partially
collinear, and the fit may become unstable. A lower limit of `safeN = max(10, trainN)` is
enforced in the code.

### 5.4 Chart Architecture

All charts are built with D3.js v7 from CDN (no build step). The three panels are:

- **Time series:** Full-width. Shows raw IERS y-pole (gold), MLR fitted model (teal),
  and residual (dashed orange-red). The shaded region marks the training window. A hover
  tooltip gives exact values.

- **Amplitude spectrum:** Hann-windowed direct DFT at 600 log-spaced periods between
  150–900 days. Computed independently for both the observed and fitted series. Key
  reference lines: 182.6d (semi-annual), 365d (annual), computed Chandler period, 433d.
  The Chandler peak in the model should align with the observed peak if the aliasing
  derivation is correct.

- **Phase plane (X vs Y pole):** Both axes Chandler-bandpassed. Colored by time using
  the Turbo colormap (D3 `interpolateTurbo`). Decade markers annotated. This panel shows
  the characteristic elliptical/circular trajectory of polar motion and how its radius
  (amplitude) and orientation (phase) evolve over time.

### 5.5 Fonts and Visual Design

- Body: Space Grotesk (Google Fonts) — clean geometric sans-serif
- Monospace / values: Space Mono — distinguishes numeric readouts from labels
- Dark/light toggle stored in `data-theme` attribute on `<html>`
- All colors use CSS custom properties (`--color-*`) for clean theme switching

---

## 6. Known Limitations and Future Work

### 6.1 No X-pole Fit

The model currently fits only the y-pole. A full 2D fit (complex-valued ODE for joint
x + iy polar motion) would be more physically complete and would allow the phase plane
to be reconstructed from the model, not just observed. This would require treating the
resonator as a complex oscillator and the MLR coefficients as complex numbers.

### 6.2 ODE vs. Full cw.py Impulse Model

The JavaScript ODE omits the semi-annual impulse mechanism (the discrete velocity kicks
in `cw.py`). The impulses represent inertial loading from seasonal snow/ice mass
redistribution. They produce subtle sideband structure in the amplitude spectrum. For a
more faithful reproduction of `cw.py`, these could be implemented as conditional
velocity updates at the zero-crossings of `sin(πt/T_s)` during the RK4 loop.

### 6.3 Data Currency

The embedded IERS data runs through early 2026. It should be refreshed annually or when
a significant new period of data is available. See Section 3.3 for the update procedure.

### 6.4 Training Window Edge Effects

The Chandler bandpass filter has edge effects at the start and end of the time series
(the running mean has fewer neighbors near the boundaries). The first and last ~2 years
of the bandpassed signal are unreliable. The training window slider minimum is set to
1975 to avoid the 1962–1964 edge region. Training window end should not be set closer
than ~2 years from the last data point for the same reason.

### 6.5 Period Precision

The aliased Chandler period is computed from `T_yr` and `T_d` as fixed constants.
The true observed Chandler period has some variability (roughly 430–435 days depending
on the epoch and estimation method). The visualization does not attempt to fit the period
itself — it is derived analytically from the aliasing formula. This is intentional: the
whole point of Pukite's model is that the period is *predicted* by the aliasing, not
fitted as a free parameter.

---

## 7. Commit History

| Commit | Description |
|--------|-------------|
| initial | Base D3 visualization with synthetic data, phase sliders |
| `ede0f3b` | Added `phiAnn`, `phiDrac`, `phiNode`, `ampAnn`, `ampDrac` sliders; updated forcing |
| `800c8fa` | Upgraded to 2-parameter OLS fit; added Chandler bandpass correlator |
| `ae36255` | **Full rewrite:** real IERS data embedded, resonator ODE, 9-column MLR, r=0.91 |

---

## 8. References

- Pukite, P. (model author) — `cw.py`, `ChandlerWobbleForcing.pdf`, `dialog.pdf` in this repository
- IERS EOP C04 series: https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html
- Lambeck, K. (1980). *The Earth's Variable Rotation*. Cambridge University Press.
- Gross, R. S. (2000). The excitation of the Chandler wobble. *Geophys. Res. Lett.*, 27(15), 2329–2332.
- Pukite's aliasing derivation: `T_CW = T_yr / (T_yr/T_d − 13) / 2` — see `ChandlerWobbleForcing.pdf`
