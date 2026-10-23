# The Chandler wobble as a pear-shaped, polar-magnetized spheroid in a pulsed electromagnet (2026-10-23)

Companion to the QBO device in `~/eval/gem-lte-core/docs/qbo_tokamak.md`. There, a toroidal plasma stood in for the
stratosphere, with the lunar tide as the coil windings. Here, a different laboratory system stands in for the solid
Earth's free nutation: a magnetized body whose spin precesses. Code: `cw_magnetized_spheroid.py` and
`cw_sidebands.py` (section 5). Figures: `cw_spheroid_equivalence.png`, `cw_spheroid_spectra.png` and
`cw_spheroid_sidebands.png`. Numbers: `cw_magnetized_spheroid.json` and `cw_sidebands.json`. Data: `../cw.dat`
(the Chandler band of polar motion, 1935–2018.6, sampled every 0.05 yr).

![Equivalence and device](cw_spheroid_equivalence.png)

## 1. The exact part: Euler's top is a Landau–Lifshitz macrospin

In the body frame, a free rigid body obeys Euler's equation:

  dL/dt = L × (I⁻¹ L)

A uniformly magnetized ellipsoid (a "macrospin") obeys the Landau–Lifshitz equation. Its unit magnetization m precesses
in its own demagnetizing field, H = −M_s N m, where N is the shape tensor:

  dm/dt = γ M_s m × (N m)

The two equations are the same equation under the substitution

  m = L/|L|,  γ M_s N = |L| I⁻¹

Both describe a quadratic energy on the sphere |L| = const: ½ L·I⁻¹L for the top, ½ μ₀M_s² m·N m for the magnet. Each
term matches:

| Rotating Earth | Magnetized spheroid |
|---|---|
| spin angular momentum L, nearly polar | magnetization m, polar ("polar-magnetized") |
| inverse inertia tensor I⁻¹ | shape (demagnetizing) tensor N |
| spin about the axis of greatest moment C (stable) | m along the axis of smallest N (shape-easy axis) |
| free Eulerian nutation (polar motion seen from the body) | uniform (k = 0) Kittel precession mode |
| Euler: σ = Ω √((C−A)(C−B)/(AB)) | Kittel, no applied field: ω = γM_s √((N_x−N_z)(N_y−N_z)) |
| Q ≈ 85 (anelastic mantle, oceans) | Gilbert damping α = 1/(2Q) = 0.0059 |
| external torque (lunisolar) | external field torque |

**Check** (figure, panel b). A triaxial top (A, B, C = 0.8, 0.9, 1.0) and its mapped macrospin were integrated side by
side. They agree to 1.1 × 10⁻¹⁰ over 50 periods. The measured frequency, 0.16666, matches both the Kittel and Euler
formulas, 0.16667.

**Earth's numbers.** H = (C−A)/A = 0.0032738 gives an Euler period of 304.6 days. The observed 432.3 days is longer by
a factor of 1.419 (elastic yielding of the mantle, the oceans and the fluid core). In the magnet, this corresponds to a
softened effective anisotropy N⊥ − N_z: part of the "inertia" does not follow the spin. The device takes the free
period as given (432.3 days, as in `../cw_physics.py`). It does not derive it.

The pear shape (a degree-3, north–south asymmetry) does **not** enter the free frequency. Both Euler and Kittel
depend only on second moments. The pear shape enters only through the forcing (section 2).

## 2. The device

| Earth | Device | In the code |
|---|---|---|
| Moon's ecliptic latitude β (draconic month, 27.2122 d) | coaxial (zonal wavenumber 0) electromagnet, current I(t) ∝ sin β | `field(t, "beta")` (Meeus mean elements, so the lunar phase is astronomical) |
| tidal torque on the bulge, (3GM/r³)(C−A) sin δ cos δ: blind to the perturber's sign | torque on the induced moment of a shape-anisotropic body, ∝ (N_a−N_c) B² sin 2ε: blind to the field's sign | torque ∝ B² |
| obliquity ε = 23.4° between spin axis and ecliptic normal | spheroid axis tilted by ε to the coil axis | geometry only |
| seasonal mass redistribution, impulsive changes of the inertia tensor twice a year | shape (anisotropy) pulsed twice a year: the **carrier comb** | `comb_times(t0)` |
| northern and southern hemispheres unequal | pear shape: north and south pulses unequal, w = 1 ± a | `kicks(t0, a)` |
| polar motion p = x + iy | transverse magnetization m_x + i m_y | `response_linear`, `response_llg` |

**Coupling through the comb.** The coil's torque is continuous, but by itself it is far off resonance. Its B² content
sits at 2f_D = 26.84 cycles/yr, against the mode's 0.845 cycles/yr. It reaches the precession mode only during the
inertia pulses: each pulse at t_n tilts m by

  Δθ_n = s · w_n · B(t_n)²,  w_n = 1 + a (north pulse), 1 − a (south pulse)

So the comb is the carrier and the electromagnet is the modulator. This is the circuit of `../cw_physics.py` (a
unit-RMS squared draconic sinusoid as the impulse magnitude), now with a physical reading for each factor. Between
pulses, the mode precesses freely with Gilbert damping. The linearized propagator is exact for p. A full nonlinear LLG
integration at the realistic amplitude (rms 1.0 × 10⁻⁶ rad ≈ 0.2″) differs from it by 0.56% rms, mostly from where the
pulses fall on the time grid.

**Optional parametric part ("inertia phase steps").** While the shape is pulsed, the precession frequency itself
shifts, so each pulse also advances the precession phase. If the north and south pulses differ, the alternating part
±ε of this phase step is the one that matters. The constant part only renames the mean frequency.

## 3. Alias algebra: one rule for both the QBO and the CW

The comb samples the drive. A same-polarity semiannual comb has harmonics at even integers (cycles/yr); an
alternating-polarity comb has them at odd integers. The observed line is the drive's frequency folded to the nearest
harmonic:

| lunar month | drive B (sign-sensitive), same comb | B, alternating comb | B² (sign-blind), same comb | B², alternating |
|---|---|---|---|---|
| **draconic** 27.2122 d | 0.578 /yr (632 d) | **0.422 /yr (2.370 yr) = QBO** | **0.844 /yr (432.8 d) = CW** | 0.156 |
| tropical 27.3216 d | 0.632 | 0.368 | 0.737 (495.9 d) | 0.264 |
| anomalistic 27.5545 d | 0.745 | 0.255 | 0.511 (715.5 d) | 0.490 |
| synodic 29.5306 d | 0.368 | 0.632 | 0.737 (495.9 d) | 0.264 |

Observed CW peak in `cw.dat`: 0.8428 /yr (433.4 d). The record is 83.6 yr long, so the frequency resolution is about
0.012 /yr, and the predicted 0.8440 /yr lies well within it.

The QBO and the CW come out of one table. The difference is which physical quantity responds:
- **QBO:** a wind, which has a direction. The drive enters linearly, and the solar declination flips the comb's
  polarity: B¹ with an alternating comb.
- **CW:** an axis, which has no head or tail (the dialog's π-symmetry of the two poles). The drive enters squared, and
  the comb keeps one polarity: B² with a same-polarity comb.

In magnetic terms, the QBO device is a reversing (H-bridge) poloidal drive. The CW device is the field-squared torque
on an induced moment: a soft-iron needle turns toward a field whichever way the field points.

In both systems only the draconic month gives the observed line. That is, the electromagnet must be referenced to the
ecliptic, which is the plane that sets the comb, not to the equator.

## 4. Results

![Spectra and fit](cw_spheroid_spectra.png)

Fit: two free timing parameters, the comb's trigger day t₀ and the asymmetry a. After the fit, a linear regression
sets the amplitude, the projection (the coefficients of Re p and Im p), a constant and an annual sine/cosine (the
annual wobble, which is a separate forced line). The fit maximizes correlation on 1935–1990; 1990–2018.6 is held out.
Spectra use a Hann window, are normalized to each curve's CW peak, and cover 1935–2018.

| Device variant | strongest line 0.45–0.95 | 1 − CW sat. (power/CW) | 1 + CW sat. | train r | held-out r | log-spectrum r vs data |
|---|---|---|---|---|---|---|
| continuous B² torque, no comb | (CW power 6 × 10⁻¹⁰ of the comb case) | — | — | 0.48 | 0.53 | 0.33 |
| comb × B², symmetric (a = 0) | 0.844 | < 10⁻⁸ | < 10⁻⁷ | 0.840 | 0.853 | 0.55 |
| comb × B², pear (fitted a) | 0.844 | 5 × 10⁻⁶ | 5 × 10⁻⁵ | 0.841 | 0.853 | **0.71** |
| comb × B², pear + inertia phase steps ±0.2 rad | 0.844 | 4.0 × 10⁻³ | 4.1 × 10⁻³ | 0.836 | 0.852 | 0.67 |
| comb × B², declination field (tropical) | 0.737 | — | — | 0.55 | 0.69 | 0.30 |
| comb × B, same polarity | 0.578 | — | — | 0.40 | 0.61 | 0.08 |
| comb × B, alternating (QBO rule) | 0.422 | — | — | 0.52 | 0.25 | 0.18 |
| *sinusoid at 0.8440 /yr (same regression)* | 0.844 | — | — | 0.840 | **0.877** | — |
| **observed** | **0.843** | **4.5 × 10⁻³ at 0.157** | **3.7 × 10⁻³ at 1.844** | | | |

### What holds

- **The carrier is necessary.** Without the inertia comb, the B² torque leaves no CW line: its power at 0.84 /yr is
  6 × 10⁻¹⁰ of the comb case's. The comb converts an off-resonant 26.84 /yr torque into a
  resonant 0.844 /yr one.
- **The period is a prediction, not a fit.** 432.8 days follows from the draconic month and a sign-blind (B²) drive on
  a same-polarity comb. The same table gives the QBO's 2.370 yr from the sign-sensitive rule. Every other combination
  of month and rule misses (0.42–0.74 /yr), and those variants fit `cw.dat` far worse (r = 0.40–0.55 against 0.84).
- **The pear shape places the satellites.** Unequal north and south pulses put lines at 1 − CW = 0.156 and
  1 + CW = 1.844 /yr. The data have peaks at 0.157 and 1.844, the 2021 result (`../ChandlerWobbleForcing.pdf`)
  reproduced in the device. The pear shape raises the log-spectrum correlation from 0.55 to 0.71.

### What does not hold, or is not decided

- **The satellites are too weak if they pass through the resonance.** As additive forcing lines, they lie off the
  mode and Q = 85 suppresses them; at any asymmetry a they come out 100–1000 times below the data. Section 5 shows from
  their phases that they are not resonance-filtered forcing lines at all, but products of multiplying the CW line by
  an annual factor. The ±0.2 rad phase-step variant in the table matches their power but has the wrong phase
  relation, so section 5 excludes it.
- **In the time domain the device is a sinusoid.** Its held-out correlation (0.853) does not beat a plain sinusoid at
  the predicted frequency (0.877). Reversing the split (fit 1990–2018, evaluate 1935–1990) gives 0.778 against 0.803.
  The device's slow envelope does not track the observed amplitude history: the rms in 5-year blocks goes from 0.025
  (1935–45) to 0.064 (1950–55), down to 0.023 (1965–70), up to 0.052 (1990–2000) and down to 0.031 (2010–15),
  and the device does not follow this. The draconic comb has almost no 18.6-yr envelope (β's
  amplitude is nearly constant), so these must come from elsewhere: a nodal envelope as in `cw_physics.py`, or
  stochastic excitation of the free mode.
- **The trigger day is not identifiable.** The fit prefers a ≈ ±1, a single strong pulse per year. Its day is 171
  (fit 1935–1990) or 35 + 182.6 = 217 (fit 1990–2018). These differ by 46 days, 3.4 × 13.6 days, and 13.6 days is the
  period of B²: shifting the comb by one B² period leaves the alias phase unchanged. As with the QBO's trigger day, the
  fit fixes the alias phase, not a calendar date.
- **A hint to follow up.** `cw.dat` has a secondary peak at 0.7367 /yr (5.7 × 10⁻³ of the CW power), the B² alias of
  the tropical month (0.7365). It lies 0.107 /yr from the CW line, about nine resolution widths, so it is not a
  sidelobe. Whether it is a real declination-driven line or chance is not yet tested (for example, a surrogate test on
  the band 0.6–1.0 /yr).


## 5. The satellites are mixing products, not resonance-filtered lines (`cw_sidebands.py`)

![Sidebands](cw_spheroid_sidebands.png)

Figure: `cw_spheroid_sidebands.png`; numbers: `cw_sidebands.json`.

**Two ways a line can appear at CW ± 1.** In the device, the comb with unequal north and south pulses has odd (annual)
harmonics of relative strength a. Mixing these with the B² drive (26.844 /yr) gives forcing lines at
|26.844 − 27| = 0.156 and |26.844 − 25| = 1.844 /yr. There are two ways for those lines to reach the output:

1. **Additive, then filtered.** The two lines are separate forcing terms and each passes through the resonance. Each
   is about 1 /yr off the mode, against a linewidth of f₀/2Q = 0.005 /yr, so both are suppressed by
   |H(f ± 1)/H(f)| = 0.0051 in amplitude, i.e. 2.6 × 10⁻⁵ in power. Matching the observed 5 × 10⁻³ (least-squares
   amplitudes; the periodogram gives 4 × 10⁻³) would need a ≈ 14: impossible, since a ≤ 1.
2. **Multiplicative.** The CW line itself is multiplied by an annual factor, x(t) = CW(t) · (1 + m cos 2π(t − ψ)). The
   sidebands are then mixing products of the carrier. Their size is set by the modulation depth m, not by the
   resonance: each has amplitude m/2 relative to the CW line, the two are equal, and they inherit the CW's coherence.

**The phase test.** Fit the five lines f, f ± 1, f ± 2 (and the annual and semiannual wobble) jointly to `cw.dat`,
and form the phase closure

  D₁ = φ(f + 1) + φ(f − 1) − 2φ(f)

D₁ does not depend on the time origin, on the modulation phase ψ, or on a small error in f (checked: an AM test signal
gives D₁ = 0 ± 1° for any ψ). Each mechanism has a signature:

| Mechanism | D₁ | upper / lower sideband amplitude (relative to CW) | f − 2 line (1.156) |
|---|---|---|---|
| amplitude modulation (AM) of the CW | **0°** | equal | only with a semiannual modulation |
| phase or frequency modulation (PM), including the ±0.2 rad inertia phase steps | **180°** | equal | — |
| additive pear comb through the resonance (a = 0.13–1) | 166–170° | upper 3–5× lower | 0.019, from the comb |
| additive mixing with no resonance (the forcing itself, smoothed) | — | lower ≫ upper | 0.46, as strong as a carrier |
| **cw.dat, 1935–2018** | **16° ± 19°** | **0.080 / 0.063** | **0.044** |
| cw.dat, 1935–1977 / 1977–2018 | 22° / 18° | 0.131 / 0.080; 0.036 / 0.049 | 0.062 / 0.039 |

(The ± is a block bootstrap of the fit residual: 300 draws of 2-yr blocks.)

- **The data are amplitude-modulated.** D₁ = 16° ± 19° is within one standard deviation of AM and about eight from PM
  or from the resonance-filtered additive lines. Both halves of the record agree (22°, 18°). The two sidebands are
  nearly equal: a power ratio of 1.6. A real second-order resonator would make the lower 15 times stronger than the
  upper. The circular Kittel mode with the device's additive lines makes the upper several times stronger.
- **The sideband strength comes from mixing with the carrier.** It is a multiplicative (nonlinear) product, not what
  the resonance lets through. The resonance selects the carrier; the seasonal factor then multiplies the result.
- **This excludes the section 4 patch.** The ±0.2 rad inertia phase steps matched the sideband power but give
  D₁ = 180°. The inertia pulses must instead rescale the wobble's amplitude. In the variant where each pulse
  multiplies p by 1.25 and 1/1.25 alternately (`mech_amp_steps`), D₁ = 0° and both sidebands are at the observed level.

**The sidebands measure the north/south balance.** In the multiplicative reading, the sideband/carrier amplitude is
m/2, so m = 2 × 0.072 = 0.14. The wobble's amplitude swings ±14% through the year. With unequal pulses 1 ± a, that
means a ≈ 0.14: a north:south strength ratio of about 1.33. Panel (b) shows how steeply the satellite power depends on
a in each reading. The balance, not the resonance bandwidth, sets the satellite level: in the additive-through-resonance
reading even a = 1 falls about 200× short.

**The modulation has a season.** Unlike D₁, the modulation phase depends on the individual line phases:
ψ = (φ(f) − φ(f+1))/2π, averaged with (φ(f−1) − φ(f))/2π. The CW amplitude is largest around **day 337 ± 9 (early
December)**, and at the same time of year in both halves of the record (days 338 and 327).
- In the amplitude-step reading, the enlarging pulse comes a quarter year before the peak and the reducing pulse half
  a year after that: around days 246 and 63 (early September and early March).
- For comparison, the QBO comb fitted trigger days 235 and 53. That fit traded off against the integrator lag, so the
  agreement is a pointer, not a result.
- These dates bear on the question left open in `../ChandlerWobbleForcing.pdf`: which seasonal impulse is the larger.
  The sidebands say the amplitude grows across the September pulse and shrinks across the March pulse. They do not by
  themselves say which pulse is "north".

**Not settled.**
- **Semiannual modulation.** The f ± 2 lines (1.156 and 2.844 /yr) are 0.044 and 0.038 of the carrier, about ten
  times the device's. Read as AM, that is a semiannual depth of about 0.08, but D₂ = 70° ± 64° is too noisy to classify.
- **Artifact.** An annual modulation of the CW amplitude could come from how `cw.dat` was made (a seasonally dependent
  reduction or filter) rather than from the Earth. The next test is the same phase closure on the raw IERS EOP C04
  x and y series, separately: a modulation in the body would appear in both.

## 6. The device as a laboratory system (not yet checked for feasibility)

The dialog (`../dialog.pdf`) concluded that no controlled experiment could test aliased draconic forcing. The
equivalence of section 1 changes that: the free mode is the Kittel mode of a nanomagnet, and every ingredient has a
laboratory counterpart. The rescaling below is illustrative, like the QBO "shot" (1 yr = 1 s); it has not been
checked against a real device.

| Earth (per year) | Nanomagnet (1 yr → 1 ns) |
|---|---|
| CW, 0.844 cycles/yr, Q = 85 | Kittel mode at 0.84 GHz, α = 0.006 (permalloy's damping range) |
| draconic month, 13.42 cycles/yr | coaxial microwave field at 13.42 GHz; the torque's B² content at 26.84 GHz |
| semiannual inertia impulses | 2 GHz train of anisotropy pulses (e.g. voltage-controlled magnetic anisotropy) |
| north/south asymmetry | alternate pulse heights 1 ± a |
| 84-yr record | 84-ns record |

The prediction to test: the precession appears at the alias frequency |2f_coil − 2k f_comb|, not at the coil's
frequency. With the comb off, the mode stays quiet. With unequal alternate pulses, satellites appear at f_comb/2 ± the
alias frequency.

## 7. Summary

- **Exact:** Euler's equations for the spinning Earth are the Landau–Lifshitz equations for a macrospin, with
  N ↔ I⁻¹. The Chandler wobble is the Kittel mode of a polar-magnetized spheroid.
- **Mechanism:** the lunar drive is a wavenumber-0, sign-blind (B²) torque from a coaxial electromagnet. It reaches the
  mode only through the twice-yearly inertia (anisotropy) pulses: the carrier comb. That product puts the line at
  432.8 days with no tuned parameter, and the same folding rule gives the QBO's 2.370 yr.
- **Pear shape:** north–south asymmetry adds the 0.157 and 1.844 /yr satellites at the observed positions. Their
  phases show they are amplitude-modulation products of the CW line (section 5), not resonance-filtered forcing lines
  and not phase modulation. Read that way, their strength measures the N/S balance: the wobble's amplitude swings
  ±14% through the year, largest in early December.
- **Limits:** in the time domain the device is no better than a sinusoid at the predicted frequency. The CW's amplitude
  history needs another ingredient.
