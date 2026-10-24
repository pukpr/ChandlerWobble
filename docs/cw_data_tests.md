# Chandler wobble: data tests, and what they overturned

*Reconciled 2026-10-23. The argument these tests support: [cw_synchronized_wobble.md](cw_synchronized_wobble.md).
Every number here is produced by a script in this directory (listed at the end). §8 records the claims made earlier
in this work that the tests withdrew.*

## 1. The two data sets

**Raw IERS polar motion.** These are the IERS Rapid Service daily x and y (USNO `finals2000A.all`, measured rows only,
1973.0–2026.8), averaged into 0.05-yr bins. `fetch_eop.py` downloads them to `data/`, which is not in the repository.
Polar motion is analysed as the complex series p = x − iy, in which the CW turns prograde (positive frequency). Two
other sources could not be used: the IERS datacenter rejected scripted downloads, and the Paris Observatory server
(C01, C04) timed out. So nothing before 1973 has been checked against raw data.

**`../cw.dat`.** This is a processed x-pole series, 1935–2018.6 at 0.05 yr, of unknown provenance. Over 1973–2018 it
equals −0.334 x + 0.05 y (r = −0.967 with x alone), but with signs of further processing:
- **Drifting scale:** the coefficient on x drifts from −0.29 (1973–85) to −0.39 (2009–18).
- **A splice:** the time grid breaks at 2004.95 → 2004.9975.
- **Semiannual removed:** that line comes out at a gain of 0.10.
- **A seasonal residual:** after regression on x and y, what is left has an rms twice as large in April–August as in
  January–February.

## 2. The 1 ∓ CW side peaks of `cw.dat` are products of its processing (`cw_sidebands.py`, `cw_eop_closure.py`, `cw_peaks.py`)

**What `cw.dat` shows.** Peaks at 0.157 and 1.844 /yr, about 0.06–0.08 of the CW in amplitude, with phase closure
D₁ = φ(f+1) + φ(f−1) − 2φ(f) = 16° ± 19°. That is the signature of amplitude modulation (0°), not phase modulation
(180°), and the same in both halves of the record. Read as Earth physics, it would mean a ±14% annual swing in the
wobble's amplitude.

**Raw polar motion does not have them.** Amplitudes are relative to the CW line:

| window | raw p, upper sideband (+1.843) | raw x, upper / lower | `cw.dat`, upper / lower |
|---|---|---|---|
| 1973–1990 | 0.010 | 0.011 / 0.001 | 0.064 / 0.064 |
| 1990–2026 | 0.0003 | 0.004 / 0.038 | 0.032 / 0.048 |
| 1973–2018.6 | 0.005 | 0.009 / 0.016 | 0.033 / 0.055 |

The raw phase closures are noise (spreads of 70–160°). The upper sideband sits in a clean part of the raw spectrum,
and there it is 6–100 times weaker in amplitude than in `cw.dat`. Line by line over 1973–2018, `cw.dat` ÷ (−0.334 × raw x):

| CW | annual | semiannual | 2 − CW | 1 − CW | 1 + CW | 2 + CW | 1.5 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.10 | 0.10 | 0.98 | **3.44** | **4.11** | 3.73 | 3.18 | 5.41 |

No fixed (time-invariant) filter jumps from a gain of 1 to 3.4–4.1 at selected lines. Some multiplicative step is
involved.

**The mechanism: multiplicative cross-terms from a calendar-locked gain.** Multiplying a series that contains the CW
by anything with an annual period puts equal lines at f_CW ± 1 with closure 0°, the observed signature. A semiannual
factor adds 2 ± CW; annual × semiannual adds 3.0. Two candidate products were tested by regression on raw x and y.
The table gives line power left over, relative to the CW:

| model regressed onto `cw.dat` | 0.157 | 1.843 | 3.0 |
|---|---|---|---|
| `cw.dat` itself | 2.2 × 10⁻³ | 5.6 × 10⁻⁴ | 7.5 × 10⁻⁴ |
| a·x + b·y + trend | 1.8 × 10⁻³ | 4.5 × 10⁻⁴ | 7.9 × 10⁻⁴ |
| + x², y², xy (squared amplitude) | 1.7 × 10⁻³ | 3.1 × 10⁻⁴ | 7.7 × 10⁻⁴ |
| + seasonal gain on x, y (4 terms) | **6.5 × 10⁻⁵** | 5.7 × 10⁻⁴ | 7.4 × 10⁻⁴ |
| + annual and semiannual gain (8 terms) | 5.7 × 10⁻⁵ | 5.6 × 10⁻⁴ | **2.2 × 10⁻⁴** |

- **The squared amplitude of x and y does not produce them.** It also predicts the wrong season: a product with the
  annual wobble would peak on day 75 or 258, against the measured day 337.
- **A seasonal gain of ±7% on x does:** it removes 96% of the 0.157 excess, and with a semiannual term the 3.0 line
  as well.
- **The 1.843 excess comes mostly from before 1973.** Its window amplitudes are 0.154 (1935–63), 0.057 (1963–91) and
  0.034 (1991–2018), a decline that follows observing technology. A seasonal error is expected in optical latitude
  observations.

**Consequence.** The `--amp-step` option of `../cw.py` (annual rescaling of the wobble, s = 0.25) reproduces
`cw.dat`, not the Earth. Raw data limit any annual amplitude modulation of the CW to about 1% since 1990. The 2021
satellite result (`../ChandlerWobbleForcing.pdf`) appears to rest on a similar processed series. Before 1973 is
untested.

## 3. Every peak (`cw_peaks.py`)

Every local maximum of the `cw.dat` periodogram above 3 × 10⁻⁴ of the CW power, matched to n₁·CW + n₂ and to the lunar
alias lines, and checked against raw p:

| peak (/yr) | power / CW | raw p, prograde / retrograde | share left after a·x + b·y (1973–2018) | identification | verdict |
|---|---|---|---|---|---|
| 0.843 | 1 | 1 / 10⁻⁵ | 0.2% | CW: draconic B² alias 0.8440 | Earth, phase-locked |
| 1.000 | 0.32 | 0.39 / 9 × 10⁻⁴ | 1.5% | annual wobble | Earth |
| 0.737–0.748 | 5.7 × 10⁻³ | 1.2 × 10⁻² / 2 × 10⁻⁵ | 0.5% | **Mm: 14 − anomalistic = 0.7448** (tropical B² 0.7365 is 0.008 away) | Earth, stationary line (§4) |
| 0.817 | 0.064 (beside the CW) | not resolved | — | — | slow modulation of the CW (§4) |
| 0.940, 0.680 | 2 × 10⁻³ – 8 × 10⁻³ | similar, prograde | 3–12% | — | Earth: broadened CW band |
| 2.0 | removed in `cw.dat` | 2.5 mas, steady | — | semiannual (comb) | Earth |
| 0.157, 1.844 | 4.5 × 10⁻³, 3.7 × 10⁻³ | ≤ 2 × 10⁻⁴ | 80% | 1 ∓ CW | processing (§2) |
| 1.524, 2.984 | ~10⁻³ | ≤ 4 × 10⁻⁵ | 50–100% | 3CW − 1; 3 | processing |
| 1.7–5 (about 30 peaks) | 3 × 10⁻⁴ – 1.3 × 10⁻³ | 10⁻⁶ – 10⁻⁵ | — | none | `cw.dat` noise floor, ~100× raw |

## 4. Phase stability: which lines are forced (`cw_peaks.py`)

A forced line keeps its absolute phase from window to window; a noise peak, or a sidelobe of a changing CW, does not.
This is raw p in four 13.5-yr windows, with amplitudes in mas:

| line | 1973–86 | 1986–2000 | 2000–13 | 2013–27 | phase spread | p (random phases) | best-fit frequency |
|---|---|---|---|---|---|---|---|
| CW 0.8440 | 156 | 177 | 109 | 35 | ±18° | 0.002 | 0.8435 |
| **Mm 0.7448** | **18** | **24** | **26** | **18** | **±34°** | **0.017** | **0.7478** |
| tropical B² 0.7365 | 15 | 19 | 20 | 16 | ±54° | 0.065 | 0.7480 (the same peak) |
| Mm2 0.5105 | 3 | 3 | 7 | 1 | ±141° | 0.73 | 0.494 |
| 14 − tropical 0.6317 | 4 | 4 | 8 | 6 | ±135° | 0.68 | 0.621 |
| 2 − CW 1.1560 | 2 | 1 | 14 | 7 | ±104° | 0.41 | 1.163 |
| 1.1848 | 3 | 4 | 6 | 4 | ±86° | 0.25 | 1.165 |
| semiannual 2.0 | 3 | 4 | 3 | 1 | ±22° | 0.005 | 2.000 |

- **The CW** keeps its phase while its amplitude collapses after 2013.
- **Mm keeps phase and amplitude while the CW changes 5×,** so it is not a CW sidelobe. It is prograde (retrograde
  part a few percent), as a line passing through the prograde mode should be.
- **The tropical fit at 0.7365** is the same peak, fitted 0.008 /yr off its true frequency.

**The 0.817 /yr feature** was once a candidate for the free mode, but its phase is steady. This is `cw.dat` in
windows long enough to separate it from the CW:

| windows | 0.817 amplitude / CW | 0.817 phase | 0.871 phase | closure with the CW (AM = 0°) |
|---|---|---|---|---|
| 1935–77 / 1977–2018 | 0.31 / 0.35 | 48° / 61° (±6°) | −149° / −154° | 24° / 48° |
| 1935–63 / 1963–91 / 1991–2018 | 0.45 / 0.13 / 0.42 | 42° / 26° / 74° (±27°) | −146° / −105° / 157° | 23° / 42° / 18° |

A noise-driven free mode would wander. This feature holds its phase and closes with the CW like amplitude modulation,
so it reads as a slow (~37-yr) modulation of the forced CW. With only two or three independent windows, that reading
is suggestive.

## 5. Joint chance of the two locks (`cw_joint_lock.py`)

Both lines are named in advance from the alias rule, so there is no look-elsewhere penalty. The null model is a free
wobble (432.3 d) excited by white noise, with a small white floor, analysed in the same four windows (40,000 runs
per Q):

| null model | CW lock | Mm lock | both locks | both + Mm amplitude steady |
|---|---|---|---|---|
| Q = 50 | 0.080 | 0.018 | 0.0014 | ~5 × 10⁻⁵ (2 hits) |
| Q = 85 | 0.157 | 0.018 | 0.0024 | ~1.5 × 10⁻⁴ (6 hits) |
| Q = 150 | 0.282 | 0.017 | 0.0050 | ~1 × 10⁻⁴ (4 hits) |

The amplitude criterion uses the observed value as its threshold, and the last column rests on few hits. With three
18-yr windows the CW lock alone has p ≈ 0.05.

## 6. Line strengths and the bandpass (`cw_bandpass.py`)

The device has to produce each line at its observed strength:
- **Consensus resonance (432.3 d, Q = 85) with sharp kicks:** Mm comes out at 0.008 of the CW, against 0.106
  observed: 15× too weak.
- **The perigee/apogee distances don't change that.** They (357,000–407,000 km) bound the (a/r)³ modulation at
  ±19.5%, against the ±16.5% eccentricity part already used. That is a ≤ 25% change, not 15×.
- **The fix is to free f₀, Q and the pulse shape.** The result is in lead §3: Q ≳ 30, f₀ ≈ 425–440 d, push-pull
  pulses of about 5.5 d rms, and every line within its noise limit.
- **The time-domain device** (`cw_flagship.py`, which gates the astronomical torque through the pulses) reproduces
  the analytic line ratios: Mm/CW 0.103 against 0.106. The semiannual comes out about 2× high.

## 7. Open checks

- The IERS C01 series (1846–now): CW and Mm phase locks, and the 1 ∓ CW sidebands, before 1973.
- The provenance of `cw.dat`: its source series, the splice near 2005, the semiannual removal and the seasonal gain.
- A physically fixed comb date (the fits repeat every ~27 days), to make the CW–Mm relative phase a test.

## 8. Claims withdrawn during this work

| earlier claim | status | where |
|---|---|---|
| The 1 ∓ CW satellites are amplitude-modulation products of the Earth's wobble; they measure a ±14% N/S balance, largest in early December | **withdrawn**: they come from `cw.dat`'s processing | §2 |
| `--amp-step 0.25` makes the model match the observed sidebands | it matches `cw.dat`, not the Earth; default 0 | §2 |
| The 0.737 peak is the tropical B² alias and is not a forced line | **corrected**: it is a stationary Mm line at 0.7448 | §4 |
| Mm is 15× too strong for the distance modulation | **resolved**: an artifact of fixing f₀ = 432 d, Q = 85 and sharp pulses | §6, lead §3 |
| The natural period is ≈ 450 d, with 0.817 as the free mode and a 37-yr free/forced beat | **withdrawn**: with pulse shape free, f₀ ≈ 425–440 d; 0.817 is phase-steady | §4, lead §3 |
| Low Q (single digits) is allowed once the wobble is forced | **bounded**: Q ≳ 30, set by the 2 − CW line | lead §3 |
| CW counter-torque from a finite-difference inverse | **corrected**: computed exactly in the frequency domain | `cw_counter_torque.py` |
| The Kittel (nanomagnet) mode as the working analog | kept only as a mathematical identity; rotating machinery is the analog | lead §1, [cw_magnetized_spheroid.md](cw_magnetized_spheroid.md) |

## Scripts

| script | what | outputs |
|---|---|---|
| `fetch_eop.py` | download raw IERS polar motion | `data/finals2000A.all` |
| `cw_sidebands.py` | `cw.dat` sideband phase closure (the original AM reading) | `cw_spheroid_sidebands.png`, `cw_sidebands.json` |
| `cw_eop_closure.py` | raw IERS against `cw.dat`: closures, sense test, line gains, seasonal residual | `cw_eop_closure.png`, `.json` |
| `cw_peaks.py` | every peak; processing models; phase stability; blog lines; the 0.817 feature | `cw_peaks.png`, `.json` |
| `cw_joint_lock.py` | joint chance of the CW and Mm locks; Mm strength under fixed resonance | `cw_joint_lock.json` |
| `cw_bandpass.py` | free bandpass fitted to the line set | `cw_bandpass.png`, `.json` |
