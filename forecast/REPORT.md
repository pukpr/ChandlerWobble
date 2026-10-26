# Chandler wobble forecast ledger

Updated 2026-10-26 06:56 UTC. Data end 2026.8049 (feeds: c01 updated, c04 updated, usno updated). Error = |Δp|, the vector error of complex polar motion, in mas. `SS` = 1 − MSE(engine)/MSE(baseline); positive beats the baseline.

## Live issues

| engine | issue | data end | r train | notes | C1 regrowth | C2 phase | C3 Mm | C4 7/6 | C5 sidebands |
|---|---|---|---|---|---|---|---|---|---|
| alias_cw | live-2026-10 | 2026.775 | 0.922 | CW 59 mas | 60 mas | 6° (60 mas) | 1° (16 mas) | 64° (4 mas) | 3 mas |
| device | live-2026-10 | 2026.775 | 0.931 | comb day 163, Q 31.62 | 60 mas | 6° (60 mas) | 1° (17 mas) | 59° (3 mas) | 3 mas |
| free_decay | live-2026-10 | 2026.775 | 0.974 | CW 55 mas at origin, Q 85 | 52 mas | 21° (52 mas) | 17° (14 mas) | 51° (5 mas) | 1 mas |
| ls_free | live-2026-10 | 2026.775 | 0.960 | CW 437.4 d, 38 mas | 42 mas | 5° (42 mas) | 12° (17 mas) | 52° (3 mas) | 1 mas |
| two_line_fm | live-2026-10 | 2026.775 | 0.797 | CW line 67, F−M line 58 mas | 90 mas | 43° (90 mas) | 56° (8 mas) | 53° (8 mas) | 1 mas |

Claim columns are each engine's prediction, from its projection spliced onto the observations (C3/C4: phase change, with amplitude).

## Rolling-origin hindcasts

Origins every 3 months from 2005; each uses only data at or before its origin. Median over origins.

| engine | horizon (yr) | origins | median error (mas) | median SS vs persistence | median SS vs ls_free | share beating ls_free |
|---|---|---|---|---|---|---|
| alias_cw | 0–0.5 | 86 | 89.8 | 0.67 | -1.12 | 0.20 |
| alias_cw | 0.5–1 | 85 | 95.3 | 0.62 | -1.40 | 0.19 |
| alias_cw | 1–2 | 83 | 103.4 | 0.60 | -0.57 | 0.28 |
| alias_cw | 2–3 | 79 | 111.6 | 0.58 | -0.19 | 0.35 |
| ls_free | 0–0.5 | 86 | 63.3 | 0.85 |  |  |
| ls_free | 0.5–1 | 85 | 60.6 | 0.83 |  |  |
| ls_free | 1–2 | 83 | 84.3 | 0.76 |  |  |
| ls_free | 2–3 | 79 | 92.6 | 0.68 |  |  |
| free_decay | 0–0.5 | 86 | 50.5 | 0.93 | 0.27 | 0.69 |
| free_decay | 0.5–1 | 85 | 53.2 | 0.92 | 0.20 | 0.68 |
| free_decay | 1–2 | 83 | 66.6 | 0.86 | 0.34 | 0.77 |
| free_decay | 2–3 | 79 | 72.5 | 0.84 | 0.43 | 0.84 |
| device | 0–0.5 | 86 | 92.8 | 0.67 | -1.39 | 0.19 |
| device | 0.5–1 | 85 | 87.3 | 0.67 | -1.30 | 0.20 |
| device | 1–2 | 83 | 105.7 | 0.58 | -0.72 | 0.24 |
| device | 2–3 | 79 | 109.9 | 0.63 | -0.31 | 0.29 |
| two_line_fm | 0–0.5 | 86 | 82.4 | 0.61 | -1.02 | 0.28 |
| two_line_fm | 0.5–1 | 85 | 72.9 | 0.71 | -0.56 | 0.31 |
| two_line_fm | 1–2 | 83 | 76.3 | 0.70 | -0.35 | 0.47 |
| two_line_fm | 2–3 | 79 | 69.6 | 0.81 | 0.32 | 0.63 |

### Through the collapse (origins 2008–2013, the CW falling from ~110 to ~25 mas)

| engine | 0–1 yr error (mas) | 1–3 yr error (mas) |
|---|---|---|
| alias_cw | 103.3 | 114.5 |
| ls_free | 70.5 | 119.3 |
| free_decay | 54.3 | 73.6 |
| device | 95.6 | 134.4 |
| two_line_fm | 123.9 | 77.2 |

## Registered claims

| claim | evaluate after | status | provisional reading (data so far) | engines predicting PASS |
|---|---|---|---|---|
| **C1_regrowth**: The CW regrows: its amplitude at the draconic alias, centred 2029.0 (window 2026-2032), is at least 50 mas. | 2032.0 | pending | value 55.2 | alias_cw, device, free_decay, two_line_fm |
| **C2_phase_lock**: The CW keeps the phase of its 1939-2008 plateau: within 45 deg of it, centred 2029.0. | 2032.0 | pending | value 23.2, signed 23.2, amp_mas 55.2 | alias_cw, device, free_decay, ls_free, two_line_fm |
| **C3_mm_persists**: The Mm line (14 - anomalistic, 0.7448 /yr, prograde) persists over 2017-2031 at 10-35 mas and within 45 deg of its 1973-2016 phase. | 2031.0 | pending | value 18.8, signed -18.8, amp_mas 22.5, ref_amp_mas 10.8 | alias_cw, device, free_decay, ls_free |
| **C4_seven_sixths**: The 7/6 line (1 + F - M, 1.16675 /yr, prograde) keeps its phase: over 2019-2031, within 45 deg of its 1973-2019 phase. | 2031.0 | pending | value 33.4, signed -33.4, amp_mas 5.4, ref_amp_mas 5.6 | — |
| **C5_no_sidebands**: No 1 -/+ CW sidebands in raw polar motion over 2017-2031: both below 3 mas (they were cw.dat processing products). | 2031.0 | pending | value 2.5, upper_mas 1.0, lower_mas 2.5 | alias_cw, device, free_decay, ls_free, two_line_fm |

Claims: `claims.yml` (registered 2026-10-24). Engines: `targets.yml`. Method: `README.md`.
