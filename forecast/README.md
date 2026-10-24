# Chandler wobble forecast ledger

A standing, automated test of the forced-CW reading in [`../docs`](../docs/README.md). Competing engines project
polar motion from the same origins and are scored on the same IERS observations. Each projection is frozen when
it is issued, and git history timestamps it. The current scores are in [REPORT.md](REPORT.md).

## Engines (`targets.yml`)

| engine | represents |
|---|---|
| `alias_cw` | port of the gem-lte-core forecast: a prograde CW line at the forced (draconic) alias, the annual line, and drift |
| `ls_free` | benchmark: least-squares harmonics with the CW frequency fitted over 10 yr (standard EOP practice) |
| `free_decay` | consensus: the CW as a free mode (432.3 d, Q 85), decaying from its state at the origin |
| `device` | the gated lunar device (`docs/cw_flagship.py`): sin²β (a/r)³ gated by push-pull semiannual pulses through one prograde mode |
| `two_line_fm` | steady forced lines at the draconic alias and the F − M alias, fitted on C01 + C04 since 1890; their ~93-yr beat carries the envelope |

The engine settings and the source of `engines.py` are hashed into each issue's `fingerprint`. Changing either one
makes the next issue a new issue, and the issues already in the ledger stay as they were.

## Issues and scoring

- **Live issues.** One per engine per calendar month (`ledger/<engine>/live-YYYY-MM/`), 6 yr ahead, from the last
  complete 0.05-yr bin. A live issue is never overwritten.
- **Hindcasts.** Rolling origins every 3 months from 2005, 3 yr ahead. Each uses only bins at or before its origin.
  Missing origins are filled in as the data grow.
- **Scores** (`scores/scores.csv`). RMS of the vector error |Δp| (mas) per horizon bin (0–0.5, 0.5–1, 1–2, 2–3,
  3–6 yr), with skill against persistence and against `ls_free`. Live issues are also scored on the samples after
  their issue date (`post_issue`), which no fit could have seen.

## Registered claims (`claims.yml`)

Five dated, falsifiable statements, registered 2026-10-24: CW regrowth to ≥ 50 mas around 2029; the CW keeps its
1939–2008 phase; the Mm and 7/6 lines keep their phase; and the 1 ∓ CW sidebands stay absent. Each claim is
judged PASS / FAIL / INCONCLUSIVE once the data pass its `evaluate_after` date. Until then `claims_status.json`
carries a provisional reading, together with each engine's prediction from its latest live issue.

## Data (`sources.py`)

IERS EOP C01 (1846–1961), EOP 20 C04 (daily from 1962) and the USNO finals2000A measured rows after C04 ends,
fetched over FTP or HTTPS and binned to 0.05 yr in `observations/polar_motion.csv`. Raw downloads go to `cache/`,
which is not committed; the workflow keeps them in the Actions cache. A failed download falls back to the cached copy.

## Running

```
pip install -r forecast/requirements.txt
python3 forecast/ledger.py all        # observe, issue, backfill, score, claims, report (about 1 min)
python3 forecast/ledger.py list       # engines and claims
```

`.github/workflows/cw-forecast.yml` runs `all` every Monday (and on manual dispatch) and commits the observations,
new issues, scores and report.
