#!/usr/bin/env python3
"""Chandler wobble forecast ledger: competing engines, frozen projections, rolling-origin hindcasts, and registered
claims, all scored against IERS polar motion.

    python3 forecast/ledger.py all          # observe + issue + backfill + score + claims + report
    python3 forecast/ledger.py observe      # refresh C01 / C04 / USNO feeds, rebuild observations/
    python3 forecast/ledger.py issue        # one live issue per engine per month (frozen, never overwritten)
    python3 forecast/ledger.py backfill     # rolling-origin hindcasts (only data at or before each origin)
    python3 forecast/ledger.py score | claims | report | list

Layout (committed, so git history is the timestamp):
    forecast/targets.yml, forecast/claims.yml          engine settings, registered claims
    forecast/observations/polar_motion.csv             0.05-yr bins of x, y (arcsec), with source
    forecast/observations/status.json                  feed status and data end
    forecast/ledger/<engine>/<issue>/projection.csv    date, x, y (arcsec), frozen
    forecast/ledger/<engine>/<issue>/manifest.json     fingerprint, settings, fit stats, claim predictions
    forecast/scores/scores.csv                         per issue x horizon bin
    forecast/claims_status.json, forecast/REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engines as E  # noqa: E402
import sources as S  # noqa: E402

LEDGER, OBS, SCORES = HERE / "ledger", HERE / "observations", HERE / "scores"
CFG = yaml.safe_load((HERE / "targets.yml").read_text())
CLAIMS = yaml.safe_load((HERE / "claims.yml").read_text())["claims"]
SET = CFG["settings"]
STEP = SET["grid_years"]


def now_year():
    n = dt.datetime.now(dt.timezone.utc)
    return n.year + (n.timetuple().tm_yday - 1 + n.hour / 24) / (366 if n.year % 4 == 0 else 365)


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------------- observations
def observe():
    state = S.refresh()
    t, x, y, src = S.merged()
    OBS.mkdir(exist_ok=True)
    bt, bx, by = S.binned(t, x, y, STEP)
    bsrc = [src[np.searchsorted(t, c, side="right") - 1] for c in bt]
    with open(OBS / "polar_motion.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["year", "x_arcsec", "y_arcsec", "source"])
        for a, b, c, s in zip(bt, bx, by, bsrc):
            w.writerow([f"{a:.4f}", f"{b:.6f}", f"{c:.6f}", s])
    status = dict(feeds=state, data_end=float(t[-1]), last_bin=float(bt[-1]), n_bins=int(len(bt)),
                  sources={s: int(np.sum(src == s)) for s in np.unique(src)},
                  observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (OBS / "status.json").write_text(json.dumps(status, indent=1))
    print(f"[observe] {len(bt)} bins, data end {t[-1]:.4f} ({status['sources']})")


def load_obs():
    a = np.genfromtxt(OBS / "polar_motion.csv", delimiter=",", names=True, dtype=None, encoding=None)
    t = a["year"].astype(float); p = a["x_arcsec"].astype(float) - 1j * a["y_arcsec"].astype(float)
    status = json.loads((OBS / "status.json").read_text())
    return t, p, status


def complete_bins(t, status):
    """The last bin is used only if the data cover it fully."""
    return t <= status["data_end"] - STEP / 2


# ------------------------------------------------------------------------------------------------------- claims
def _fitwin(t, p, a, b, freqs):
    m = (t >= a) & (t <= b)
    if m.sum() < 20:
        return None
    tt = t[m]; tc = (tt - tt.mean()) / max(b - a, 1)
    cols = [np.exp(2j * np.pi * f * tt) for f in freqs] + [np.exp(2j * np.pi * f * tt) for f in (1.0, -1.0, 2.0, -2.0)] + [np.ones_like(tt), tc, tc ** 2]
    c, *_ = np.linalg.lstsq(np.column_stack(cols), p[m], rcond=None)
    return c


def claim_value(spec, t, p):
    """The claim's metric on series (t, p); None if the series does not cover the window."""
    k = spec["kind"]
    if k in ("cw_amp", "cw_phase"):
        a, b = spec["centre"] - spec["window"] / 2, spec["centre"] + spec["window"] / 2
        if t[-1] < b - STEP:
            return None
        c = _fitwin(t, p, a, b, [E.F_CW])
        if c is None:
            return None
        if k == "cw_amp":
            return dict(value=float(abs(c[0]) * 1e3))
        r0, r1 = spec["reference_centres"]; refs = []
        for cc in np.arange(r0, r1 + 1e-9, 1.0):
            q = _fitwin(t, p, cc - spec["window"] / 2, cc + spec["window"] / 2, [E.F_CW])
            if q is not None:
                refs.append(np.exp(1j * np.angle(q[0])))
        d = np.degrees(np.angle(np.exp(1j * np.angle(c[0])) / np.mean(refs)))
        return dict(value=float(abs(d)), signed=float(d), amp_mas=float(abs(c[0]) * 1e3))
    if k == "line":
        if t[-1] < spec["end"] - STEP:
            return None
        f = spec["freq"]; others = [E.F_CW] + ([0.744759] if abs(f - 0.744759) > 1e-3 else [])
        c = _fitwin(t, p, spec["start"], spec["end"], [f] + others)
        r = _fitwin(t, p, spec["reference"][0], spec["reference"][1], [f] + others)
        if c is None or r is None:
            return None
        d = np.degrees(np.angle(c[0] / r[0]))
        return dict(value=float(abs(d)), signed=float(d), amp_mas=float(abs(c[0]) * 1e3), ref_amp_mas=float(abs(r[0]) * 1e3))
    if k == "sidebands":
        if t[-1] < spec["end"] - STEP:
            return None
        c = _fitwin(t, p, spec["start"], spec["end"], [E.F_CW + 1, E.F_CW - 1, E.F_CW])
        return dict(value=float(max(abs(c[0]), abs(c[1])) * 1e3), upper_mas=float(abs(c[0]) * 1e3), lower_mas=float(abs(c[1]) * 1e3))
    raise ValueError(k)


def _test(expr, v):
    op, num = expr.split(); num = float(num)
    return dict(zip([">=", "<=", "<", ">"], [v >= num, v <= num, v < num, v > num]))[op]


def judge(spec, val):
    if val is None:
        return "pending"
    v = val["value"]
    ok_amp = True
    if spec["kind"] == "line":
        lo, hi = spec["amp_range"]; ok_amp = lo <= val["amp_mas"] <= hi
    if _test(spec["pass_if"], v) and ok_amp:
        return "PASS"
    if _test(spec["fail_if"], v) or not ok_amp:
        return "FAIL"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------------------------------------------- issues
def _write(engine, iid, proj, extra):
    out = LEDGER / engine / iid
    if out.exists():
        return False
    out.mkdir(parents=True)
    with open(out / "projection.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "x_arcsec", "y_arcsec"])
        for a, z in zip(proj["grid"], proj["p"]):
            w.writerow([f"{a:.4f}", f"{z.real:.6f}", f"{-z.imag:.6f}"])
    meta = {**proj["meta"], **extra, "engine": engine,
            "issued_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "issued_year": now_year(),
            "git_rev": git_rev()}
    (out / "manifest.json").write_text(json.dumps(meta, indent=1, default=float))
    return True


def _claim_predictions(t, p, origin, proj):
    ts = np.r_[t[t <= origin], proj["grid"]]; ps = np.r_[p[t <= origin], proj["p"]]
    out = {}
    for cid, spec in CLAIMS.items():
        v = claim_value(spec, ts, ps)
        out[cid] = None if v is None else {k: round(x, 2) for k, x in v.items()}
    return out


def issue_all(only=None, origin=None, iid=None, horizon=None, hindcast=False):
    t, p, status = load_obs(); ok = complete_bins(t, status); t, p = t[ok], p[ok]
    origin = float(t[-1]) if origin is None else origin
    iid = iid or ("live-" + dt.date.today().strftime("%Y-%m"))
    horizon = horizon or SET["live_horizon_years"]
    n = 0
    for name, fn in E.ENGINES.items():
        if only and name not in only:
            continue
        if (LEDGER / name / iid).exists():
            continue
        cfg = CFG["engines"][name]
        proj = fn(t, p, origin, horizon, cfg, STEP)
        extra = dict(issue=iid, data_end=origin, horizon_years=horizon, settings=cfg, fingerprint=E.fingerprint(name, cfg)[:16])
        if hindcast:
            extra["hindcast_origin"] = origin
        else:
            extra["claim_predictions"] = _claim_predictions(t, p, origin, proj)
        n += _write(name, iid, proj, extra)
    return n


def issue(only=None):
    n = issue_all(only)
    print(f"[issue] {n} new live projections")


def backfill(only=None):
    t, p, status = load_obs(); ok = complete_bins(t, status); t = t[ok]
    last = t[-1] - 0.5; origins = np.arange(SET["hindcast_start"], last + 1e-9, SET["hindcast_step_months"] / 12.0)
    n = 0
    for o in origins:
        o_bin = float(t[t <= o][-1])
        y0 = int(o); m0 = int(round((o - y0) * 12)) + 1
        n += issue_all(only, origin=o_bin, iid=f"hindcast-{y0:04d}-{m0:02d}", horizon=SET["hindcast_horizon_years"], hindcast=True)
    print(f"[backfill] {n} new hindcast projections over {len(origins)} origins")


# ------------------------------------------------------------------------------------------------------- scoring
def _read_proj(d):
    a = np.genfromtxt(d / "projection.csv", delimiter=",", names=True)
    return a["date"].astype(float), a["x_arcsec"].astype(float) - 1j * a["y_arcsec"].astype(float)


def score():
    t, p, status = load_obs(); ok = complete_bins(t, status); t, p = t[ok], p[ok]
    bins = SET["horizon_bins"]; rows = []
    for edir in sorted(LEDGER.glob("*")):
        for idir in sorted(edir.glob("*")):
            meta = json.loads((idir / "manifest.json").read_text()); g, z = _read_proj(idir)
            origin = meta["data_end"]
            ls_dir = LEDGER / "ls_free" / idir.name
            gl, zl = _read_proj(ls_dir) if ls_dir.exists() else (None, None)
            last = p[np.argmin(abs(t - origin))]
            segs = [("post_origin", origin)] + ([("post_issue", meta["issued_year"])] if "hindcast_origin" not in meta else [])
            for seg, start in segs:
                for b0, b1 in zip(bins[:-1], bins[1:]):
                    sel = (g > start) & (g - origin > b0) & (g - origin <= b1)
                    idx = np.searchsorted(t, g[sel] - 1e-6)
                    hit = (idx < len(t)) & (np.abs(t[np.minimum(idx, len(t) - 1)] - g[sel]) < STEP / 4)
                    if hit.sum() < 3:
                        continue
                    obs = p[idx[hit]]; mod = z[sel][hit]
                    e = mod - obs; mse = float(np.mean(np.abs(e) ** 2))
                    row = dict(engine=meta["engine"], issue=idir.name, kind="hindcast" if "hindcast_origin" in meta else "live",
                               origin=round(origin, 4), segment=seg, h0=b0, h1=b1, n=int(hit.sum()),
                               rmse_mas=1e3 * mse ** 0.5, rmse_x_mas=1e3 * float(np.sqrt(np.mean(e.real ** 2))),
                               rmse_y_mas=1e3 * float(np.sqrt(np.mean(e.imag ** 2))),
                               ss_persistence=1 - mse / float(np.mean(np.abs(last - obs) ** 2)))
                    if gl is not None:
                        zl_sel = np.interp(g[sel][hit], gl, zl.real) + 1j * np.interp(g[sel][hit], gl, zl.imag)
                        mse_l = float(np.mean(np.abs(zl_sel - obs) ** 2))
                        row["ss_ls_free"] = 1 - mse / mse_l if mse_l > 0 else None
                    rows.append(row)
    SCORES.mkdir(exist_ok=True)
    keys = ["engine", "issue", "kind", "origin", "segment", "h0", "h1", "n", "rmse_mas", "rmse_x_mas", "rmse_y_mas", "ss_persistence", "ss_ls_free"]
    with open(SCORES / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    print(f"[score] {len(rows)} rows")
    return rows


def claims():
    t, p, status = load_obs(); ok = complete_bins(t, status); t, p = t[ok], p[ok]
    out = {}
    for cid, spec in CLAIMS.items():
        val = claim_value(spec, t, p) if status["data_end"] >= spec["evaluate_after"] else None
        # a provisional reading with the data so far (window truncated at the data end), for monitoring only
        prov = None
        try:
            sp = dict(spec)
            if spec["kind"] in ("cw_amp", "cw_phase"):
                sp["centre"] = min(spec["centre"], t[-1] - spec["window"] / 2)
            elif "end" in spec:
                sp["end"] = min(spec["end"], t[-1])
            prov = claim_value(sp, t, p)
        except Exception:  # noqa: BLE001
            prov = None
        preds = {}
        for edir in sorted(LEDGER.glob("*")):
            lives = sorted(d for d in edir.glob("live-*"))
            if lives:
                cp = json.loads((lives[-1] / "manifest.json").read_text()).get("claim_predictions", {}).get(cid)
                preds[edir.name] = dict(issue=lives[-1].name, prediction=cp, verdict=judge(spec, cp) if cp else "n/a")
        out[cid] = dict(statement=spec["statement"], evaluate_after=spec["evaluate_after"], status=judge(spec, val),
                        observed=val, provisional=prov, engine_predictions=preds)
    (HERE / "claims_status.json").write_text(json.dumps(out, indent=1))
    print("[claims]", {k: v["status"] for k, v in out.items()})
    return out


# ------------------------------------------------------------------------------------------------------- report
def report(rows=None, cl=None):
    rows = rows if rows is not None else score()
    cl = cl if cl is not None else claims()
    status = json.loads((OBS / "status.json").read_text())
    L = ["# Chandler wobble forecast ledger", "",
         f"Updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC. Data end {status['data_end']:.4f} "
         f"(feeds: {', '.join(f'{k} {v}' for k, v in status['feeds'].items())}). Error = |Δp|, the vector error of "
         "complex polar motion, in mas. `SS` = 1 − MSE(engine)/MSE(baseline); positive beats the baseline.", ""]
    L += ["## Live issues", "", "| engine | issue | data end | r train | notes | C1 regrowth | C2 phase | C3 Mm | C4 7/6 | C5 sidebands |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for edir in sorted(LEDGER.glob("*")):
        lives = sorted(edir.glob("live-*"))
        if not lives:
            continue
        m = json.loads((lives[-1] / "manifest.json").read_text())
        note = {"alias_cw": f"CW {m.get('cw_amp_mas', 0):.0f} mas", "ls_free": f"CW {m.get('cw_period_days', 0):.1f} d, {m.get('cw_amp_mas', 0):.0f} mas",
                "free_decay": f"CW {m.get('cw_amp_mas_at_origin', 0):.0f} mas at origin, Q {m.get('Q')}",
                "device": f"comb day {m.get('comb_day')}, Q {m.get('Q')}",
                "two_line_fm": f"CW line {m.get('cw_line_mas', 0):.0f}, F−M line {m.get('fm_line_mas', 0):.0f} mas"}.get(edir.name, "")
        cp = m.get("claim_predictions", {})
        def c(k, unit):
            v = cp.get(k)
            if not v:
                return ""
            return f"{v['value']:.0f}{unit}" + (f" ({v['amp_mas']:.0f} mas)" if unit == "°" and "amp_mas" in v else "")
        L.append(f"| {edir.name} | {lives[-1].name} | {m['data_end']:.3f} | {m.get('r_train', 0):.3f} | {note} | "
                 f"{c('C1_regrowth', ' mas')} | {c('C2_phase_lock', '°')} | {c('C3_mm_persists', '°')} | {c('C4_seven_sixths', '°')} | {c('C5_no_sidebands', ' mas')} |")
    L += ["", "Claim columns are each engine's prediction, from its projection spliced onto the observations (C3/C4: phase change, with amplitude).", ""]
    hind = [r for r in rows if r["kind"] == "hindcast" and r["segment"] == "post_origin"]
    if hind:
        L += ["## Rolling-origin hindcasts", "", f"Origins every {SET['hindcast_step_months']} months from {SET['hindcast_start']:.0f}; "
              "each uses only data at or before its origin. Median over origins.", "",
              "| engine | horizon (yr) | origins | median error (mas) | median SS vs persistence | median SS vs ls_free | share beating ls_free |",
              "|---|---|---|---|---|---|---|"]
        for e in E.ENGINES:
            for b0, b1 in zip(SET["horizon_bins"][:-1], SET["horizon_bins"][1:]):
                rr = [r for r in hind if r["engine"] == e and r["h0"] == b0]
                if not rr:
                    continue
                med = lambda k: np.nanmedian([r[k] for r in rr if r.get(k) is not None])  # noqa: E731
                share = np.mean([r["ss_ls_free"] > 0 for r in rr if r.get("ss_ls_free") is not None]) if e != "ls_free" else float("nan")
                vs_ls = "" if e == "ls_free" else f"{med('ss_ls_free'):.2f}"
                share_s = "" if e == "ls_free" else f"{share:.2f}"
                L.append(f"| {e} | {b0:g}–{b1:g} | {len(rr)} | {med('rmse_mas'):.1f} | {med('ss_persistence'):.2f} | {vs_ls} | {share_s} |")
        coll = [r for r in hind if 2008.0 <= r["origin"] <= 2013.0]
        if coll:
            L += ["", "### Through the collapse (origins 2008–2013, the CW falling from ~110 to ~25 mas)", "",
                  "| engine | 0–1 yr error (mas) | 1–3 yr error (mas) |", "|---|---|---|"]
            for e in E.ENGINES:
                a = [r["rmse_mas"] for r in coll if r["engine"] == e and r["h1"] <= 1.0]
                b = [r["rmse_mas"] for r in coll if r["engine"] == e and r["h0"] >= 1.0]
                if a or b:
                    L.append(f"| {e} | {np.median(a) if a else float('nan'):.1f} | {np.median(b) if b else float('nan'):.1f} |")
    L += ["", "## Registered claims", "", "| claim | evaluate after | status | provisional reading (data so far) | engines predicting PASS |",
          "|---|---|---|---|---|"]
    for cid, v in cl.items():
        prov = v["provisional"]; pv = "" if not prov else ", ".join(f"{k} {x:.1f}" for k, x in prov.items())
        passing = [e for e, d in v["engine_predictions"].items() if d["verdict"] == "PASS"]
        L.append(f"| **{cid}**: {v['statement']} | {v['evaluate_after']:.1f} | {v['status']} | {pv} | {', '.join(passing) or '—'} |")
    L += ["", "Claims: `claims.yml` (registered 2026-10-24). Engines: `targets.yml`. Method: `README.md`."]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print("[report] REPORT.md written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "observe", "issue", "backfill", "score", "claims", "report", "list"])
    ap.add_argument("-e", "--engines", nargs="*", help="subset of engines")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in CFG["engines"].items():
            print(k, v)
        for k, v in CLAIMS.items():
            print(k, "-", v["statement"])
        return
    if a.cmd in ("all", "observe"):
        observe()
    if a.cmd in ("all", "issue"):
        issue(a.engines)
    if a.cmd in ("all", "backfill"):
        backfill(a.engines)
    if a.cmd in ("all", "report"):
        report()
    elif a.cmd == "score":
        score()
    elif a.cmd == "claims":
        claims()


if __name__ == "__main__":
    main()
