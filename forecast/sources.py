"""Polar motion observations for the Chandler wobble ledger.

    C01  IERS EOP C01, 1846-1961 (0.1 yr to 1890, 0.05 yr after)        ftp://hpiers.obspm.fr/iers/eop/eopc01/
    C04  IERS EOP 20 C04, daily from 1962 (the official series)          ftp://hpiers.obspm.fr/iers/eop/eopc04/
    USNO IERS Rapid Service finals2000A, measured rows after C04 ends     https://maia.usno.navy.mil/ser7/

The Paris Observatory https server timed out and datacenter.iers.org refused scripted requests (2026-10-23/24); the
FTP service and the USNO mirror answered. Downloads are cached in forecast/cache/ (not committed). A failed download
keeps the cached copy, so the ledger still runs, on older data, when a server is down.
"""
from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
FEEDS = {
    "c01": "ftp://hpiers.obspm.fr/iers/eop/eopc01/eopc01.iau2000.1846-now",
    "c04": "ftp://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now",
    "usno": "https://maia.usno.navy.mil/ser7/finals2000A.all",
}


def _download(url: str, dest: Path, tries: int = 3) -> bool:
    tmp = dest.with_suffix(".part")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}) if url.startswith("http") else url
            with urllib.request.urlopen(req, timeout=180) as r:
                tmp.write_bytes(r.read())
            tmp.replace(dest)
            return True
        except Exception:  # noqa: BLE001
            if shutil.which("curl"):
                rc = subprocess.run(["curl", "-4", "-sSfL", "-A", "Mozilla/5.0", "-m", "180", "-o", str(tmp), url],
                                    capture_output=True).returncode
                if rc == 0 and tmp.exists() and tmp.stat().st_size > 1000:
                    tmp.replace(dest)
                    return True
            time.sleep(3 * (k + 1))
    tmp.unlink(missing_ok=True)
    return False


FALLBACK = {"c01": HERE.parent / "docs" / "data" / "eopc01.iau2000.1846-now", "usno": HERE.parent / "docs" / "data" / "finals2000A.all"}


def refresh(log=print) -> dict:
    CACHE.mkdir(exist_ok=True)
    state = {}
    for name, url in FEEDS.items():
        dest = CACHE / name
        ok = _download(url, dest)
        if not ok and not dest.exists() and name in FALLBACK and FALLBACK[name].exists():
            shutil.copy(FALLBACK[name], dest); state[name] = "copied from docs/data"
            log(f"[observe] {name}: {state[name]}"); continue
        state[name] = "updated" if ok else ("cached" if dest.exists() else "missing")
        log(f"[observe] {name}: {state[name]}")
    return state


def _year(mjd):
    return 2000.0 + (np.asarray(mjd, float) - 51544.5) / 365.25


def read_c01():
    rows = [l.split()[:3] for l in (CACHE / "c01").read_text().splitlines() if l.strip() and not l.startswith("#")]
    a = np.array(rows, float)
    return _year(a[:, 0]), a[:, 1], a[:, 2]


def read_c04():
    t, x, y = [], [], []
    for l in (CACHE / "c04").read_text().splitlines():
        if l.startswith("#") or len(l.split()) < 7:
            continue
        p = l.split(); t.append(float(p[4])); x.append(float(p[5])); y.append(float(p[6]))
    return _year(t), np.array(x), np.array(y)


def read_usno():
    t, x, y = [], [], []
    for l in (CACHE / "usno").read_text().splitlines():
        if len(l) < 46 or l[16] != "I":
            continue
        t.append(float(l[7:15])); x.append(float(l[18:27])); y.append(float(l[37:46]))
    return _year(t), np.array(x), np.array(y)


def merged():
    """C01 before 1962, C04 daily from 1962, USNO measured rows after C04's last day. Arcsec."""
    parts = []
    if (CACHE / "c01").exists():
        t, x, y = read_c01(); m = t < 1962.0; parts.append((t[m], x[m], y[m], "C01"))
    end = 1962.0
    if (CACHE / "c04").exists():
        t, x, y = read_c04(); parts.append((t, x, y, "C04")); end = t[-1]
    if (CACHE / "usno").exists():
        t, x, y = read_usno(); m = t > end; parts.append((t[m], x[m], y[m], "USNO"))
    t = np.concatenate([p[0] for p in parts]); x = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts]); src = np.concatenate([[p[3]] * len(p[0]) for p in parts])
    o = np.argsort(t)
    return t[o], x[o], y[o], src[o]


def binned(t, x, y, step=0.05, t0=1846.0):
    """Bin means on a fixed grid (centres t0 + (k + 1/2) step); bins with no data are dropped."""
    k = np.floor((t - t0) / step).astype(int)
    n = np.bincount(k); ok = n > 0
    c = t0 + (np.arange(len(n)) + 0.5) * step
    bx = np.bincount(k, x) / np.maximum(n, 1); by = np.bincount(k, y) / np.maximum(n, 1)
    return c[ok], bx[ok], by[ok]
