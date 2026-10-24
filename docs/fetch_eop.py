#!/usr/bin/env python3
"""Download the IERS polar motion series into docs/data/:
  finals2000A.all            IERS Rapid Service daily x, y, LOD, 1973-now (USNO mirror)
  eopc01.iau2000.1846-now    IERS EOP C01 x, y, 1846-now (0.1 yr to 1890, 0.05 yr after), Paris Observatory FTP

The IERS datacenter (datacenter.iers.org) rejected scripted requests and hpiers.obspm.fr (C01/C04) timed out on
2026-10-23; the USNO mirror of the IERS Rapid Service/Prediction Centre series is used instead. Its measured rows
(flag 'I') are the combined IERS polar motion; rows flagged 'P' are predictions and are dropped on reading.
    python3 fetch_eop.py   -> data/finals2000A.all, data/eopc01.iau2000.1846-now (not in the repository)
"""
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
FILES = {"finals2000A.all": "https://maia.usno.navy.mil/ser7/finals2000A.all",
         # the https server (hpiers.obspm.fr) timed out on 2026-10-23/24; its FTP service answers
         "eopc01.iau2000.1846-now": "ftp://hpiers.obspm.fr/iers/eop/eopc01/eopc01.iau2000.1846-now"}


def download(url, dest, tries=3):
    """urllib first, then curl; the existing file is kept unless a new copy arrives complete."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}) if url.startswith("http") else url
            with urllib.request.urlopen(req, timeout=180) as r:
                tmp.write_bytes(r.read())
            break
        except Exception as e:  # noqa: BLE001
            if shutil.which("curl"):
                rc = subprocess.run(["curl", "-4", "-sSfL", "-A", "Mozilla/5.0", "-m", "180", "-o", str(tmp), url]).returncode
                if rc == 0 and tmp.exists() and tmp.stat().st_size > 1000:
                    break
            print(f"  attempt {k + 1} failed for {url}: {e}"); time.sleep(5 * (k + 1))
    else:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


def main():
    DATA.mkdir(exist_ok=True)
    for name, url in FILES.items():
        ok = download(url, DATA / name)
        state = "updated" if ok else ("FAILED, kept previous copy" if (DATA / name).exists() else "FAILED")
        size = (DATA / name).stat().st_size if (DATA / name).exists() else 0
        print(f"{DATA / name}: {state} ({size} bytes)")


if __name__ == "__main__":
    main()
