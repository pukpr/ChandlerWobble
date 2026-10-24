#!/usr/bin/env python3
"""Download the IERS Rapid Service daily polar motion series (USNO finals2000A.all, 1973-now) into docs/data/.

The IERS datacenter (datacenter.iers.org) rejected scripted requests and hpiers.obspm.fr (C01/C04) timed out on
2026-10-23; the USNO mirror of the IERS Rapid Service/Prediction Centre series is used instead. Its measured rows
(flag 'I') are the combined IERS polar motion; rows flagged 'P' are predictions and are dropped on reading.
    python3 fetch_eop.py   -> data/finals2000A.all (not in the repository)
"""
import urllib.request
from pathlib import Path

URL = "https://maia.usno.navy.mil/ser7/finals2000A.all"
OUT = Path(__file__).resolve().parent / "data" / "finals2000A.all"


def main():
    OUT.parent.mkdir(exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        OUT.write_bytes(r.read())
    print(f"{OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
