#!/usr/bin/env python3
"""download_kgs.py — fetch KGS high-dan archives from u-go.net.

    python scripts/download_kgs.py --data-dir data/kgs --years 2012 2013 2014

Downloads one zip per year (6d+/7d+ games, SGF, variations stripped),
extracts to <data-dir>/kgs-19-<year>/. ~10 MB per year, ~13k games.
Source: https://www.u-go.net/gamerecords/ (KGS, with Bill Shubert's
permission). Files already on disk are kept.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# per-year game counts are part of the URL; the ones we use
_COUNTS = {2012: 13665, 2013: 13783, 2014: 13029}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/kgs")
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2012, 2013, 2014])
    args = ap.parse_args()

    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    for y in args.years:
        if y not in _COUNTS:
            print(f"no URL known for {y}; see "
                  f"https://www.u-go.net/gamerecords/", file=sys.stderr)
            return 2
        dest = out / f"kgs-19-{y}"
        if dest.exists():
            n = len(list(dest.glob("*.sgf")))
            print(f"{dest.name}: {n} games (kept)")
            continue
        url = f"https://dl.u-go.net/gamerecords/KGS-{y}-19-{_COUNTS[y]}-.zip"
        zpath = out / f"KGS-{y}.zip"
        if not zpath.exists():
            print(f"fetching {url}", flush=True)
            urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        zpath.unlink()
        n = len(list(dest.glob("*.sgf")))
        print(f"{dest.name}: {n} games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
