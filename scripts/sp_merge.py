#!/usr/bin/env python3
"""sp_merge.py — concatenate GOSHARD1 shards, renumbering game_index.

    python3 scripts/sp_merge.py out.spbin a.spbin b.spbin ...

The file layout is a 24-byte header then self-delimiting game records, so a
merge is a byte copy of the record bodies with the first u32 (game_index) of
each rewritten to stay unique — ingest dedups on (seed, game_index), so
without renumbering the second shard would silently vanish. A truncated
trailing record is dropped (same rule the readers use).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

HDR = 24


def records(path: Path):
    """Yield (game_index, n_plies, body_bytes) for each whole record."""
    data = path.read_bytes()
    if len(data) < HDR or data[:8] != b"GOSHARD1":
        raise SystemExit(f"{path}: not a GOSHARD1 shard")
    off = HDR
    n = len(data)
    while off + 20 <= n:
        gi, n_plies, b = struct.unpack_from("<IIB", data, off)
        end_kind = data[off + 9]
        n_sims = struct.unpack_from("<H", data, off + 10)[0]
        off += 20 + 2 * b * b               # fixed part + ownership
        ok = True
        for _ in range(n_plies):            # ply blocks
            if off + 11 > n:
                ok = False
                break
            n_pi = struct.unpack_from("<H", data, off + 3)[0]
            off += 11 + 2 * n_pi
            if off > n:
                ok = False
                break
        if not ok:
            break                           # truncated tail: drop it
        yield gi, n_plies, off


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, srcs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    hdr = srcs[0].read_bytes()[:HDR]
    bodies, n_total = [], 0
    for s in srcs:
        data = s.read_bytes()
        if data[:8] != b"GOSHARD1":
            raise SystemExit(f"{s}: not a GOSHARD1 shard")
        prev = HDR
        for _gi, _np, end in records(s):
            bodies.append(data[prev:end])
            prev = end
            n_total += 1
    # rewrite game_index (first u32 of each record) to be dense
    with out.open("wb") as f:
        f.write(hdr)
        for i, body in enumerate(bodies):
            f.write(struct.pack("<I", i) + body[4:])
    print(f"{out}: {n_total} games from {len(srcs)} shards "
          f"({out.stat().st_size / 2**20:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
