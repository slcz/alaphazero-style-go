#!/bin/bash
# ship_src.sh — package the working tree for the node.
#
#   bash scripts/ship_src.sh [out.tgz]
#   scripts/salad.sh put /tmp/go-src.tgz /workspace/go-src.tgz
#   scripts/salad.sh ssh 'cd /workspace/go && tar xzf /workspace/go-src.tgz'
#
# Ships the working tree, NOT git, the human corpora, the SGF dumps, or
# the LOCAL build dir (engine/build holds macOS binaries — a node must
# build its own; `make -C engine all nn`).
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/tmp/go-src.tgz}
find . -type f \
    -not -path './.git/*' -not -path '*/__pycache__/*' \
    -not -path './data/*' -not -path './sup/*' \
    -not -path './engine/build/*' -not -path './build/*' \
    -not -path './gate_sgf_it001/*' -not -path './gen0_sgf/*' \
    -not -path './rand0_sgf/*' \
    -not -name '*.pdf' -not -name 'fm_*.png' -not -name '.DS_Store' \
    -print | tar -czf "$OUT" -T -
echo "$OUT: $(du -h "$OUT" | cut -f1), $(tar tzf "$OUT" | wc -l) files"
tar tzf "$OUT" | grep -E 'shard_to_sgf|go_ai/data/sgf.py|engine/build' || true
