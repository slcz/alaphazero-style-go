#!/bin/bash
# gtp_mac.sh — the Go engine as a GTP subprocess for Sabaki (MPS).
#
# Sabaki → Manage Engines → + → (Local), command:
#     /Users/sileizhang/go/scripts/gtp_mac.sh
# Sabaki spawns this itself; there is no port to connect to.
#
# knobs (env):
#   PUCT_CKPT   checkpoint (default ckpts/it006.pt)
#   GO_SIMS     sims per genmove (default 2000). MPS does ~860 sims/s
#               (the forward is ~0.8 ms/row and it dominates), so:
#               400 -> 0.5 s, 2000 -> 2.3 s, 5000 -> 6 s, 10000 -> 12 s
#   GO_MAXPLY   ply wall that triggers TT adjudication (default 400)
#   GO_BLOCK_MB node block in MB (default: auto, ~4 MB per 100 sims at
#               9x9 scaled by board area; 2000 sims on 9x9 -> ~77 MB)
#   GO_ANALYZE_SIMS  playouts per analyze session (default = GO_SIMS);
#               the stream stops there instead of growing the tree forever
#   GO_DEVICE   mps | cpu | dummy    (default mps)
#   PUCT_WALKERS concurrent evals (default 4)
#
# A single numeric ARGUMENT overrides GO_SIMS, so per-engine strength is
# set from Sabaki's "Arguments" field:  gtp_mac.sh 1000
set -e
cd "${0%/*}/.."        # no dirname: Sabaki may spawn us with an empty PATH
export PUCT_CKPT="${PUCT_CKPT:-$PWD/ckpts/it006.pt}"
export GO_SIMS="${GO_SIMS:-2000}"
# Sabaki "Arguments": a single number overrides GO_SIMS.
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    export GO_SIMS="$1"
fi
export GO_MAXPLY="${GO_MAXPLY:-400}"
export PUCT_WALKERS="${PUCT_WALKERS:-4}"
# torch comes from the chess project's venv (Python 3.9 + MPS): the
# engine was built against that same CLT Python and carries the venv's
# site-packages on its embedded sys.path.
exec engine/build/gtp "${GO_DEVICE:-mps}" 32
