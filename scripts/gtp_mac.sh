#!/bin/bash
# gtp_mac.sh — the Go engine as a GTP subprocess for Sabaki (MPS).
#
# Sabaki → Manage Engines → + → (Local), command:
#     /Users/sileizhang/go/scripts/gtp_mac.sh
# Sabaki spawns this itself; there is no port to connect to.
#
# knobs (env):
#   PUCT_CKPT   checkpoint (default ckpts/it006.pt)
#   GO_SIMS     sims per genmove (default 400)
#   GO_MAXPLY   ply wall that triggers TT adjudication (default 400)
#   GO_DEVICE   mps | cpu | dummy    (default mps)
#   PUCT_WALKERS concurrent evals (default 4)
set -e
cd "$(dirname "$0")/.."
export PUCT_CKPT="${PUCT_CKPT:-$PWD/ckpts/it006.pt}"
export GO_SIMS="${GO_SIMS:-400}"
export GO_MAXPLY="${GO_MAXPLY:-400}"
export PUCT_WALKERS="${PUCT_WALKERS:-4}"
# torch comes from the chess project's venv (Python 3.9 + MPS): the
# engine was built against that same CLT Python and carries the venv's
# site-packages on its embedded sys.path.
exec engine/build/gtp "${GO_DEVICE:-mps}" 32
