#!/bin/bash
# gtp.sh — the Go engine as a GTP subprocess for any GTP platform.
#
# knobs (env):
#   PUCT_CKPT   checkpoint (default ckpts/best.pt)
#   GO_SIMS     MCTS sims per genmove (default 10000).
#   GO_MAXPLY   ply wall that triggers TT adjudication (default 1600)
#   GO_BLOCK_MB node block in MB (default: auto, ~4 MB per 100 sims at
#               9x9 scaled by board area; 2000 sims on 9x9 -> ~77 MB)
#   GO_ANALYZE_SIMS  playouts per analyze session (default = GO_SIMS);
#               the stream stops there instead of growing the tree forever
#   GO_DEVICE   cpu | dummy    (default cpu)
#   PUCT_WALKERS concurrent evals (default 256)
#   GO_UTIL_LAMBDA utility = λ * win/loss + (1-λ) * tanh(score_margin / 2b)
#                  controls how MCTS values win/loss prob vs. score margin
#                  (default 0.9)
#
# A single numeric ARGUMENT overrides GO_SIMS. gtp.sh 20000
set -e
cd "${0%/*}/.."        # no dirname: Sabaki may spawn us with an empty PATH
export PUCT_CKPT="${PUCT_CKPT:-$PWD/ckpts/it023.pt}"
export GO_SIMS="${GO_SIMS:-10000}"
export GO_UTIL_LAMBDA="${GO_UTIL_LAMBDA:-0.9}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    export GO_SIMS="$1"
fi
export GO_MAXPLY="${GO_MAXPLY:-1600}"
export PUCT_WALKERS="${PUCT_WALKERS:-256}"
exec engine/build/gtp "${GO_DEVICE:-cpu}" ${PUCT_WALKERS}
