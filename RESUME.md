# resume state (2026-11-04 morning)

Node: 180.189.55.43:52591, work dir /workspace/go-run-9x9
Stopped after it018 (stopat18 watchdog); it018 records ~06:15 UTC.

## what is on disk
- best.pt = it017 (3795715ef233) unless it018 gates >= 0.52
- it011..it018 shards + checkpoints; HF has through it017 (watcher)
- mixed plan live: --sizes 9:15000,11:5000,13:5000,15:5000, lambda 0.8, noise off

## resume
    bash /workspace/go/scripts/node_run9x9_tmux.sh
resumes at it019. The supervisor starts a NEW selfplay_loop process, so the
pool fix (pool = rows/2 -> fills the compiled wave instead of padding 22%)
takes effect. Expect plans: rows/pool 4096/2048, 2742/1371, 1963/981, 1475/737.

## numbers so far (9x9 gate vs previous base)
it013 0.650 | it014 0.603 | it015 0.603 | it016 0.5075 (noise, rejected)
it017 0.570 (mixed sizes, promoted) -- 811k rows, 34% more than 9x9-only
