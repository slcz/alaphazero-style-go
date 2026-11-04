# resume state (2026-11-04, after it019)

Node: 180.189.55.41:15594, work dir /workspace/go-run-9x9
it019 (mixed 9/11/13/15) runs until it records (~19:05 UTC); the stopat19
watchdog then kills the supervisor, loop and engine and drops it020.

## next round is already configured: PLAIN 9x9, 30 000 games
node_loop9.sh has no --sizes line any more, so `--games 30000 --b 9` applies:
~60 min generation, ~10 min train, ~4 min gate. Same utility as it019:
  GO_UTIL_LAMBDA=1.0, GO_SWITCH_V=0.95   (win/loss first, then pure score)
  GO_ROOT_NOISE_EPS=0  (root noise did nothing: it016 gated 0.5075)
  --buffer-keep 0      (fresh round only)
  gate: lambda 1.0 fixed, 400 games, promote bar 0.52

    bash /workspace/go/scripts/node_run9x9_tmux.sh      # resumes at it020

## why the 9x9 round
The mixed rounds plateaued (it017 0.570, it018 0.568) and the two-phase
switch showed no change in endgame settlement (41 % of games still end with
a group in atari, same as it018). Measured on it019's 9x9 games: dead stones
usually cancel (11 % both sides), mean |score change| from a cleanup sweep
1.8 points, median 0. A plain 9x9 round is ~6x faster than the mixed round,
so it is the cheap way to see whether more iterations fix pass valuation
and improve the ownership/score labels.

## history (9x9 gate, vs the previous base)
it012 0.735 | it013 0.650 | it014 0.603 | it015 0.603 | it016 0.5075 (noise,
rejected) | it017 0.570 (mixed) | it018 0.568 (mixed) | it019 pending
