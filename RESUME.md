# resume state (2026-11-06, after the 7x7 lineage, it009)

Run: 7x7-anchored, now 3-board. Work dir /workspace/go-run-7x7 on the
per-instance box (SSH_HOST/PORT in .env change every provision).
Everything through it009 is on HF under `7x7/` (both repos).

## where it stands
best.pt = it009 (cd0f650235ee), gated at it9: 0.623 pooled
  7x7 0.570 | 9x9 0.620 | 11x11 0.680   (100 games per board)
it8 gated 0.747 (7x7 0.690 | 9x9 0.720 | 11x11 0.830)
--gate-min 0.0: every round promotes (cold-start policy). Gate series:
  1.000 0.995 0.943 0.838 0.705 0.693 0.730 0.747 0.623

Heads at it007 (best score head of the whole project): pol 0.676
val 0.892 own 0.282 score 7.72     (9x9 trunk was 9.84 at it020)

## selfplay shape (it007, data from it006's net)
board  plies(b2)  cap   pass/g  dame  |score|
7x7     57 (49)  1.2%    7.6   2.02    2.5
9x9     92 (81)  6.2%    9.2   5.92    5.5
11x11  137 (121) 10.5%  12.5   7.10    7.5
7x7 is close to done (cap solved, length ~b2, |score| 2.5 = tight games).
9x9/11x11 lag on completeness (dame 6-7).

## utility: plain blend, no knobs
terminal_value = blend(lambda, wdl, margin, b)
             = lambda*wdl + (1-lambda)*tanh(margin/2b)
No settle/dame term. Four shapes and both signs were A/B'd on one fixed
net (it009, 400 games/arm) and every one made endings worse than leaving
it off: dame 7x7/9x9  off 5.23/4.43 | k=1 sign-corrected 9.44/6.34 |
k=1 sign-flipped 7.46/7.02 | sign-aware scale 5.27/5.77 | unconditional
6.16/-. Caps and game length moved with them. The dame a game leaves is
set by the play that leads there (score resolution), not by a terminal
charge, so the code is gone (no GO_SETTLE_* anywhere).

## next round
node_loop7.sh is set for equal 10k shares:
  --sizes 7:10000,9:10000,11:10000
  --gate-games 300 --gate-sizes 7:9.5,9:7.5,11:7.5   (100 per board)
  GO_UTIL_LAMBDA=0.8  GO_SWITCH_V=0 (fixed blend)  GO_SETTLE_W=0
  --buffer-keep 0, --gate-min 0.0
~100 min/round. Launch: bash scripts/node_run7x7_tmux.sh
New box: scripts/node_provision.sh + node_pip.sh (torch), then rebuild.

## open questions
- it9 regressed (0.623 vs it8 0.747). Was it the K=1 settle rule (it was on
  for it8/it9), the 11x11 share, or noise? Next rounds run with W=0, so the
  series restarts cleanly.
- 9x9/11x11 dame 6-7 and cap 5-10%: more data or more sims, not a new knob.
- Data volume beat every knob so far (it023: 2.3x rows -> +47 Elo).
  --buffer-keep 0.5 is the automatic version and has NOT been tried on this
  run yet.

---

# history (earlier: 9x9 trunk)

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
