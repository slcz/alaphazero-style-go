# Training / Selfplay / Eval Loop

One self-contained bootstrap loop: `scripts/selfplay_loop.py` runs N
iterations of **generate → train → evaluate → back up** on a single GPU
box (the salad RTX 5090 node). The node holds HF_TOKEN in
`/workspace/go/.env` and does its own uploads.

    # on the node (tmux survives ssh disconnects):
    bash /workspace/go/scripts/node_run9x9_tmux.sh

`node_restore_fresh.sh` is the fresh-round restore: newest
`9x9/itNNN-train.pt` from HF becomes `best.pt`, `loop.jsonl` comes back from
HF (pushed with every iteration), and no shards or caches are fetched. With
`--buffer-keep 0` the manifest then holds only the round just generated, so
each round trains on fresh on-policy data only. Set `--buffer-keep` back to
1/3 in `scripts/node_loop9.sh` to resume the decaying buffer.

It runs FOREVER: `node_loop9.sh` supervises it and restarts after a crash
(killing orphaned stage children first), the loop resumes at the next
unrecorded iteration from `loop.jsonl`, and `push.ok` keeps it from
re-uploading rounds that already landed. To stop it: `tmux kill-session
-t go9`.

Each iteration backs itself up to HF as soon as its gate lands — the node
pushes, nothing comes back to the Mac (`hf_push_iter.py --root <work>` is
called by the loop): the checkpoint, `itNNN-summary.json` and the gate SGF
pack to the model repo, the shard to the dataset repo.

## Configuration (9x9 current)

| knob | value | where |
|---|---|---|
| board / komi / maxply | 9 / 7.5 / 250 | `node_run9x9_tmux.sh` args |
| concurrent games (pool) | 2048 | `--pool` |
| landing zone | 4096 rows/bank | `--rows` (gen) |
| walkers per game | 4 | `PUCT_WALKERS` |
| mux walk tasks | 8 | `PUCT_TASKS` |
| sims per ply (PCR) | 200 (25% of plies) / 40 (75%) | `--sims --fast-sims --fast-p` |
| training plies | ONLY slow-cap plies (`ply.sims == n_sims`) | `selfplay_data.ply_trainable` |
| optimizer | Muon lr 0.01 + companion AdamW 2e-4 | `train_round.py` defaults |
| batch / epochs | 3072 / 1 | `--batch --epochs` |
| replay buffer | **off** for the fresh round (`kept = 0`: new games only) | `--buffer-keep` |
| FP8 inference | static waves, one graph per board size | `NN_B=<b> NN_COMPILE=1 NN_BF16=1` |
| gate | 400 games (2×200, colors swapped), 200 sims, argmax after 10 sampled plies, resign OFF, 200-ply cap, rows 800 | `selfplay_loop.gate()` |
| promote policy | EVAL-ONLY: latest checkpoint always becomes next base (no accept/reject) | `selfplay_loop.py` |

## Stage 1 — selfplay (engine: `selfplay_nn`)

`engine/build/selfplay_nn <ckpt> <out_dir> 30000 9 7.5 200 2048 4096 10
250 4 cuda <seed> 40 0.75`

- 2048 games in lockstep on one mux; every mux step draws ONE budget for
  the whole pool: 200 sims with p=0.25, else 40 (PCR, KataGo §4).
  Playout-cap plies are stored in the shard with `ply.sims` and are the
  only ones trained.
- Search: RPO MCTS (`puct.c`), pi-bar sampling at every descent, 4
  walkers/game, double-banked eval landing zone batched across the
  pool, FP8 static-wave inference on the 5090.
- Memory: byte-arena node blocks; children are 8B `PuctEdge` arrays,
  materializing a 20B `PuctNode` on first visit (`PuctEdge.child`).
- Move selection: sampled pi-bar for the first `sample_plys` (10),
  argmax after.
- Output: one append-per-game shard `itNNN/selfplay/shard-0.spbin`
  (crash-safe to the last in-flight game) + `DONE.json` when 30K games
  are in. Wave telemetry in the log when `NN_WAVE_STATS=1` (fill %,
  rows/wave).
- With a trained net, "fast play" (nsim=0 policy-only plies) can be
  enabled later; those plies are recorded but never trained.

## Replay buffer (`buffer.json`)

The loop keeps a *game-level* replay buffer instead of training on the
newest round only. After a round's shard is DONE, `selfplay_loop.py`
writes `<work>/buffer.json`:

    {"version": 1, "keep": 0.333, "iteration": 4,
     "rounds": [{"round": 3, "shard": "it003/selfplay/shard-0.spbin",
                 "count": 30000, "games": [12, 45, ...]}, ...]}

- A round's games survive with probability `keep ** (iteration - round)`:
  the new round in full, half of the previous, a quarter of the one
  before it. Expected buffer ≈ 1.5 rounds (45K games) at `keep = 1/3`,
  i.e. +50% training per round instead of +100%.
- It is a PURE FUNCTION of (iteration, round, seed) and the rounds on
  disk (`DONE.json` gives each round's game count), so a resume rebuilds
  exactly the same manifest and the trim only ever removes games.
- Pushed to HF as `9x9/buffer.json` right after it is written — it is the
  one artifact a node loss cannot rebuild cheaply (the shards come from
  HF). `node_restore_hf.sh` pulls it back.
- `train_round.py --buffer` parses ONLY the listed games (the rest of each
  shard is fast-forwarded), so the buffer costs no extra parsing.

## Stage 2 — training (`scripts/train_round.py`)

- Rows: every slow-cap ply of every buffered game (≈25% of plies;
  ~1.0-1.3M rows per 30K games, ~2.5M over the full buffer).
- 8× D4 symmetry augmentation sampled on the fly; steps = rows×8×epochs/batch.
- Losses (KataGo-style): policy CE + value one-hot (win/loss, no jigo —
  komi always fractional) + opponent-reply aux (0.15) + ownership (1.5)
  + score pdf (0.02) + score cdf (0.02). Muon on 2D matrices ≥64,
  AdamW on the rest; grad clip 1.0; bf16 autocast; torch.compile.
- Warm start: `--base-ckpt best.pt` (previous iteration's net + optimizer
  state). `train.ck.pt` written every 500 steps (`--save-every`).
- Off-policy note: an old round's π̄ comes from an older net. At
  `keep = 1/3` a 3-round-old round carries 4% of the draws, so it is
  mild; the value/score/ownership labels do not age at all.
- Known node quirk: cuBLAS bf16 GEMM fails at batch 4096 on this
  2.8.0/2.11+cu128 stack — batch 3072 verified.

## Stage 3 — evaluation (`scripts/gate_match.py`)

- Two `engine_mux` processes (candidate A, best B), 200 games each wave,
  wave 1 swaps colors. Referee plays r∈{0,1} random opening moves,
  then engines think at 200 sims/move.
- `PUCT_SAMPLE_PLYS=10`: both engines sample ∝ visits for 10 plies
  (opening diversity; measured: 392/392 unique ply-10 board states).
  Argmax-visits after — eval is deterministic given openings.
- Resign OFF (0.0) for now; maxply 200; TT scoring, fractional komi ⇒
  no jigo.
- Result: `eval.json` (`score_a` = candidate win rate) + one SGF per
  game (`gate-sgf/`, PB/PW/RE set). Score margins saturate at
  whitewash values (±73.5/±88.5 on 9x9) when nets are far apart —
  win-rate is the informative metric then.
- After the gate the checkpoint is promoted unconditionally; gate is
  purely measurement. Cached: `eval.json` existing ⇒ gate skipped on
  resume.

## Iteration bookkeeping

- `loop.jsonl` (work dir): one record per iteration — games, gate
  score, checkpoint sha, elapsed, buffer params. It is also the resume
  point (and is written AFTER the HF push, so a crash in between retries
  nothing). `itNNN/push.ok` memoizes the upload.
- Disk stays bounded: rounds whose kept games fall below 0.5% of the
  newest round leave the buffer, and their row caches + old
  `train.ck.pt` files are deleted.
- Node layout: `/workspace/go-run-9x9/{best.pt, it001/, it002/, ...}`,
  each `itNNN/` = `{train.pt, train.json, train.log, eval.json,
  generate.log, selfplay/shard-0.spbin, gate-sgf/}`.
- HF model repo `chriszhang01/go-value-net` @ `9x9/`: `buffer.json`,
  `gen0.pt` (alias
  of it001), `itNNN-train.pt`, `itNNN-summary.json` (train loss
  start/end + head metrics + gate result), `itNNN-eval.json`, match
  SGF packs. HF dataset repo `chriszhang01/go-selfplay` @ `9x9/itNNN/`:
  the raw shard.
- `gen0.pt` = it001's net = sha `1f9bd854…`; it002 net sha
  `42ef3a6f…`. Match results so far: it1 vs random 0.500, it2 vs it1
  1.000, it2 vs random 0.995.

## Resume after instance loss (nodes are cattle)

1. Ship repo: `tar` + `salad.sh put`, extract on node (or re-run the
   ship step of `node_restore1.sh`).
2. `bash /workspace/go/scripts/node_restore1.sh` — builds engine,
   checks torch (must be **2.11.0+cu128**; `node_torch211.sh` upgrades).
3. Restore artifacts from HF (`node_restore_hf.sh` does this ON the
   node, reading HF_TOKEN from `/workspace/go/.env`):
   `hf_hub_download` `9x9/itNNN-train.pt`, `9x9/it001-eval.json`
   (model repo) and `9x9/itNNN/shard-0.spbin` (dataset, `repo_type=
   "dataset"`), scp up, then `node_restore3.sh` places them, writes
   `DONE.json` and `best.pt = train.pt`.
4. `bash /workspace/go/scripts/node_run9x9_tmux.sh <next_it>` — cached
   stages are skipped (DONE-gated shard, existing train.pt/eval.json).

## Gotchas (learned the hard way)

- `salad.sh ssh 'cat file' > local` corrupts binaries (gateway banner
  on stdout) — use `salad.sh get` (scp) for anything binary.
- The gateway execs without a shell: pipes/`&&`/`&` break — put
  compound logic in a script file and `bash` it. Background children
  get reaped unpredictably — use the tmux launcher for long runs.
- Zombies (`Z` state) are this container's init not reaping — cosmetic.
- `node_cleanup.sh` keeps the work dir (FORCE_RM=1 to wipe) — wiping
  cost a full it1 regeneration on 2026-10-31.
- Ship with `scripts/ship_src.sh` — it drops `engine/build` (macOS
  binaries; the node builds its own) and the corpora. A hand-rolled
  `tar --exclude='*_sgf*'` silently ate `scripts/shard_to_sgf.py` and
  broke the gate.
- Landing zone must match pool: gate rows = walkers × games (800);
  mismatching (4096) pads FP8 waves ~5x → 10x slowdown.
- Score-lattice: TT margins are integers + komi ⇒ all `X.5` values;
  whitewash constants ±73.5/±88.5 dominate lopsided matches.
