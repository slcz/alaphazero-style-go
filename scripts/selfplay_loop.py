#!/usr/bin/env python3
"""selfplay_loop.py — one self-contained train/upgrade loop on a single
machine (ported from the chess loop; Go knobs, 8× D4 augmentation in
training, no opening book — games start empty).

    python scripts/selfplay_loop.py --work-dir /data/go-run \
        --sizes 7:30000 --iterations 20 --device cuda

Per iteration, in <work-dir>/itNNN/:
    spB/          the engine's GOSHARD1 shard per board size (+ DONE.json)
    train.pt      the trained checkpoint (+ train.json)
and one JSONL record per iteration appended to <work-dir>/loop.jsonl.
Every trained checkpoint becomes best.pt unconditionally; compare nets
out-of-band with scripts/gate_match.py.

`--init-ckpt` is optional: omitted (and no best.pt) = random-init start.
Delete an iteration's spB/ to regenerate; finished stages are skipped on
resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import zlib
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SELFPLAY = REPO / "engine" / "build" / "selfplay_nn"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def random_init_checkpoint(path: Path, seed: int) -> None:
    import torch
    from go_ai.net.model import GoNet, NetConfig
    from go_ai.train.trainer import Trainer, TrainConfig

    torch.manual_seed(seed)
    model = GoNet(NetConfig())
    Trainer(model, TrainConfig(device="cpu", compile=False,
                               seed=seed)).save(path)
    log(f"created random-init {path.name} "
        f"({sum(p.numel() for p in model.parameters()):,} params)")


def update_buffer(work: Path, it: int, keep: float, seed: int) -> dict:
    """Rebuild buffer.json from the rounds on disk (pure function of the
    iteration, so a resume reproduces it exactly).

    A round's games survive with probability keep ** (it - round): the new
    round in full, half of the previous, a quarter of the one before, ...
    (expected buffer ≈ keep/(1-keep) extra rounds). The same per-round RNG
    stream at a smaller p makes each trim a subset of the last."""
    rounds = []
    shards = sorted(list(work.glob("it*/selfplay/shard-0.spbin"))
                    + list(work.glob("it*/sp*/shard-0.spbin")))
    for f in shards:
        r = int(f.parent.parent.name[2:])
        done = f.parent / "DONE.json"
        if r > it or not done.exists():
            continue
        n = (json.loads(done.read_text()).get("games_done")
             or count_games(f.parent))
        if n:
            rounds.append((r, n, f.relative_to(work).as_posix()))
    rec = {"version": 1, "keep": keep, "seed": seed, "iteration": it,
           "rounds": []}
    newest = max((n for _r, n, _p in rounds), default=0)
    for r, n, rel in rounds:
        # stable across processes: builtin hash() is salted per run, which
        # would make a resumed round select a different subset
        rng = random.Random(seed * 7919 + r * 4096 + zlib.crc32(rel.encode()))
        p = keep ** (it - r)
        games = [g for g in range(n) if rng.random() < p]
        if len(games) < 0.005 * newest:      # < 0.5% of the draw: drop the
            continue                          # round (bounds the window)
        rec["rounds"].append({"round": r, "shard": rel, "count": n,
                              "games": games})
    path = work / "buffer.json"
    path.write_text(json.dumps(rec))
    log(f"it{it}: buffer {[(x['round'], len(x['games'])) for x in rec['rounds']]}"
        f" = {sum(len(x['games']) for x in rec['rounds'])} games")
    prune(work, it, {x["round"] for x in rec["rounds"]})
    return rec


def prune(work: Path, it: int, live: set) -> None:
    """Bound disk for a run that never ends: row caches of rounds the
    buffer dropped, and mid-training checkpoints of finished rounds."""
    for f in list(work.glob("it*/selfplay/rows-*.npz")) \
            + list(work.glob("it*/sp*/rows-*.npz")):
        if int(f.parent.parent.name[2:]) not in live:
            f.unlink(missing_ok=True)
    for f in work.glob("it*/train.ck.pt"):
        if int(f.parent.name[2:]) < it:
            f.unlink(missing_ok=True)


def pump(cmd: list[str], log_path: Path, env: dict[str, str] | None = None,
         poll: int = 60) -> int:
    """Run a command, streaming its log tail so a long stage is not
    silent."""
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, cwd=str(REPO))
        stop = threading.Event()

        def tail() -> None:
            while not stop.wait(poll):
                lines = [l for l in log_path.read_text().splitlines()
                         if l.strip()]
                if lines:
                    log(f"  … {lines[-1][:110]}")

        t = threading.Thread(target=tail, daemon=True)
        t.start()
        rc = proc.wait()
        stop.set()
        t.join(timeout=1)
    if rc != 0:
        log(f"FAILED ({rc}); tail of {log_path}:")
        for line in log_path.read_text().splitlines()[-15:]:
            print("    " + line)
    return rc


def count_games(unit_dir: Path) -> int:
    if not (unit_dir / "shard-0.spbin").exists():
        return 0
    from go_ai.selfplay.shard import read_shard
    return len(read_shard(unit_dir / "shard-0.spbin"))


def ensure_done(unit_dir: Path, seed: int) -> bool:
    """Write DONE.json for a shard that has no marker (a killed run)."""
    if not (unit_dir / "shard-0.spbin").exists():
        return False
    if (unit_dir / "DONE.json").exists():
        return False
    n = count_games(unit_dir)
    if n == 0:
        return False
    (unit_dir / "DONE.json").write_text(json.dumps(
        {"n_workers": 1, "unit_seed": seed, "games_done": n}) + "\n")
    log(f"recovered {n} games from an unfinished shard in {unit_dir}")
    return True


def max_plys_for(b: int) -> int:
    """Ply wall for a board: 2.5 x area. Scales with the board instead of a
    flat cap, so a 19x19 game is not adjudicated where a 9x9 game would
    only be half over."""
    return max(60, int(round(2.5 * b * b)))


def size_plans(args):
    """[(b, games, komi, rows, block_mb, pool, max_plys)] for this round.

    Mixed boards keep the wave TOKEN count constant (rows ∝ 1/b²), the node
    block ∝ b² (a 15x15 node is ~3x a 9x9 one) and the pool inside a host
    RAM budget, so VRAM and memory do not move with the board size."""
    out = []
    for spec in args.sizes.split(","):
        b, games = (int(x) for x in spec.split(":"))
        komi = 9.5 if b == 7 else 7.5
        rows = max(64, int(round(args.rows * 81.0 / (b * b))))
        block = max(2, int(round(args.block_mb * b * b / 81.0)))
        # A bank ships at the round boundary, and a round parks roughly one
        # row per live game, so a bank of `rows` only fills if
        # rows = k * pool; otherwise it ships at ~pool rows and the rest of
        # the compiled width is padding (11x11: 2048/2742 = 75 %, 13x13:
        # 1536/1963 = 78 %, both measured). Fixing the pool instead of rows
        # keeps the full token budget per wave.
        pool = max(64, rows // 2)
        pool = min(args.pool, pool, max(64, (12 * 1024) // block))
        maxply = max_plys_for(b)
        out.append((b, games, komi, rows, block, pool, maxply))
    return out


def unit_of(d: Path, b: int) -> Path:
    """Unit dir itNNN/spB (the buffer and prune globs also cover the
    legacy itNNN/selfplay dirs of the single-board era)."""
    return d / f"sp{b}"


def generate(args, best: Path, unit_dir: Path, seed: int, b: int,
             komi: float, rows: int, block_mb: int, pool: int,
             max_plys: int, games: int) -> int:
    # NN_B turns on the FP8 static-wave inference path (one graph per
    # board size; see go_ai/net/nn_worker.py setup()).
    env = dict(os.environ, NN_B=str(b))
    cmd = [str(SELFPLAY), str(best), str(unit_dir), str(games),
           str(b), str(komi), str(args.sims), str(pool),
           str(rows), str(args.sample_plys), str(max_plys),
           str(block_mb), args.device, str(seed), str(args.fast_sims),
           str(args.fast_p)]
    return pump(cmd, unit_dir.parent / "generate.log", env=env)


def train(args, work: Path, base: Path, out: Path) -> int:
    cmd = [sys.executable, "scripts/train_round.py",
           "--buffer", str(work / "buffer.json"),
           "--base-ckpt", str(base), "--out", str(out),
           "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--lr", str(args.lr), "--seed", str(args.seed),
           "--log-every", "10",
           "--device", args.device, "--json", str(out.parent /
                                                   "train.json")]
    return pump(cmd, out.parent / "train.log")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--init-ckpt")
    ap.add_argument("--iterations", type=int, default=20,
                    help="stop after this iteration (ignored with --forever)")
    ap.add_argument("--forever", action="store_true",
                    help="run until stopped; iteration numbers advance from "
                         "loop.jsonl on every restart")
    ap.add_argument("--sizes", required=True,
                    help="boards for generation as b:games, e.g. "
                         "9:15000,11:8000,13:5000,15:2000 or 7:30000; "
                         "writes itNNN/spB/")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--fast-sims", type=int, default=50)
    ap.add_argument("--fast-p", type=float, default=0.75)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--sample-plys", type=int, default=10)
    ap.add_argument("--block-mb", type=int, default=4,
                    help="MCTS node block per game in MB; a 19x19 node is "
                         "~4x a 9x9 one, so bigger boards want more")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--buffer-keep", type=float, default=1 / 3,
                    help="fraction of an old round kept per iteration "
                         "(1/3 → 1, 1/3, 1/9 ... of successive rounds)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not SELFPLAY.exists():
        log(f"engine not built: {SELFPLAY} (run `make -C engine nn`)")
        return 1
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    best = work / "best.pt"
    if not best.exists():
        if args.init_ckpt:
            init = Path(args.init_ckpt).expanduser()
            if not init.exists():
                log(f"init checkpoint not found: {init}")
                return 1
            shutil.copy(init, best)
            log(f"seeded best.pt from {init}")
        else:
            random_init_checkpoint(best, args.seed)
    log(f"work {work} | best {best.name} sha {sha256(best)[:12]} | "
        f"{'forever' if args.forever else args.iterations} x "
        f"({sum(p[1] for p in size_plans(args))} games @ {args.sizes} "
        f"{args.sims}/{args.fast_sims} sims -> {args.epochs} epoch(s) -> "
        f"promote)")

    # resume point: every completed iteration is in loop.jsonl, and stage
    # caches mean a restart never redoes work that landed
    it = 1
    if (work / "loop.jsonl").exists():
        done = [json.loads(l)["iteration"] for l in
                (work / "loop.jsonl").read_text().splitlines() if l.strip()]
        if done:
            it = max(done) + 1
            log(f"resuming at it{it:03d}")
    limit = None if args.forever else args.iterations
    while limit is None or it <= limit:
        d = work / f"it{it:03d}"
        d.mkdir(parents=True, exist_ok=True)
        ckpt = d / "train.pt"
        plans = size_plans(args)
        units = [(p, unit_of(d, p[0])) for p in plans]
        log(f"it{it}: sizes " + " ".join(
            f"{p[0]}x{p[0]}x{p[1]}g(rows {p[3]}, blk {p[4]}MB, "
            f"pool {p[5]})" for p in plans))
        t0 = time.time()

        games = sum(count_games(u) for _p, u in units)
        if not ckpt.exists():
            for plan, unit in units:
                b, ng, komi, rows, block, pool, maxply = plan
                ensure_done(unit, args.seed + it)
                have = count_games(unit)
                if have < ng:
                    if have:
                        log(f"it{it}: {b}x{b} shard has {have}/{ng}, "
                            f"regenerating")
                        shutil.rmtree(unit, ignore_errors=True)
                    log(f"it{it}: generating {ng} games at {b}x{b} "
                        f"(rows {rows}, block {block}MB, pool {pool})")
                    rc = generate(args, best, unit, args.seed + it, b, komi,
                                  rows, block, pool, maxply, ng)
                    if rc != 0:
                        return rc
                    log(f"it{it}: {b}x{b} generated {count_games(unit)} games")
                else:
                    log(f"it{it}: reusing {b}x{b} shard ({have} games)")
            games = sum(count_games(u) for _p, u in units)
            log(f"it{it}: {games} games in {time.time() - t0:.0f}s")
            buf = update_buffer(work, it, args.buffer_keep, args.seed)
            n_buf = sum(len(x["games"]) for x in buf["rounds"])
            log(f"it{it}: training on {n_buf} buffered games "
                f"from {len(buf['rounds'])} round(s)")
            rc = train(args, work, best, ckpt)
            if rc != 0:
                return rc

        # unconditional upgrade: every trained checkpoint becomes the
        # next iteration's teacher (copy-then-move so best.pt swaps
        # atomically; net-vs-net checks live in scripts/gate_match.py)
        shutil.copy(ckpt, work / "best.pt.next")
        shutil.move(work / "best.pt.next", best)
        log(f"it{it}: promoted {ckpt.name} -> best.pt "
            f"sha {sha256(best)[:12]}")

        rec = {"iteration": it, "games": games, "epochs": args.epochs,
               "elapsed_s": round(time.time() - t0),
               "ckpt": str(ckpt.relative_to(work)),
               "ckpt_sha256": sha256(ckpt) if ckpt.exists() else None,
               "best_sha256": sha256(best)[:16],
               "params": {**{k: getattr(args, k) for k in
                             ("sizes", "sims", "fast_sims", "fast_p",
                              "buffer_keep", "lr", "batch", "epochs")},
                          # the search blend is environment, not an argv
                          "util_lambda":
                              float(os.environ.get("GO_UTIL_LAMBDA", 0.5)),
                          "switch_v":
                              float(os.environ.get("GO_SWITCH_V", 0.95))}}
        with open(work / "loop.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        it += 1

    log(f"done — best.pt sha {sha256(best)[:12]} | records in "
        f"{work / 'loop.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
