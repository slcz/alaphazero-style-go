#!/usr/bin/env python3
"""selfplay_loop.py — one self-contained train/eval loop on a single
machine (ported from the chess loop; Go knobs, 8× D4 augmentation in
training, no opening book — games start empty with referee-randomized
openings in the gate).

    python scripts/selfplay_loop.py --work-dir /data/go-run \
        --b 7 --komi 9.5 --iterations 20 --games 30000 --device cuda

Per iteration, in <work-dir>/itNNN/:
    selfplay/     the engine's GOSHARD1 shard (+ DONE.json)
    train.pt      the trained checkpoint (+ train.json)
    eval.json     the gate result against best.pt
and one JSONL record per iteration appended to <work-dir>/loop.jsonl.

`--init-ckpt` is optional: omitted (and no best.pt) = random-init start.
Delete an iteration's selfplay/ to regenerate; finished stages are
skipped on resume.
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


def _hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def hf_enabled() -> bool:
    """Nodes that cannot reach HF (this one resolves it but times out) set
    HF_PUSH=0; the Mac pulls the artifacts and pushes instead."""
    return os.environ.get("HF_PUSH", "1") != "0"


def push_buffer(path: Path, it: int = 0) -> None:
    """The manifest is the one thing a node loss cannot rebuild cheaply
    (the shards come back from HF) — push it as soon as it changes."""
    if not hf_enabled():
        log("  buffer push skipped (HF_PUSH=0; pull from the Mac)")
        return
    try:
        from huggingface_hub import HfApi
        tok = _hf_token()
        if not tok:
            log("  buffer push skipped (no HF_TOKEN)")
            return
        HfApi(token=tok).upload_file(
            path_or_fileobj=str(path), path_in_repo="9x9/buffer.json",
            repo_id=os.environ.get("HF_MODEL_REPO",
                                   "chriszhang01/go-value-net"),
            repo_type="model",
            commit_message=f"buffer through it{it:03d}")
        log(f"  buffer pushed to HF 9x9/buffer.json")
    except Exception as e:                       # never fails the loop
        log(f"  buffer push failed ({type(e).__name__}: {e})")


def update_buffer(work: Path, it: int, keep: float, seed: int) -> dict:
    """Rebuild buffer.json from the rounds on disk (pure function of the
    iteration, so a resume reproduces it exactly).

    A round's games survive with probability keep ** (it - round): the new
    round in full, half of the previous, a quarter of the one before, ...
    (expected buffer ≈ keep/(1-keep) extra rounds). The same per-round RNG
    stream at a smaller p makes each trim a subset of the last."""
    rounds = []
    for f in sorted(work.glob("it*/selfplay/shard-0.spbin")):
        r = int(f.parent.parent.name[2:])
        done = f.parent / "DONE.json"
        if r > it or not done.exists():
            continue
        n = (json.loads(done.read_text()).get("games_done")
             or count_games(f.parent))
        if n:
            rounds.append((r, n))
    rec = {"version": 1, "keep": keep, "seed": seed, "iteration": it,
           "rounds": []}
    newest = rounds[-1][1] if rounds else 0
    for r, n in rounds:
        rng = random.Random(seed * 7919 + r)
        p = keep ** (it - r)
        games = [g for g in range(n) if rng.random() < p]
        if len(games) < 0.005 * newest:      # < 0.5% of the draw: drop the
            continue                          # round (bounds the window)
        rec["rounds"].append({
            "round": r, "shard": f"it{r:03d}/selfplay/shard-0.spbin",
            "count": n, "games": games})
    path = work / "buffer.json"
    path.write_text(json.dumps(rec))
    log(f"it{it}: buffer {[(x['round'], len(x['games'])) for x in rec['rounds']]}"
        f" = {sum(len(x['games']) for x in rec['rounds'])} games")
    prune(work, it, {x["round"] for x in rec["rounds"]})
    push_buffer(path, it)
    return rec


def prune(work: Path, it: int, live: set) -> None:
    """Bound disk for a run that never ends: row caches of rounds the
    buffer dropped, and mid-training checkpoints of finished rounds."""
    for f in work.glob("it*/selfplay/rows-*.npz"):
        if int(f.parent.parent.name[2:]) not in live:
            f.unlink(missing_ok=True)
    for f in work.glob("it*/train.ck.pt"):
        if int(f.parent.name[2:]) < it:
            f.unlink(missing_ok=True)


def push_iter(work: Path, it: int, rename: str | None = None) -> int:
    """Back up the finished iteration to HF *from the node* (checkpoint,
    summary json, gate SGFs to the model repo; the shard to the dataset
    repo). Nothing comes back to the Mac. `push.ok` memoizes it so a
    restart does not re-upload."""
    if not hf_enabled():
        log(f"it{it}: HF push skipped (HF_PUSH=0; pull from the Mac)")
        return 0
    mark = work / f"it{it:03d}" / "push.ok"
    if mark.exists() and (rename is None
                          or f"rename={rename}" in mark.read_text()):
        log(f"it{it}: already on HF — push skipped")
        return 0
    cmd = [sys.executable, "scripts/hf_push_iter.py", str(it),
           "--root", str(work)]
    if rename:
        cmd += ["--rename", rename]
    log(f"it{it}: pushing to HF" + (f" (+{rename})" if rename else ""))
    rc = pump(cmd, work / f"it{it:03d}" / "push.log")
    log(f"it{it}: HF push {'ok' if rc == 0 else f'FAILED ({rc})'}")
    if rc == 0:
        with open(mark, "a") as fh:
            fh.write(f"{time.strftime('%FT%T')} rename={rename}\n")
    return rc


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


def generate(args, best: Path, unit_dir: Path, seed: int) -> int:
    # NN_B turns on the FP8 static-wave inference path (one graph per
    # board size; see go_ai/net/nn_worker.py setup()).
    env = dict(os.environ, NN_B=str(args.b),
               NN_BF16="1", NN_COMPILE="1")
    cmd = [str(SELFPLAY), str(best), str(unit_dir), str(args.games),
           str(args.b), str(args.komi), str(args.sims), str(args.pool),
           str(args.rows), str(args.sample_plys), str(args.max_plys),
           str(args.block_mb), args.device, str(seed), str(args.fast_sims),
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



def gate(args, candidate: Path, best: Path, eval_json: Path) -> float | None:
    log_path = eval_json.with_suffix(".log")
    # opening diversity: sample ∝ visits for the first N plies (the
    # referee's 0-1 random moves alone would repeat openings)
    env = dict(os.environ, PUCT_SAMPLE_PLYS=str(args.gate_sample_plys))
    cmd = [sys.executable, "scripts/gate_match.py",
           "--ckpt-a", str(candidate), "--ckpt-b", str(best),
           "--games", str(args.gate_games // 2),
           "--a-sims", str(args.gate_sims), "--b-sims", str(args.gate_sims),
           "--b", str(args.b), "--komi", str(args.komi),
           "--max-plys", str(args.gate_max_plys),
           "--resign", str(args.gate_resign),
           "--device", args.device,
           # rows per bank: walkers × games is the exact park bound —
           # a 4096 landing zone would pad 5x per FP8 wave (gen's size)
           "--rows", str(int(os.environ.get("PUCT_WALKERS", "4"))
                         * (args.gate_games // 2)),
           "--json", str(eval_json),
           "--sgf-dir", str(eval_json.parent / "gate-sgf")]
    if pump(cmd, log_path, env=env) != 0:
        return None
    if eval_json.exists():
        return float(json.loads(eval_json.read_text())["score_a"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--init-ckpt")
    ap.add_argument("--b", type=int, default=7)
    ap.add_argument("--komi", type=float, default=9.5)
    ap.add_argument("--iterations", type=int, default=20,
                    help="stop after this iteration (ignored with --forever)")
    ap.add_argument("--forever", action="store_true",
                    help="run until stopped; iteration numbers advance from "
                         "loop.jsonl on every restart")
    ap.add_argument("--games", type=int, default=30000)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--fast-sims", type=int, default=50)
    ap.add_argument("--fast-p", type=float, default=0.75)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--sample-plys", type=int, default=10)
    ap.add_argument("--block-mb", type=int, default=4,
                    help="MCTS node block per game in MB; a 19x19 node is "
                         "~4x a 9x9 one, so bigger boards want more")
    ap.add_argument("--max-plys", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--buffer-keep", type=float, default=1 / 3,
                    help="fraction of an old round kept per iteration "
                         "(1/3 → 1, 1/3, 1/9 ... of successive rounds)")
    ap.add_argument("--gate-games", type=int, default=400)
    ap.add_argument("--gate-sims", type=int, default=200)
    ap.add_argument("--gate-max-plys", type=int, default=200)
    ap.add_argument("--gate-resign", type=float, default=0.0,
                    help="mover-view loss value that resigns (0 = off)")
    ap.add_argument("--gate-sample-plys", type=int, default=10,
                    help="gate engines sample moves for this many plies")
    ap.add_argument("--adopt-min", type=float, default=0.5)
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
        f"({args.games} games @ {args.b}x{args.b} "
        f"{args.sims}/{args.fast_sims} sims -> {args.epochs} epoch(s) -> "
        f"{args.gate_games}-game gate @ {args.adopt_min})")

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
        unit = d / "selfplay"
        d.mkdir(parents=True, exist_ok=True)
        ckpt = d / "train.pt"
        t0 = time.time()

        games = count_games(unit)   # defined on every path (record below)
        if not ckpt.exists():
            ensure_done(unit, args.seed + it)
            games = count_games(unit)
            if games < args.games:
                if games:
                    log(f"it{it}: shard has {games}/{args.games}, "
                        f"regenerating")
                    shutil.rmtree(unit, ignore_errors=True)
                log(f"it{it}: generating {args.games} games")
                rc = generate(args, best, unit, args.seed + it)
                if rc != 0:
                    return rc
                games = count_games(unit)
                log(f"it{it}: generated {games} games in "
                    f"{time.time() - t0:.0f}s")
            else:
                log(f"it{it}: reusing existing shard ({games} games)")
            buf = update_buffer(work, it, args.buffer_keep, args.seed)
            n_buf = sum(len(x["games"]) for x in buf["rounds"])
            log(f"it{it}: training on {n_buf} buffered games "
                f"from {len(buf['rounds'])} round(s)")
            rc = train(args, work, best, ckpt)
            if rc != 0:
                return rc

        # gate is EVAL-ONLY: the latest checkpoint always becomes the
        # next iteration's base (no accept/reject decision).
        eval_json = d / "eval.json"
        if eval_json.exists():
            score = float(json.loads(eval_json.read_text())["score_a"])
            log(f"it{it}: reusing cached gate {score:.3f}")
        else:
            log(f"it{it}: gating {ckpt.name} against best.pt (eval only)")
            score = gate(args, ckpt, best, eval_json)
        shutil.copy(ckpt, work / "best.pt.next")
        shutil.move(work / "best.pt.next", best)
        log(f"it{it}: gate {'FAILED' if score is None else f'{score:.3f}'}"
            f" -> promoted latest checkpoint")

        rec = {"iteration": it, "games": games, "epochs": args.epochs,
               "gate_score": score,
               "elapsed_s": round(time.time() - t0),
               "ckpt": str(ckpt.relative_to(work)),
               "ckpt_sha256": sha256(ckpt) if ckpt.exists() else None,
               "best_sha256": sha256(best)[:16],
               "params": {**{k: getattr(args, k) for k in
                             ("b", "komi", "sims", "fast_sims", "fast_p",
                              "buffer_keep", "lr", "batch", "epochs",
                              "gate_games")},
                          # the search blend is environment, not an argv
                          "util_lambda":
                              float(os.environ.get("GO_UTIL_LAMBDA", 0.5))}}
        # durability first: the checkpoint/summary/SGFs/shard leave the
        # node (gen0 aliases it001's net), THEN the iteration is recorded —
        # a crash in between re-runs the push-free path on restart
        push_iter(work, it, rename="gen0" if it == 1 else None)
        with open(work / "loop.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        it += 1

    log(f"done — best.pt sha {sha256(best)[:12]} | records in "
        f"{work / 'loop.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
