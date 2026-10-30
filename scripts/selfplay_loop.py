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
    env = dict(os.environ, NN_BF16="1", NN_COMPILE="1")
    cmd = [str(SELFPLAY), str(best), str(unit_dir), str(args.games),
           str(args.b), str(args.komi), str(args.sims), str(args.pool),
           str(args.rows), str(args.sample_plys), str(args.max_plys),
           "4", args.device, str(seed), str(args.fast_sims),
           str(args.fast_p)]
    return pump(cmd, unit_dir.parent / "generate.log", env=env)


def train(args, unit_dirs: list[Path], base: Path, out: Path) -> int:
    cmd = [sys.executable, "scripts/train_round.py",
           "--data-dir", *[str(d) for d in unit_dirs],
           "--base-ckpt", str(base), "--out", str(out),
           "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--lr", str(args.lr), "--seed", str(args.seed),
           "--device", args.device, "--json", str(out.parent /
                                                   "train.json")]
    return pump(cmd, out.parent / "train.log")


def gate(args, candidate: Path, best: Path, eval_json: Path) -> float | None:
    log_path = eval_json.with_suffix(".log")
    cmd = [sys.executable, "scripts/gate_match.py",
           "--ckpt-a", str(candidate), "--ckpt-b", str(best),
           "--games", str(args.gate_games // 2),
           "--a-sims", str(args.gate_sims), "--b-sims", str(args.gate_sims),
           "--b", str(args.b), "--komi", str(args.komi),
           "--max-plys", str(args.max_plys),
           "--device", args.device, "--rows", str(args.rows),
           "--json", str(eval_json)]
    if pump(cmd, log_path) != 0:
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
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--games", type=int, default=30000)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--fast-sims", type=int, default=50)
    ap.add_argument("--fast-p", type=float, default=0.75)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--sample-plys", type=int, default=10)
    ap.add_argument("--max-plys", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--history", type=int, default=3,
                    help="train on the last N rounds' shards at full "
                         "weight (1 = newest only)")
    ap.add_argument("--gate-games", type=int, default=400)
    ap.add_argument("--gate-sims", type=int, default=200)
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
        f"{args.iterations} x ({args.games} games @ {args.b}x{args.b} "
        f"{args.sims}/{args.fast_sims} sims -> {args.epochs} epoch(s) -> "
        f"{args.gate_games}-game gate @ {args.adopt_min})")

    for it in range(1, args.iterations + 1):
        d = work / f"it{it:03d}"
        unit = d / "selfplay"
        d.mkdir(parents=True, exist_ok=True)
        ckpt = d / "train.pt"
        t0 = time.time()

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
            first = max(1, it - args.history + 1)
            round_dirs = [work / f"it{j:03d}" / "selfplay"
                          for j in range(first, it + 1)
                          if (work / f"it{j:03d}" / "selfplay" /
                              "DONE.json").exists()]
            log(f"it{it}: training on {len(round_dirs)} round(s)")
            rc = train(args, round_dirs, best, ckpt)
            if rc != 0:
                return rc

        log(f"it{it}: gating {ckpt.name} against best.pt")
        score = gate(args, ckpt, best, d / "eval.json")
        if score is None:
            log(f"it{it}: gate failed; keeping the current best")
            adopted = False
        else:
            adopted = score > args.adopt_min
            if adopted:
                shutil.copy(ckpt, work / "best.pt.next")
                shutil.move(work / "best.pt.next", best)
            log(f"it{it}: gate {score:.3f} -> "
                f"{'ADOPTED' if adopted else 'kept previous best'}")

        rec = {"iteration": it, "games": games, "epochs": args.epochs,
               "gate_score": score, "adopted": adopted,
               "adopt_min": args.adopt_min,
               "elapsed_s": round(time.time() - t0),
               "ckpt": str(ckpt.relative_to(work)),
               "ckpt_sha256": sha256(ckpt) if ckpt.exists() else None,
               "best_sha256": sha256(best)[:16],
               "params": {k: getattr(args, k) for k in
                          ("b", "komi", "sims", "fast_sims", "fast_p",
                           "history", "lr", "batch", "epochs",
                           "gate_games")}}
        with open(work / "loop.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    log(f"done — best.pt sha {sha256(best)[:12]} | records in "
        f"{work / 'loop.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
