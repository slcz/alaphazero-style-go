"""publish_web_net.py — export a checkpoint as the web GUI's latest net.

The browser cannot run a `.pt` checkpoint, so the web build needs one
ONNX file per board size. This script does the whole publish:

  1. export net-{b}.onnx for every requested size,
  2. run the torch↔onnxruntime parity gate on real positions,
  3. write a `latest.json` pointer (checkpoint + per-size files),
  4. publish the files as a GitHub release, and/or upload them to the
     HuggingFace model repo.

The Pages workflow turns the newest `net-*` release into
`/web/net/latest.json` plus `/web/net/<tag>/net-{b}.onnx`; the GUI reads
that pointer at load time, so a new checkpoint reaches the page without
a code change.

    # publish it026 to GitHub Releases (Pages picks it up automatically)
    python scripts/publish_web_net.py ckpts/it026.pt \
        --checkpoint-url https://huggingface.co/chriszhang01/gonet/blob/main/checkpoints/it026-train.pt

    # local only: write web/net/latest.json + web/net/<tag>/ for dev
    python scripts/publish_web_net.py ckpts/it026.pt --no-upload

    # also refresh the private HF fallback (needs HF_TOKEN)
    python scripts/publish_web_net.py ckpts/it026.pt --hf-repo chriszhang01/gonet

The exporter needs torch + onnx + onnxruntime and the native engine
(`make -C engine all`) for the parity gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.export_onnx import export_onnx, parity_gate  # noqa: E402

DEFAULT_SIZES = "7,9,13,19"
DEFAULT_HF_REPO = "chriszhang01/gonet"
DEFAULT_CHECKPOINTS_URL = \
    "https://huggingface.co/chriszhang01/gonet/blob/main/checkpoints"


def _env(key: str) -> str | None:
    v = os.environ.get(key)
    if v:
        return v
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return None


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip())
    return s.strip("-") or "net"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model(ckpt: Path):
    import torch
    # checkpoints pickle optimizer state that references the training
    # env's torch.optim.Muon monkey-patch; inference only needs the model
    if not hasattr(torch.optim, "Muon"):
        def _swallow(self, *a, **k):
            self.__dict__.update(k)
        torch.optim.Muon = type("Muon", (object,), {
            "__init__": _swallow,
            "__setstate__": _swallow,
            "load_state_dict": _swallow,
        })
    from go_ai.train.trainer import Trainer
    return Trainer.load(ckpt, device="cpu")


def _gh(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], env=env, text=True,
                          capture_output=True)


def _release_exists(tag: str, repo: str, env: dict[str, str]) -> bool:
    return _gh(["release", "view", tag, "--repo", repo], env).returncode == 0


def _publish_github(tag: str, files: list[Path], notes: str, repo: str,
                    force: bool, dry_run: bool) -> bool:
    env = dict(os.environ)
    if not env.get("GH_TOKEN") and not env.get("GITHUB_TOKEN"):
        key = _env("GITHUB_KEY")
        if key:
            env["GH_TOKEN"] = key
    if not repo:
        r = _gh(["repo", "view", "--json", "nameWithOwner",
                 "--jq", ".nameWithOwner"], env)
        if r.returncode != 0:
            raise SystemExit(f"cannot determine GitHub repo: {r.stderr.strip()}")
        repo = r.stdout.strip()

    exists = _release_exists(tag, repo, env)
    if exists and not force:
        print(f"GitHub release {tag} already exists on {repo} — skipping "
              f"the upload (use --force to replace assets)")
        return False
    if dry_run:
        print(f"[dry-run] would {'update' if exists else 'create'} "
              f"GitHub release {tag} on {repo} with "
              f"{len(files)} assets")
        return True

    if exists:
        r = _gh(["release", "upload", tag, "--repo", repo, "--clobber",
                 *map(str, files)], env)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".md",
                                         delete=False) as f:
            f.write(notes)
            notes_path = f.name
        try:
            r = _gh(["release", "create", tag, "--repo", repo,
                     "--title", f"web net {tag}",
                     "--notes-file", notes_path, *map(str, files)], env)
        finally:
            os.unlink(notes_path)
    if r.returncode != 0:
        raise SystemExit(f"gh release failed: {r.stderr.strip()}")
    print(f"GitHub release {tag} on {repo}: "
          f"https://github.com/{repo}/releases/tag/{tag}")
    return True


def _publish_hf(tag: str, files: list[Path], manifest: Path, repo: str,
                dry_run: bool) -> None:
    token = _env("HF_TOKEN")
    if not token:
        raise SystemExit("no HF_TOKEN (env or .env) for --hf-repo")
    if dry_run:
        print(f"[dry-run] would upload {len(files)} assets to "
              f"huggingface.co/{repo}")
        return
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit(f"huggingface_hub not installed: {e}")
    api = HfApi(token=token)
    for p in files:
        api.upload_file(path_or_fileobj=str(p),
                        path_in_repo=f"web/{tag}/{p.name}",
                        repo_id=repo, repo_type="model",
                        commit_message=f"web net {tag}: {p.name}")
    # manifest last: clients see a complete set of files
    api.upload_file(path_or_fileobj=str(manifest),
                    path_in_repo="web/latest.json",
                    repo_id=repo, repo_type="model",
                    commit_message=f"web net {tag}: latest.json")
    print(f"HuggingFace fallback updated: "
          f"https://huggingface.co/{repo}/blob/main/web/latest.json")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a checkpoint to per-size ONNX and publish it "
                    "as the web GUI's latest net.")
    ap.add_argument("ckpt", type=Path,
                    help="trainer checkpoint (.pt)")
    ap.add_argument("--sizes", default=DEFAULT_SIZES,
                    help=f"board sizes (default {DEFAULT_SIZES})")
    ap.add_argument("--tag", default=None,
                    help="release/tag and directory name (default net-<ckpt>)")
    ap.add_argument("--out-dir", default="web/net",
                    help="manifest + <tag>/ output dir (default web/net)")
    ap.add_argument("--checkpoint-name", default=None,
                    help="public checkpoint name (default <ckpt>-train.pt)")
    ap.add_argument("--checkpoint-url", default=None,
                    help="public URL the manifest points at")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--gate", type=int, default=24,
                    help="parity positions per size (0 = skip)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing net-{b}.onnx in <out-dir>/<tag>/")
    ap.add_argument("--github-release", action="store_true",
                    help="publish the files as a GitHub release")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""),
                    help="GitHub repo for --github-release")
    ap.add_argument("--hf-repo", default=None,
                    help="also upload to this HF model repo (needs HF_TOKEN)")
    ap.add_argument("--force", action="store_true",
                    help="replace assets on an existing GitHub release")
    ap.add_argument("--dry-run", action="store_true",
                    help="do not create/upload anything")
    ap.add_argument("--no-upload", action="store_true",
                    help="local manifest/export only")
    args = ap.parse_args()

    ckpt = args.ckpt.expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint at {ckpt}")
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    stem = ckpt.parent.name if ckpt.stem in ("train", "best") else ckpt.stem
    tag = _slug(args.tag or f"net-{stem}")
    name = args.checkpoint_name or (
        f"{stem}-train.pt" if re.fullmatch(r"it\d+", stem) else ckpt.name)
    url = args.checkpoint_url
    if url is None and re.fullmatch(r"it\d+", stem):
        url = f"{DEFAULT_CHECKPOINTS_URL}/{name}"
    iteration = None
    m = re.search(r"it(\d+)", name, re.IGNORECASE)
    if m:
        iteration = int(m.group(1))

    out_dir = (REPO / args.out_dir).resolve()
    net_dir = out_dir / tag
    net_dir.mkdir(parents=True, exist_ok=True)

    trainer = _load_model(ckpt)
    model = trainer.model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {ckpt.name}  ({n_params:,} params, "
          f"{trainer.steps} steps)")
    print(f"tag: {tag}   checkpoint name: {name}")

    files: list[Path] = []
    assets: dict[str, dict] = {}
    parity: dict[str, dict] = {}
    for b in sizes:
        out = net_dir / f"net-{b}.onnx"
        if args.reuse:
            if not out.is_file():
                raise SystemExit(f"--reuse: missing {out}")
        else:
            export_onnx(model, b, out, args.opset)
        size_mb = out.stat().st_size / 1e6
        line = f"{out.name}: {size_mb:.1f} MB"
        if args.gate:
            res = parity_gate(model, b, out, n_pos=args.gate, seed=b)
            parity[str(b)] = res
            line += (f"  parity ok (d_logp {res['max_d_logp_legal']:.1e},"
                     f" argmax {res['argmax_agree']}/{res['positions']})")
        assets[str(b)] = {"bytes": out.stat().st_size,
                          "sha256": _sha256(out)}
        files.append(out)
        print(line, flush=True)

    manifest = {
        "schema": 1,
        "release": tag,
        "checkpoint": name,
        "checkpoint_url": url,
        "iteration": iteration,
        "steps": trainer.steps,
        "params": n_params,
        "published": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat().replace("+00:00", "Z"),
        "dir": tag,
        "files": {str(b): f"net-{b}.onnx" for b in sizes},
        "assets": assets,
        "parity": parity,
    }
    manifest_path = net_dir / "latest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # also at the manifest root for a local `web/serve.py` run
    (out_dir / "latest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")
    print(f"          {(out_dir / 'latest.json')}")

    # The release notes are public: no local paths, just provenance.
    notes = [
        f"Web nets for `{name}`" + (f" (iteration {iteration})" if iteration
                                    else "") + ".",
        "",
        f"- checkpoint: {url or ckpt.name}",
        f"- {n_params:,} parameters, {trainer.steps} training steps",
        "- sizes: " + ", ".join(f"{b}x{b}" for b in sizes),
    ]
    if parity:
        notes.append("- parity gate: torch vs onnxruntime-CPU, real "
                     "positions, argmax agreement per size")
    notes_path = net_dir / "RELEASE_NOTES.md"
    notes_path.write_text("\n".join(notes) + "\n")

    if not args.no_upload:
        if args.github_release:
            _publish_github(tag, files + [manifest_path], "\n".join(notes),
                            args.repo, args.force, args.dry_run)
        if args.hf_repo:
            _publish_hf(tag, files, manifest_path, args.hf_repo,
                        args.dry_run)
        if not args.github_release and not args.hf_repo:
            print("no upload target requested (--github-release / "
                  "--hf-repo); wrote local files only")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
