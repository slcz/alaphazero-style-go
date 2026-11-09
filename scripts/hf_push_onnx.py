"""hf_push_onnx.py — export a checkpoint to per-size ONNX and publish it
to the HuggingFace repo the web GUI loads from.

The browser cannot run .pt checkpoints: this is the one command that
takes a trainer checkpoint to a GUI-ready net under
`https://huggingface.co/<repo>/resolve/main/web/net-{b}.onnx`. It also
writes `web/latest.json`, the small pointer the GUI reads for the
checkpoint name; the stable `net-{b}.onnx` files are overwritten on
every publish, so the page always loads the newest export.

    HF_TOKEN=... python scripts/hf_push_onnx.py ckpts/it026.pt 7,9,13,19
    HF_TOKEN=... python scripts/hf_push_onnx.py ckpts/it026.pt \
        --checkpoint-url https://huggingface.co/chriszhang01/gonet/blob/main/checkpoints/it026-train.pt

Token: HF_TOKEN env or the repo's .env (HF_TOKEN=...). The repo is
private, so GUI users pass the token once via ?hf=<token> (stored in
localStorage), or the repo has to be made public.

No ONNX is stored in the GitHub repo: GitHub Pages only serves the GUI
and the wasm engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.export_onnx import export_onnx, parity_gate  # noqa: E402

DEFAULT_SIZES = "7,9,13,19"
DEFAULT_HF_REPO = "chriszhang01/gonet"


def hf_token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("no HF_TOKEN (env or .env)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="trainer checkpoint (.pt)")
    ap.add_argument("sizes", nargs="?", default=DEFAULT_SIZES,
                    help=f"board sizes (default {DEFAULT_SIZES})")
    ap.add_argument("--checkpoint-name", default=None,
                    help="public checkpoint name (default <stem>-train.pt)")
    ap.add_argument("--checkpoint-url", default=None,
                    help="public URL recorded in latest.json")
    ap.add_argument("--gate", type=int, default=48,
                    help="parity positions per size (0 = skip)")
    args = ap.parse_args()

    ckpt = args.ckpt.expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint at {ckpt}")
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    stem = ckpt.stem
    name = args.checkpoint_name or (
        f"{stem}-train.pt" if re.fullmatch(r"it\d+", stem) else ckpt.name)
    repo_id = os.environ.get("HF_MODEL_REPO", DEFAULT_HF_REPO)
    url = args.checkpoint_url or (
        f"https://huggingface.co/{repo_id}/blob/main/checkpoints/{name}")

    # the export unpickles optimizer state referencing torch.optim.Muon
    import torch
    if not hasattr(torch.optim, "Muon"):
        def _swallow(self, *a, **k):
            self.__dict__.update(k)
        torch.optim.Muon = type("Muon", (object,),
                                {"__init__": _swallow,
                                 "__setstate__": _swallow,
                                 "load_state_dict": _swallow})
    from go_ai.train.trainer import Trainer
    trainer = Trainer.load(ckpt, device="cpu")
    model = trainer.model

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())

    assets: dict[str, dict] = {}
    parity: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for b in sizes:
            out = Path(tmp) / f"net-{b}.onnx"
            export_onnx(model, b, out)
            res = parity_gate(model, b, out, n_pos=args.gate, seed=b)
            parity[str(b)] = res
            api.upload_file(
                path_or_fileobj=str(out),
                path_in_repo=f"web/net-{b}.onnx",
                repo_id=repo_id, repo_type="model",
                commit_message=f"web GUI net {b}x{b} "
                               f"({name} export)")
            assets[str(b)] = {
                "bytes": out.stat().st_size,
                "sha256": _sha256(out),
            }
            print(f"web/net-{b}.onnx pushed "
                  f"(d_logp {res['max_d_logp_legal']:.1e}, "
                  f"argmax {res['argmax_agree']}/{res['positions']})",
                  flush=True)

        # the manifest goes last: once it lands, every referenced file
        # is already in place. No `dir`: files sit beside latest.json.
        manifest = {
            "schema": 1,
            "checkpoint": name,
            "checkpoint_url": url,
            "published": datetime.now(timezone.utc).replace(
                microsecond=0).isoformat().replace("+00:00", "Z"),
            "files": {str(b): f"net-{b}.onnx" for b in sizes},
            "assets": assets,
            "parity": parity,
        }
        mpath = Path(tmp) / "latest.json"
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
        api.upload_file(
            path_or_fileobj=str(mpath),
            path_in_repo="web/latest.json",
            repo_id=repo_id, repo_type="model",
            commit_message=f"web GUI latest.json ({name})")

    print(f"GUI now loads {name} from "
          f"https://huggingface.co/{repo_id}/tree/main/web")
    print("private repo: add ?hf=<token> to the GUI URL "
          "(or make the model repo public)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
