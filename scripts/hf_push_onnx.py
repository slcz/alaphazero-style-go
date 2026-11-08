"""hf_push_onnx.py — export a checkpoint to per-size ONNX and publish it
to the HF repo the web GUI loads from (gui branch).

The browser cannot run .pt checkpoints: this is the one command that
takes a trainer checkpoint to a GUI-ready net at
`https://huggingface.co/<repo>/resolve/main/web/net-{b}.onnx`.

    HF_TOKEN=... python scripts/hf_push_onnx.py ckpts/best.pt 7,9,13,19

Token: HF_TOKEN env or the repo's .env (HF_TOKEN=...). The repo is
private, so GUI users pass the token once via ?hf=<token> (stored in
localStorage).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.export_onnx import export_onnx, parity_gate  # noqa: E402


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


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ckpt = Path(sys.argv[1]).expanduser()
    sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2
                              else "7,9,13,19").split(",")]

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
    model = Trainer.load(ckpt, device="cpu").model

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    repo_id = os.environ.get("HF_MODEL_REPO", "chriszhang01/gonet")

    with tempfile.TemporaryDirectory() as tmp:
        for b in sizes:
            out = Path(tmp) / f"net-{b}.onnx"
            export_onnx(model, b, out)
            res = parity_gate(model, b, out)
            api.upload_file(path_or_fileobj=str(out),
                            path_in_repo=f"web/net-{b}.onnx",
                            repo_id=repo_id, repo_type="model",
                            commit_message=f"web GUI net {b}x{b} "
                                           f"({ckpt.name} export)")
            print(f"web/net-{b}.onnx pushed "
                  f"(d_logp {res['max_d_logp_legal']:.1e}, "
                  f"argmax {res['argmax_agree']}/{res['positions']})")
    print(f"GUI default now serves this checkpoint from {repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
