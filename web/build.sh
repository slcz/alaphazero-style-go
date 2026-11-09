#!/bin/sh
# web/build.sh — build everything the GUI serves: the two wasm modules
# (engine + rules) and the per-size ONNX nets. Run from anywhere.
#
#   deps: emsdk (~/dev/emsdk), python with torch+onnx+onnxruntime
#         (the chessgui venv works: ~/chessgui/.venv)
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)

. "$HOME/dev/emsdk/emsdk_env.sh" >/dev/null 2>&1 || true
command -v emcc >/dev/null || { echo "emcc not found (emsdk at ~/dev/emsdk?)"; exit 1; }

make -C "$ROOT/engine" wasm wasm-rules

PY=${PY:-$HOME/chessgui/.venv/bin/python}
# bundled fallback nets (?net=local); the site itself loads the latest
# published release via web/net/latest.json, which publish_web_net.py
# generates from a real checkpoint
"$PY" "$ROOT/scripts/export_onnx.py" --random --sizes 7,9,13,19

echo "built:"
ls -la "$ROOT/engine/build/wasm/" "$ROOT/web/assets/net/"
echo "serve locally: python3 web/serve.py  → http://127.0.0.1:8642/web/"
