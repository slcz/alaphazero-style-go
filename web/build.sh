#!/bin/sh
# web/build.sh — build the two wasm modules the GUI serves (engine +
# rules). The ONNX nets are loaded from HuggingFace at runtime, so they
# are not built here. Run from anywhere.
#
#   deps: emsdk (~/dev/emsdk)
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)

. "$HOME/dev/emsdk/emsdk_env.sh" >/dev/null 2>&1 || true
command -v emcc >/dev/null || { echo "emcc not found (emsdk at ~/dev/emsdk?)"; exit 1; }

make -C "$ROOT/engine" wasm wasm-rules

echo "built:"
ls -la "$ROOT/engine/build/wasm/"
echo "serve locally: python3 web/serve.py  → http://127.0.0.1:8642/web/"
