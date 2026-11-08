# web GUI — the Go engine in a browser, serverless

The whole engine (rules + RPO MCTS + GTP protocol) compiles to plain
WASM 1.0 — no pthreads, no SharedArrayBuffer, no server: any static
host serves it. The trained net runs through onnxruntime-web (WebGPU
when available, WASM backend otherwise).

**Architecture** (three threads, per the design brief):

```
main thread          referee worker             engine worker
-----------          --------------             ------------
board canvas    ⇄    match controller      ⇄    wasm GTP engine
controls             own go_rules wasm          (gtp.c, unmodified
                     legality/captures/score     protocol, via gtp_exec)
                     speaks GTP over a          net = ort-web session,
                     MessageChannel             Asyncify-suspended
                                                inside the search
```

- **GTP is kept intact**: the engine worker runs the same `gtp.c`
  dispatcher, one command per `gtp_exec` call; responses are captured
  from printf and framed back over the port. `analyze` streams frames
  and closes on `gtp_interrupt` (any queued command).
- **The referee is a separate thread with its own rules engine**
  (`go_rules` wasm, the same FFI surface the Python cffi binding
  uses): it replays every move, adjudicates legality, captures, double
  pass / ply-wall, and the final TT score — the player cannot cheat
  itself.
- **The net**: `wasm_glue.c` is the evaluator seam — the search's PUSH
  suspends the wasm stack (Asyncify) while the worker runs the ONNX
  session on the staged rows. One net file PER BOARD SIZE (the
  transformer derives b = isqrt(T) inside forward; tracing bakes it
  in). A failed forward degrades to the dummy evaluator — the engine
  never freezes.

## Run it

```sh
web/build.sh                     # wasm + nets (~1 min)
python3 web/serve.py             # http://127.0.0.1:8642/web/
```

Or any static server rooted at the repo (`Cache-Control: no-store`
helps during development).

## Files

- `engine/src/wasm_glue.c` — the JS evaluator seam + `puct_nn_raw`
- `engine/Makefile` targets `wasm` / `wasm-rules`
- `web/app/engine_worker.js` — wasm engine + ort session + GTP framing
- `web/app/referee_worker.js` — the match controller (its own thread)
- `web/app/ort_session.js` — per-size session cache, webgpu→wasm
- `web/app/board.js`, `web/app/main.js` — canvas board, controls
- `scripts/export_onnx.py` — checkpoint → per-size ONNX + parity gate
- `web/test/gtp_smoke.mjs` — node: GTP protocol over the dummy engine
- `web/test/referee_harness.mjs` — node: referee flow, mock engine

## Notes

- **Real checkpoints**: `scripts/export_onnx.py --ckpt best.pt
  --sizes 7,9` (fp8 payloads dequantize on export). The checked-in
  nets are random-init — legal moves, garbage play.
- **Board size support** costs one exported file per size (18 MB fp32
  each, ~7 MB if int8-quantized); the engine, rules, referee and UI
  are size-generic.
- Perf: WebGPU ≈ sub-second genmove at a few hundred sims; the WASM
  backend is ~5-10× slower — raise sims with care there.
