// ort_session.js — per-board-size ONNX session cache for the engine
// worker. Tries WebGPU first (the jsep build), falls back to the WASM
// backend; without cross-origin isolation ORT runs the wasm build
// single-threaded, which is fine at our batch sizes.
import * as ort from "../vendor/ort.all.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url);

const cache = new Map();          // b → InferenceSession

export const tensor = (type, data, dims) => new ort.Tensor(type, data, dims);

export async function session(b) {
    if (cache.has(b))
        return cache.get(b);
    // ORT wants a string path (or buffer), not a URL object
    const url = new URL(`../assets/net/net-${b}.onnx`, import.meta.url).href;
    let s;
    try {
        s = await ort.InferenceSession.create(url, {
            executionProviders: ["webgpu", "wasm"],
        });
    } catch (e) {
        console.warn(`webgpu session failed (${e?.message ?? e}); wasm only`);
        s = await ort.InferenceSession.create(url, {
            executionProviders: ["wasm"],
        });
    }
    cache.set(b, s);
    return s;
}
