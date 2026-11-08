// ort_session.js — per-board-size ONNX session cache for the engine
// worker. Tries WebGPU first (the jsep build), falls back to the WASM
// backend; without cross-origin isolation ORT runs the wasm build
// single-threaded, which is fine at our batch sizes.
import * as ort from "../vendor/ort.all.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url);

const cache = new Map();          // b → InferenceSession
export let activeEP = "wasm";     // honestly probed, reported by the worker
const FORCE_WASM = false;         // dev toggle: benchmark the CPU backend

export const tensor = (type, data, dims) => new ort.Tensor(type, data, dims);

export async function session(b) {
    if (cache.has(b))
        return cache.get(b);
    // dev escape hatch: localStorage.forceWasm = "1" benchmarks the CPU
    // backend against webgpu without touching the code
    if (FORCE_WASM) {
        const s = await ort.InferenceSession.create(
            new URL(`../assets/net/net-${b}.onnx`, import.meta.url).href,
            { executionProviders: ["wasm"] });
        activeEP = "wasm";
        cache.set(b, s);
        return s;
    }
    // ORT wants a string path (or buffer), not a URL object
    const url = new URL(`../assets/net/net-${b}.onnx`, import.meta.url).href;
    let s;
    try {
        // graph capture: record the kernel dispatch list once and replay
        // it — the model is small and dispatch-overhead-bound, so this
        // is the big lever on WebGPU
        s = await ort.InferenceSession.create(url, {
            executionProviders: [{
                name: "webgpu", enableGraphCapture: true,
            }],
        });
        activeEP = "webgpu+capture";
    } catch (e1) {
        try {
            s = await ort.InferenceSession.create(url, {
                executionProviders: ["webgpu"],
            });
            activeEP = "webgpu";
            console.warn(`graph capture failed (${e1?.message ?? e1})`);
        } catch (e) {
            // both webgpu paths threw: the EP genuinely cannot run the
            // model — fall all the way back to the wasm CPU backend
            console.warn(`webgpu failed (${e?.message ?? e}); wasm only`);
            s = await ort.InferenceSession.create(url, {
                executionProviders: ["wasm"],
            });
            activeEP = "wasm";
        }
    }
    cache.set(b, s);
    return s;
}
