// ort_session.js — per-board-size ONNX session cache for the engine
// worker. Tries WebGPU first (the jsep build, with graph capture),
// falls back to the WASM backend; without cross-origin isolation ORT
// runs the wasm build single-threaded, which is fine at our batches.
import * as ort from "../vendor/ort.all.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url);

const cache = new Map();          // b → InferenceSession
export let activeEP = "wasm";     // honestly probed, reported by the worker

// Net source: a URL template with {b} (one ONNX per board size). The
// default is the trained net on HF — /resolve/ endpoints serve the raw
// file (/blob/ is the HTML page). The repo is private, so the fetch
// carries the HF token (passed from the page via ?hf= or localStorage).
export const DEFAULT_NET =
    "https://huggingface.co/chriszhang01/gonet/resolve/main/web/net-{b}.onnx";
export let netTemplate = DEFAULT_NET;   // null = bundled local assets
export let netOrigin = "hf:it023";      // shown in the perf line
export let hfToken = null;

export function configure(opts) {
    // opts.net comes from the page: absent = HF default, "local" =
    // bundled assets, otherwise a custom {b} template URL
    if (opts.net === "local") {
        netTemplate = null;
        netOrigin = "local";
    } else if (opts.net) {
        netTemplate = opts.net;
        netOrigin = "custom";
    }
    hfToken = opts.hfToken ?? hfToken;
}

export const tensor = (type, data, dims) => new ort.Tensor(type, data, dims);

async function netBytes(b) {
    const local = new URL(`../assets/net/net-${b}.onnx`, import.meta.url).href;
    const url = netTemplate ? netTemplate.replaceAll("{b}", b) : local;
    if (url.startsWith("http") && url.includes("huggingface.co")) {
        const r = await fetch(url, hfToken
            ? { headers: { Authorization: `Bearer ${hfToken}` } } : {});
        if (!r.ok)
            throw new Error(`net fetch ${r.status} ${url} — ` +
                (hfToken ? "token rejected?" :
                 "private repo: add ?hf=<token> to the page URL"));
        return { bytes: new Uint8Array(await r.arrayBuffer()),
                 via: netOrigin };
    }
    return { bytes: null, via: "local" };   // ort fetches the local file
}

export async function session(b) {
    if (cache.has(b))
        return cache.get(b);
    const { bytes, via } = await netBytes(b);
    const make = async (ep) => ort.InferenceSession.create(bytes ?? (
        netTemplate ? netTemplate.replaceAll("{b}", b) :
        new URL(`../assets/net/net-${b}.onnx`, import.meta.url).href),
        typeof ep === "string" ? { executionProviders: [ep] } : {
            executionProviders: [ep] });
    let s;
    try {
        // graph capture: record the kernel dispatch list once and replay
        // it — the model is small and dispatch-overhead-bound, so this
        // is the big lever on WebGPU
        s = await make({ name: "webgpu", enableGraphCapture: true });
        activeEP = `webgpu+capture`;
    } catch (e1) {
        try {
            s = await make("webgpu");
            activeEP = "webgpu";
            console.warn(`graph capture failed (${e1?.message ?? e1})`);
        } catch (e) {
            // both webgpu paths threw: the EP genuinely cannot run the
            // model — fall all the way back to the wasm CPU backend
            console.warn(`webgpu failed (${e?.message ?? e}); wasm only`);
            s = await make("wasm");
            activeEP = "wasm";
        }
    }
    netOrigin = via;
    cache.set(b, s);
    return s;
}
