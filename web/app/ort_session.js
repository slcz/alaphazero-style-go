// ort_session.js — per-board-size ONNX session cache for the engine
// worker. Tries WebGPU first (the jsep build, with graph capture),
// falls back to the WASM backend; without cross-origin isolation ORT
// runs the wasm build single-threaded, which is fine at our batches.
import * as ort from "../vendor/ort.all.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url);

const cache = new Map();          // b → InferenceSession
export let activeEP = "wasm";     // honestly probed, reported by the worker

// Net source, in order of preference:
//   1. a same-origin net/latest.json manifest (the Pages deploy writes
//      it from the newest net-* GitHub release — no token needed),
//   2. the HuggingFace default template (private repo → ?hf=<token>),
//   3. the bundled random-init assets (?net=local).
// The manifest is just a pointer: it names the checkpoint and the
// per-size ONNX files, so publishing a new checkpoint needs no code
// change here. See scripts/publish_web_net.py.
export const DEFAULT_NET =
    "https://huggingface.co/chriszhang01/gonet/resolve/main/web/net-{b}.onnx";
export const MANIFEST_URL = new URL("../net/latest.json", import.meta.url);

export let netOrigin = "hf:default";   // shown in the status/perf lines
export let netCheckpoint = null;       // e.g. "it026-train.pt"
export let hfToken = null;

let mode = "latest";             // "latest" | "local" | "custom"
let customTemplate = null;
let sourcePromise = null;        // resolved {kind, urlFor, origin, checkpoint}

export function configure(opts) {
    // opts.net comes from the page: absent = latest manifest, "local" =
    // bundled assets, otherwise a custom {b} template URL
    if (opts.net === "local") {
        mode = "local";
    } else if (opts.net) {
        mode = "custom";
        customTemplate = opts.net;
    } else {
        mode = "latest";
    }
    hfToken = opts.hfToken ?? hfToken;
    sourcePromise = null;         // the engine inits once, but be safe
}

const localURL = (b) =>
    new URL(`../assets/net/net-${b}.onnx`, import.meta.url).href;

async function fetchManifest() {
    // Cache-bust: latest.json changes on every deploy, while the
    // versioned ONNX files beside it are immutable.
    const url = new URL(MANIFEST_URL);
    url.searchParams.set("v", Date.now().toString(36));
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok)
        throw new Error(`net manifest HTTP ${r.status}`);
    const m = await r.json();
    if (!m || typeof m.dir !== "string" || !m.files)
        throw new Error("net manifest missing dir/files");
    return m;
}

async function latestSource() {
    try {
        const m = await fetchManifest();
        return {
            kind: "manifest",
            origin: m.release ?? m.checkpoint ?? "latest",
            checkpoint: m.checkpoint ?? null,
            urlFor: (b) => {
                const f = m.files[String(b)];
                if (!f)
                    throw new Error(`no ${b}x${b} net in ` +
                                    `${m.release ?? "net manifest"}`);
                return new URL(`${m.dir}/${f}`, MANIFEST_URL).href;
            },
        };
    } catch (e) {
        console.warn(`latest net manifest unavailable (${e.message}); ` +
                     `falling back to the HuggingFace default`);
        return {
            kind: "hf",
            origin: "hf:default",
            checkpoint: null,
            urlFor: (b) => DEFAULT_NET.replaceAll("{b}", b),
        };
    }
}

function resolveSource() {
    if (!sourcePromise) {
        if (mode === "local") {
            sourcePromise = Promise.resolve({
                kind: "local", origin: "local", checkpoint: null,
                urlFor: localURL,
            });
        } else if (mode === "custom") {
            sourcePromise = Promise.resolve({
                kind: "custom", origin: "custom", checkpoint: null,
                urlFor: (b) => customTemplate.replaceAll("{b}", b),
            });
        } else {
            sourcePromise = latestSource();
        }
    }
    return sourcePromise;
}

export const tensor = (type, data, dims) => new ort.Tensor(type, data, dims);

async function netBytes(b) {
    const src = await resolveSource();
    const url = src.urlFor(b);
    netOrigin = src.origin;
    netCheckpoint = src.checkpoint;

    // Local assets are left to ORT (it streams them); remote sources
    // are fetched here so a private HF repo can carry the token and a
    // missing size/manifest can fail with a useful message.
    const remote = /^https?:/i.test(url);
    const viaHF = url.includes("huggingface.co");
    const fetchHere = remote && (src.kind === "manifest" || src.kind === "hf"
                                 || viaHF);
    if (!fetchHere)
        return { bytes: null, url, via: src.origin };

    const r = await fetch(url, viaHF && hfToken
        ? { headers: { Authorization: `Bearer ${hfToken}` } } : {});
    if (!r.ok) {
        const hint = viaHF
            ? (hfToken ? "token rejected?" :
               "private repo: add ?hf=<token> to the page URL")
            : `check ${MANIFEST_URL.pathname}`;
        throw new Error(`net fetch ${r.status} ${url} — ${hint}`);
    }
    return {
        bytes: new Uint8Array(await r.arrayBuffer()),
        url,
        via: src.origin,
    };
}

export async function session(b) {
    if (cache.has(b))
        return cache.get(b);
    const { bytes, url, via } = await netBytes(b);
    const make = (ep) => ort.InferenceSession.create(
        bytes ?? url, { executionProviders: [ep] });
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
