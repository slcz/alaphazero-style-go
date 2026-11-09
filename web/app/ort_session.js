// ort_session.js — per-board-size ONNX session cache for the engine
// worker. Tries WebGPU first (the jsep build, with graph capture),
// falls back to the WASM backend; without cross-origin isolation ORT
// runs the wasm build single-threaded, which is fine at our batches.
import * as ort from "../vendor/ort.all.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url);

const cache = new Map();          // b → InferenceSession
const inflight = new Map();       // b → Promise<InferenceSession>
const failed = new Set();         // b → net permanently failed (until reload)
const STALL_TIMEOUT_MS = 20000;   // abort only when NO bytes arrive
const MANIFEST_TIMEOUT_MS = 10000;
export let activeEP = "wasm";     // honestly probed, reported by the worker

// The net lives on HuggingFace, never in this repo. The stable
// `web/net-{b}.onnx` files are overwritten on every publish, so the GUI
// always loads the newest export; `web/latest.json` is an optional
// pointer in the model repo that names the checkpoint (and can carry
// per-size file names). The model repo is public; a private fork can
// pass ?hf=<token> once.
export const DEFAULT_NET =
    "https://huggingface.co/chriszhang01/gonet/resolve/main/web/net-{b}.onnx";
export const MANIFEST_URL =
    "https://huggingface.co/chriszhang01/gonet/resolve/main/web/latest.json";

export let netOrigin = "hf:latest";    // shown in the status/perf lines
export let netCheckpoint = null;       // e.g. "it026-train.pt"
export let hfToken = null;

let mode = "latest";             // "latest" | "local" | "custom"
let customTemplate = null;
let sourcePromise = null;        // resolved {kind, urlFor, origin, checkpoint}

export function configure(opts) {
    // opts.net comes from the page: absent = the HF default, "local" =
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
    // Cache-bust: latest.json changes on every publish, while the
    // ONNX files it names are fetched separately.
    const url = new URL(MANIFEST_URL);
    url.searchParams.set("v", Date.now().toString(36));
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), MANIFEST_TIMEOUT_MS);
    let r;
    try {
        r = await fetch(url, {
            cache: "no-store",
            signal: controller.signal,
            headers: hfToken ? { Authorization: `Bearer ${hfToken}` } : {},
        });
    } catch (e) {
        if (e.name === "AbortError")
            throw new Error("net manifest timed out");
        throw e;
    } finally {
        clearTimeout(timer);
    }
    if (!r.ok) {
        const e = new Error(`net manifest HTTP ${r.status}`);
        e.status = r.status;
        throw e;
    }
    const m = await r.json();
    if (!m || !m.files)
        throw new Error("net manifest missing files");
    return m;
}

async function latestSource() {
    try {
        const m = await fetchManifest();
        return {
            kind: "manifest",
            origin: m.release ?? "hf:latest",
            checkpoint: m.checkpoint ?? null,
            urlFor: (b) => {
                const f = m.files[String(b)];
                if (!f)
                    throw new Error(`no ${b}x${b} net in latest.json`);
                // `dir` is optional: with it, files live in a versioned
                // subdirectory; without it, beside latest.json.
                return new URL(m.dir ? `${m.dir}/${f}` : f,
                               MANIFEST_URL).href;
            },
        };
    } catch (e) {
        // 404 is the normal case until the optional pointer is published
        if (e.status !== 404)
            console.warn(`HF latest.json unavailable (${e.message}); ` +
                         `using the stable web/net-{b}.onnx files`);
        return {
            kind: "hf",
            origin: "hf:latest",
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

// Fetch a large net with a *stall* watchdog rather than a total timeout:
// a slow-but-progressing 27 MB download may take minutes, while a hung
// socket should fail after 20 s and let the engine use the dummy eval.
async function fetchNetBytes(url) {
    const controller = new AbortController();
    let timer = null;
    const arm = () => {
        clearTimeout(timer);
        timer = setTimeout(() => controller.abort(), STALL_TIMEOUT_MS);
    };
    arm();
    try {
        const r = await fetch(url, {
            signal: controller.signal,
            headers: hfToken && url.includes("huggingface.co")
                ? { Authorization: `Bearer ${hfToken}` } : {},
        });
        if (!r.ok) {
            const hint = url.includes("huggingface.co")
                ? (hfToken ? "token rejected?" :
                   "private repo: add ?hf=<token> to the page URL")
                : "check the net URL";
            throw new Error(`net fetch ${r.status} ${url} — ${hint}`);
        }
        if (!r.body)
            return new Uint8Array(await r.arrayBuffer());
        const reader = r.body.getReader();
        const chunks = [];
        let total = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done)
                break;
            arm();                 // bytes are flowing: reset the watchdog
            chunks.push(value);
            total += value.length;
        }
        const bytes = new Uint8Array(total);
        let off = 0;
        for (const c of chunks) {
            bytes.set(c, off);
            off += c.length;
        }
        return bytes;
    } catch (e) {
        if (e.name === "AbortError")
            throw new Error(`net stalled for ${STALL_TIMEOUT_MS / 1000}s`);
        throw e;
    } finally {
        clearTimeout(timer);
    }
}

async function netBytes(b) {
    const src = await resolveSource();
    const url = src.urlFor(b);
    netOrigin = src.origin;
    netCheckpoint = src.checkpoint;

    // Local assets are left to ORT (it streams them); remote sources
    // are fetched here so the private HF repo can carry the token and
    // a missing size/manifest can fail with a useful message.
    const remote = /^https?:/i.test(url);
    const viaHF = url.includes("huggingface.co");
    const fetchHere = remote && (src.kind === "manifest" || src.kind === "hf"
                                 || viaHF);
    if (!fetchHere)
        return { bytes: null, url, via: src.origin };

    return { bytes: await fetchNetBytes(url), url, via: src.origin };
}

async function evictOthers(b) {
    // Keep one session: Safari's tab budget cannot hold an fp32
    // model (27 MB of weights + ORT overhead) for every board size.
    for (const [k, s] of [...cache]) {
        if (k === b)
            continue;
        cache.delete(k);
        try {
            await s.release?.();
        } catch (e) {
            console.warn(`net ${k} release failed:`, e?.message ?? e);
        }
    }
}

async function createSession(b) {
    await evictOthers(b);
    const { bytes, url, via } = await netBytes(b);
    const make = (ep) => ort.InferenceSession.create(
        bytes ?? url, { executionProviders: [ep] });
    // WebKit/Safari: the graph-capture path is the fragile one, and
    // the tab budget is smaller. Use plain WebGPU, then wasm.
    const ua = navigator.userAgent;
    const isIOS = /iPad|iPhone|iPod/.test(ua) ||
        (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    const isSafari = /Safari/.test(ua) &&
        !/Chrome|Chromium|CriOS|FxiOS|Edg|OPR/i.test(ua);
    const lowMem = isIOS || isSafari;

    let s = null;
    if (!lowMem) {
        try {
            // graph capture: record the kernel dispatch list once and
            // replay it — the model is small and dispatch-overhead-bound
            s = await make({ name: "webgpu", enableGraphCapture: true });
            activeEP = `webgpu+capture`;
        } catch (e1) {
            console.warn(`graph capture failed (${e1?.message ?? e1})`);
        }
    }
    if (!s) {
        try {
            s = await make("webgpu");
            activeEP = "webgpu";
        } catch (e) {
            // the EP genuinely cannot run the model — fall all the way
            // back to the wasm CPU backend
            console.warn(`webgpu failed (${e?.message ?? e}); wasm only`);
            s = await make("wasm");
            activeEP = "wasm";
        }
    }
    netOrigin = via;
    return s;
}

export async function session(b) {
    if (cache.has(b))
        return cache.get(b);
    // One fetch per size: the warm-up and the first genmove both ask for
    // session(9); without this they race two 27 MB downloads.
    if (inflight.has(b))
        return inflight.get(b);
    if (failed.has(b))
        throw new Error(`net ${b}x${b} unavailable; using dummy eval ` +
                        `(reload to retry)`);
    const p = createSession(b).then((s) => {
        cache.set(b, s);
        return s;
    }, (e) => {
        failed.add(b);
        throw e;
    }).finally(() => {
        inflight.delete(b);
    });
    inflight.set(b, p);
    return p;
}
