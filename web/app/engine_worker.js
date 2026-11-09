// engine_worker.js — the wasm GTP engine + its JS net evaluator.
//
// The whole GTP protocol lives in the wasm module (gtp.c): this worker
// feeds it lines with gtp_exec and relays the printf'd responses (and
// analyze's streaming frames) to the referee. Under Asyncify the wasm
// stack suspends inside the search while we run onnxruntime-web
// (Module.netRun / Module.rawRun — the C seam in wasm_glue.c).
//
// CLASSIC worker (ort-web's module-worker support is unreliable): both
// the engine module and the ort ESM bundle come in via dynamic import()
// — which classic workers do support — the chessgui pattern, proven
// there. importScripts would parse-error on the ESM build.

const NN_TOKENS = 361, NN_DIM = 18, POLICY_MAX = 362;

let ortMod = null;               // { session, tensor } once imported

let M = null;                     // the engine module
let busy = false;                 // a gtp_exec is in flight
const queue = [];                 // commands that arrived while busy
let reqId = 0;
let gtpPort = self;               // the referee's channel end once bound

// per-genmove eval telemetry: rows evaluated, wall time, EP in use
const stats = { evals: 0, rows: 0, ms: 0 };

function post(type, obj) {
    gtpPort.postMessage({ type, ...obj });
}

// ---- the net evaluator (wasm_glue.c calls these) ----------------------

// A failed forward must never freeze the search (the C stack is
// suspended waiting for it): degrade to the dummy evaluator's outputs.
function dummyOutputs(n, logpP, wdlP, scoreP) {
    const logp = new Float32Array(M.HEAPF32.buffer, logpP, n * POLICY_MAX);
    const wdl = new Float32Array(M.HEAPF32.buffer, wdlP, n * 2);
    const score = new Float32Array(M.HEAPF32.buffer, scoreP, n);
    for (let i = 0; i < n; i++) {
        logp.fill(0, i * POLICY_MAX, (i + 1) * POLICY_MAX);
        wdl[2 * i] = 0.5;
        wdl[2 * i + 1] = 0.5;
        score[i] = 0;
    }
}

async function netRun(n, planesP, maskP, logpP, wdlP, scoreP) {
    const t0 = performance.now();
    try {
        const planes = new Float32Array(M.HEAPF32.buffer, planesP,
                                         n * NN_TOKENS * NN_DIM);
        const b = Math.round(planes[NN_DIM * 0 + 17] * 19); // GNN_SIZE dim
        const T = b * b;
        const tokens = new Float32Array(n * T * NN_DIM);
        for (let i = 0; i < n; i++)
            tokens.set(planes.subarray(i * NN_TOKENS * NN_DIM,
                                       i * NN_TOKENS * NN_DIM + T * NN_DIM),
                       i * T * NN_DIM);
        const mask = new Uint8Array(M.HEAPU8.buffer, maskP, n * POLICY_MAX);

        const s = await ortMod.session(b);
        const out = await s.run({
            tokens: ortMod.tensor("float32", tokens, [n, T, NN_DIM]),
            mask: ortMod.tensor("bool", mask, [n, POLICY_MAX]),
        });
        const logp = out.policy_logp.data;      // [n, b*b+1]
        const wdl = out.wdl.data;               // [n, 2] probs
        const score = out.score.data;           // [n]

        const logpOut = new Float32Array(M.HEAPF32.buffer, logpP,
                                         n * POLICY_MAX);
        const wdlOut = new Float32Array(M.HEAPF32.buffer, wdlP, n * 2);
        const scoreOut = new Float32Array(M.HEAPF32.buffer, scoreP, n);
        for (let i = 0; i < n; i++) {
            logpOut.fill(0, i * POLICY_MAX, (i + 1) * POLICY_MAX);
            logpOut.set(logp.subarray(i * (T + 1), (i + 1) * (T + 1)),
                        i * POLICY_MAX);
            wdlOut[2 * i] = wdl[2 * i];
            wdlOut[2 * i + 1] = wdl[2 * i + 1];
            scoreOut[i] = score[i];
        }
    } catch (e) {
        console.warn("netRun failed — dummy outputs:", e?.message ?? e);
        dummyOutputs(n, logpP, wdlP, scoreP);
    } finally {
        stats.evals++;
        stats.rows += n;
        stats.ms += performance.now() - t0;
    }
}

// One raw eval (the `value` / `analyze` seam). A failure must surface as
// a nonzero return — puct_nn_raw maps it to "no evaluator" and the GTP
// command completes — never an unhandled rejection: under Asyncify that
// would leave gtp_exec suspended forever and wedge the whole worker
// (every later `play` would queue behind it and stones stop placing).
async function rawRun(b, planesP, maskP, polP, wdlP, ownP, scoreP) {
    try {
        return await rawRunInner(b, planesP, maskP, polP, wdlP, ownP,
                                 scoreP);
    } catch (e) {
        console.warn("rawRun failed — no evaluator:", e?.message ?? e);
        return 1;
    }
}

async function rawRunInner(b, planesP, maskP, polP, wdlP, ownP, scoreP) {
    const planes = new Float32Array(M.HEAPF32.buffer, planesP,
                                     NN_TOKENS * NN_DIM);
    const T = b * b;
    const tokens = planes.slice(0, T * NN_DIM);   // copy out of the heap
    const mask = new Uint8Array(M.HEAPU8.buffer, maskP, POLICY_MAX);

    const s = await ortMod.session(b);
    const out = await s.run({
        tokens: ortMod.tensor("float32", tokens, [1, T, NN_DIM]),
        mask: ortMod.tensor("bool", mask, [1, POLICY_MAX]),
    });
    const polOut = new Float32Array(M.HEAPF32.buffer, polP, T + 1);
    for (let i = 0; i <= T; i++)
        polOut[i] = Math.exp(out.policy_logp.data[i]);
    const wdlOut = new Float32Array(M.HEAPF32.buffer, wdlP, 2);
    wdlOut[0] = out.wdl.data[0];
    wdlOut[1] = out.wdl.data[1];
    const ownOut = new Float32Array(M.HEAPF32.buffer, ownP, T);
    ownOut.set(out.own.data);
    const scoreOut = new Float32Array(M.HEAPF32.buffer, scoreP, 1);
    scoreOut[0] = out.score.data[0];
    return 0;
}

// ---- GTP plumbing ------------------------------------------------------
let curId = null;                // the command whose prints these are

async function run(line, id) {
    busy = true;
    curId = id;
    stats.evals = stats.rows = stats.ms = 0;
    const t0 = performance.now();
    const quit = await M.ccall("gtp_exec", "number", ["string"], [line],
                               { async: true });
    busy = false;
    curId = null;
    if (stats.evals)
        post("stats", { id, ep: ortMod?.activeEP ?? "?", evals: stats.evals,
                        rows: stats.rows, netMs: Math.round(stats.ms),
                        totalMs: Math.round(performance.now() - t0) });
    post("resp", { id, quit: !!quit });
    if (queue.length) {
        const next = queue.shift();
        run(next.line, next.id);
    }
}

async function onCmd(m) {
    if (m.type === "bind") {                // the referee's direct line
        gtpPort = m.port;
        gtpPort.onmessage = (e) => onCmd(e.data);
        return;
    }
    if (m.type === "init") {
        const { default: createGoEngine } = await import(
            new URL("../../engine/build/wasm/go_engine.js",
                    self.location.href).href);
        M = await createGoEngine({
            print: (l) => post("line", { id: curId, line: l }),
            printErr: (l) => console.warn("[gtp]", l),
            locateFile: (p) =>
                new URL(`../../engine/build/wasm/${p}`,
                        self.location.href).href,
        });
        M.netRun = netRun;
        M.rawRun = rawRun;
        if (M.ccall("gtp_wasm_init", "number",
                    ["number", "number", "number", "number", "number",
                     "number", "number"],
                    [m.b, m.komi, m.maxply, m.sims, m.walkers, m.rows ?? 32,
                     BigInt(m.seed ?? 1)]) !== 0) {
            post("fatal", { error: "gtp_wasm_init failed" });
            return;
        }
        M.ccall("gtp_wasm_attach_net", null, [], []);
        post("ready", {});
        // dynamic import: ort's ESM inside a classic worker (chessgui
        // pattern); warm the opening size so the first genmove is fast
        try {
            ortMod = await import(
                new URL("./ort_session.js", self.location.href).href);
            ortMod.configure({
                net: m.net ?? null,
                hfToken: m.hfToken ?? null,
            });
            await ortMod.session(m.b);
            post("ep", { webgpu: "on", net: ortMod.netOrigin,
                         checkpoint: ortMod.netCheckpoint });
        } catch (e) {
            console.warn("ort warm-up failed (dummy eval until it works):",
                         e?.message ?? e);
            post("ep", { webgpu: "unknown", net: "FAILED",
                         error: e?.message ?? String(e) });
        }
        return;
    }
    if (m.type === "setSims") {
        M.ccall("gtp_wasm_set_sims", null, ["number"], [m.sims]);
        return;
    }

    if (m.type === "gtp") {
        if (busy) {
            queue.push({ line: m.line, id: m.id ?? ++reqId });
            M.ccall("gtp_interrupt", null, [], []);  // analyze closes early
            return;
        }
        run(m.line, m.id ?? ++reqId);
    }
}

self.onmessage = (ev) => onCmd(ev.data);
