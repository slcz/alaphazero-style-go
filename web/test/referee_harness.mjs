// referee_harness.mjs — drive referee_worker.js in node with a mock
// engine port to reproduce the "stuck at thinking" hang with a stack.
// The worker script is imported as a module; self/message plumbing is
// shimmed to plain function calls.
import { readFileSync } from "node:fs";

// ---- shims -------------------------------------------------------------
const messages = [];               // referee → UI posts
const awaiting = [];
const self = {
    postMessage: (m) => {
        messages.push(m);
        if (m.type === "status")
            console.log(`[status] ${m.text}`);
    },
    location: {
        href: new URL("../app/referee_worker.js", import.meta.url).href,
    },
};
globalThis.self = self;

// dynamic import of the worker as a module (it uses self.* and
// importScripts-free dynamic imports for the rules wasm)
const src = readFileSync(new URL("../app/referee_worker.js", import.meta.url),
                         "utf8");
const mod = await import(
    "data:text/javascript," + encodeURIComponent(src));
// the worker file has no exports; its side effects registered onmessage
// via `self.onmessage = ...`? It assigns `self.onmessage` — grab it:
const onmessage = self.onmessage;
if (!onmessage)
    throw new Error("no onmessage registered");

const post = (m) => onmessage({ data: m });

// mock engine port: the referee talks GTP; we answer like the wasm one.
// Supports both onmessage-assignment and addEventListener styles.
const engineHandlers = {};
let onmsg = null;
const deliver = (data) => {
    for (const h of engineHandlers.message || [])
        h({ data });
    if (onmsg)
        onmsg({ data });
};
const mockPort = {
    set onmessage(h) { onmsg = h; },
    get onmessage() { return onmsg; },
    postMessage: async (m) => {
        if (m.type !== "gtp")
            return;
        const reply = mockReply(m.line);
        setTimeout(() => {
            deliver({ type: "line", id: m.id, line: reply });
            deliver({ type: "resp", id: m.id, quit: false });
        }, 10);
    },
    addEventListener: (ev, h) => {
        (engineHandlers[ev] = engineHandlers[ev] || []).push(h);
    },
};

function mockReply(line) {
    if (line.startsWith("genmove"))
        return "= H4";
    return "= ";                     // play/boardsize/komi/clear: ok
}

process.on("unhandledRejection", (e) => {
    console.error("UNHANDLED REJECTION:", e);
    process.exit(1);
});

// ---- drive one full exchange -------------------------------------------
post({ type: "boot", port: mockPort });
await new Promise(r => setTimeout(r, 500));   // rules module loads

await post({ type: "newGame", b: 9, komi: 7.5, humanColor: 1 });
await new Promise(r => setTimeout(r, 200));
await post({ type: "humanMove", move: 40 });  // E5
await new Promise(r => setTimeout(r, 300));

console.log("---- UI messages:");
for (const m of messages)
    console.log(JSON.stringify(m).slice(0, 120));
