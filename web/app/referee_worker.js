// referee_worker.js — the match referee, its own thread as its own
// rules engine. It owns the game state (a private go_rules wasm
// instance replays every move — captures, legality and the final TT
// score are adjudicated HERE, not by the player), drives the GTP engine
// through a MessageChannel port, and publishes board snapshots to the
// UI. The engine worker never sees the UI; the UI never touches GTP.

const PASS = 0xFFFF;
const LETTERS = "ABCDEFGHJKLMNOPQRST";       // GTP columns skip I

let R = null;                    // rules module
let game = 0;                    // GoGame* handle
let b = 9;                       // current board size
let human = 1;                   // human color: 1 black, 2 white
let over = true;
let engine = null;               // MessagePort to the engine worker

const pending = new Map();       // gtp req id → {lines, resolve}
let lines = [];                  // response lines of the in-flight command
let engineBusy = false;          // a genmove is in flight: input is locked

const ui = (type, obj) => self.postMessage({ type, ...obj });

// ---- rules helpers -----------------------------------------------------
function call1(fn, ...args) {
    return R.ccall(fn, "number", argTypes(args), args);
}
function argTypes(args) {
    // all ffi surface args are number; ccall needs the type list
    return args.map(() => "number");
}

function snapshot(extra = {}) {
    const buf = R._malloc(b * b);
    R.ccall("go_board", null, ["number", "number"], [game, buf]);
    const stones = new Uint8Array(R.HEAPU8.buffer, buf, b * b).slice();
    R._free(buf);
    ui("board", {
        b, stones,
        side: call1("go_side", game),
        ply: call1("go_ply", game),
        ...extra,
    });
}

function legalSet() {
    const buf = R._malloc(2 * (b * b + 1));
    const n = R.ccall("go_legal", "number",
                      ["number", "number", "number"], [game, buf, b * b + 1]);
    const set = new Set(new Uint16Array(R.HEAPU8.buffer, buf, n));
    R._free(buf);
    return set;
}

// GTP vertices match the C convention exactly (go_rules.c): row 1 is
// the BOTTOM row, move = (row − 1)·b + col
const toVertex = (move) =>
    move === PASS ? "pass" : LETTERS[move % b] + ((move / b | 0) + 1);
function fromVertex(v) {
    if (v === "pass")
        return PASS;
    const x = LETTERS.indexOf(v[0].toUpperCase());
    const y = parseInt(v.slice(1), 10);
    return x >= 0 && x < b && y >= 1 && y <= b ? (y - 1) * b + x : -1;
}

// ---- GTP client --------------------------------------------------------
function gtp(lineCmd) {
    return new Promise((resolve) => {
        const id = Math.random().toString(36).slice(2);
        pending.set(id, { lines: [], resolve });
        engine.postMessage({ type: "gtp", line: lineCmd, id });
    });
}
const gtpText = async (lineCmd) =>
    (await gtp(lineCmd)).lines.join("\n");

// ---- game flow ---------------------------------------------------------
function side() { return call1("go_side", game); }

// The net's read of the current position (black view): one raw eval
// through the engine's `value` command. Best effort — no bar on error.
async function publishValue() {
    try {
        const text = await gtpText("value");
        const m = text.trim().match(/^=\s*([\d.]+) (-?[\d.]+)/);
        if (m)
            ui("value", { winrate: parseFloat(m[1]),
                          score: parseFloat(m[2]) });
    } catch { /* engine busy or no evaluator: keep the last bar */ }
}

function adjudicate() {
    // go_termination(g, *score, *ownership): one call returns the term
    // code (TERM_OVER = 1: double pass or the ply wall) and fills score
    const buf = R._malloc(4);
    const ret = R.ccall("go_termination", "number",
                        ["number", "number", "number"], [game, buf, 0]);
    const score = new Float32Array(R.HEAPU8.buffer, buf, 1)[0];
    R._free(buf);
    return { term: ret !== 0, score };
}

function finish(result, score) {
    over = true;
    ui("result", { result, score });
    ui("status", { text: `game over — ${result}`, thinking: false });
}

function checkEnd() {
    const { term, score } = adjudicate();
    if (!term)
        return false;
    finish(score > 0 ? `B+${score.toFixed(1)}` :
            score < 0 ? `W+${(-score).toFixed(1)}` : "jigo", score);
    return true;
}

async function engineMove(color) {
    ui("status", { text: "engine thinking…", thinking: true });
    const text = await gtpText(`genmove ${color === 1 ? "b" : "w"}`);
    if (over)
        return;
    if (!text.trim().startsWith("="))
        throw new Error(`genmove failed: ${JSON.stringify(text)}`);
    // no-id responses print "=H4" (no space); strip only a numeric id
    const v = text.trim().replace(/^=[0-9]*/, "").trim();
    if (!v)
        throw new Error(`genmove empty payload: ${JSON.stringify(text)}`);
    const mv = fromVertex(v);
    if (mv < 0 || !legalSet().has(mv))
        throw new Error(`engine move unparsed/illegal: ${v}`);
    R.ccall("go_push", null, ["number", "number"], [game, mv]);
    ui("move", { color, move: mv, vertex: toVertex(mv), by: "engine" });
    snapshot();
    checkEnd();
}

async function afterMove() {
    if (over || checkEnd())
        return;
    await publishValue();
    const s = side();
    if (s !== human) {
        engineBusy = true;
        try {
            await engineMove(s);
        } catch (e) {
            ui("status", { text: `engine error: ${e.message}`,
                           thinking: false });
            throw e;
        } finally {
            engineBusy = false;
        }
        // the engine just moved: hand the turn back BEFORE returning
        if (!over)
            ui("status", { text: side() === 1 ? "your move (black)"
                                              : "your move (white)",
                           thinking: false });
    } else {
        ui("status", { text: s === 1 ? "your move (black)"
                                     : "your move (white)",
                       thinking: false });
    }
}

// ---- messages ----------------------------------------------------------
self.onmessage = async (ev) => {
    const m = ev.data;

    if (m.type === "boot") {
        engine = m.port;
        engine.onmessage = (e) => {
            const r = e.data;
            const p = pending.get(r.id);
            if (r.type === "line" && p)
                p.lines.push(r.line);
            else if (r.type === "resp" && p) {
                pending.delete(r.id);
                p.resolve({ lines: p.lines, quit: r.quit });
            } else if (r.type === "ready")
                ui("engine-ready", {});
            else if (r.type === "ep")
                ui("ep", r);
            else if (r.type === "stats")
                ui("stats", r);
            else if (r.type === "fatal")
                ui("status", { text: `engine failed: ${r.error}`,
                               thinking: false });
        };
        const { default: createGoRules } = await import(
            new URL("../../engine/build/wasm/go_rules.js",
                    self.location.href).href);
        R = await createGoRules({
            locateFile: (p) =>
                new URL(`../../engine/build/wasm/${p}`,
                        self.location.href).href,
        });
        R.ccall("go_init", null, [], []);   // zobrist tables, once
        ui("referee-ready", {});
        return;
    }

    if (m.type === "newGame") {
        const { komi = 7.5, humanColor = 1 } = m;
        b = m.b ?? 9;
        human = humanColor;
        over = false;
        if (game)
            R.ccall("go_game_free", null, ["number"], [game]);
        game = R.ccall("go_game_new", "number",
                       ["number", "number", "number"], [b, komi, 3 * b * b]);
        await gtpText(`boardsize ${b}`);
        await gtpText(`komi ${komi}`);
        await gtpText("clear_board");
        ui("new-game", { b, komi, humanColor });
        snapshot();
        await afterMove();
        return;
    }

    if (m.type === "humanMove") {
        if (engineBusy) {
            ui("status", { text: "wait — engine thinking", thinking: true });
            return;
        }
        if (over || !legalSet().has(m.move)) {
            if (!over)
                ui("status", { text: "illegal move", thinking: false });
            return;
        }
        const color = side();
        R.ccall("go_push", null, ["number", "number"], [game, m.move]);
        await gtpText(`play ${toVertex(m.move)}`);
        ui("move", { color, move: m.move, vertex: toVertex(m.move),
                     by: "human" });
        snapshot();
        await afterMove();
        return;
    }

    if (m.type === "humanPass") {
        if (engineBusy || over)
            return;
        const color = side();
        R.ccall("go_push", null, ["number", "number"], [game, PASS]);
        await gtpText("play pass");
        ui("move", { color, move: PASS, vertex: "pass", by: "human" });
        snapshot();
        await afterMove();
        return;
    }

    if (m.type === "resign" && !over) {
        const s = side();
        finish(s === 1 ? "W+R" : "B+R", null);
        return;
    }
};
