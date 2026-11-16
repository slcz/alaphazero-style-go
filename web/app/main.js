// main.js — the UI thread: page controls, the board canvas, and the two
// workers (referee + engine) joined by a MessageChannel. All game flow
// goes through the referee; this file only renders and forwards clicks.
import { Board } from "./board.js";

const $ = (id) => document.getElementById(id);
const parsedSize = () => parseInt($("size").value, 10);

const referee = new Worker(new URL("referee_worker.js", import.meta.url));
const engine = new Worker(new URL("engine_worker.js", import.meta.url));
const channel = new MessageChannel();
referee.postMessage({ type: "boot", port: channel.port1 }, [channel.port1]);
engine.postMessage({ type: "bind", port: channel.port2 }, [channel.port2]);

let humanColor = 1;
let moveNo = 0;
let lastMove = null;
let myTurn = false;

// The engine inits ONCE (the eval zone is per-process); New Game only
// reconfigures — sims via the export, board/komi via GTP from the referee.
let engineReady = false;       // the wasm engine has booted
let ortStarted = false;        // the first ONNX session attempt has finished
let refereeReady = false;      // the rules wasm has booted
let autoStartPending = true;   // start a game when the net is ready
const loadedSizes = new Set(); // board sizes whose net is ready
const failedSizes = new Set(); // sizes whose net failed (dummy eval)
const pendingSizes = new Set();// preloads in flight

function netReadyFor(b) {
    return loadedSizes.has(b) || failedSizes.has(b);
}

// Keep the game unplayable until the selected size's checkpoint is
// loaded (or has failed and the engine is on the dummy evaluator).
function syncControls() {
    const b = parsedSize();
    const ready = netReadyFor(b);
    for (const id of ["new", "pass", "resign"])
        $(id).disabled = !ready;
    $("board").style.pointerEvents = ready ? "" : "none";
    $("board").style.opacity = ready ? "1" : "0.55";
    $("netbar").style.display = ready ? "none" : "block";
    if (!ready) {
        $("netfill").style.width = "0%";
        if (engineReady)
            $("status").textContent = `loading net for ${b}x${b}…`;
    }
}

// bytes → status line + bar. `total` comes from the ONNX response (or
// latest.json's asset size); it may be 0 for a custom URL.
function setProgress(p) {
    if (!p)
        return;
    const b = parsedSize();
    const pct = p.total
        ? Math.min(100, Math.round(100 * p.received / p.total)) : 0;
    $("netbar").style.display = "block";
    $("netfill").style.width = pct + "%";
    if (p.cached) {
        $("status").textContent = `loading net for ${b}x${b}… cached`;
    } else if (p.total) {
        $("status").textContent = pct >= 100
            ? `loading net for ${b}x${b}… initializing`
            : `loading net for ${b}x${b}… ${pct}%`;
    } else {
        $("status").textContent = `loading net for ${b}x${b}… ` +
            `${(p.received / 1e6).toFixed(1)} MB`;
    }
}

// Start a game once both the rules worker and the selected size's net
// are ready. Called for the first load and again after a board-size
// switch, so neither needs a New Game click.
function maybeAutoStart() {
    if (!autoStartPending || !engineReady || !refereeReady)
        return;
    const b = parsedSize();
    if (!netReadyFor(b))
        return;
    autoStartPending = false;
    newGame();
}

function ensureNet(b) {
    // Before the first ep there is no ORT module to preload with; the
    // init ep handler calls us again for the currently selected size.
    if (!ortStarted || netReadyFor(b) || pendingSizes.has(b))
        return;
    pendingSizes.add(b);
    engine.postMessage({ type: "preload", b });
    $("status").textContent = `loading net for ${b}x${b}…`;
    syncControls();
}

const board = new Board($("board"), (move) => {
    if (!netReadyFor(parsedSize())) {
        $("status").textContent =
            `loading net for ${parsedSize()}x${parsedSize()}…`;
        return;
    }
    referee.postMessage({ type: "humanMove", move });
});

// Net source: default = the latest checkpoint on HuggingFace (its
// web/latest.json pointer, then the stable web/net-{b}.onnx files). A
// private fork can supply a token via ?hf=<token> — stored in
// localStorage after first use. A custom ?net=<url-with-{b}> wins, and
// ?net=local uses the bundled assets.
const params = new URLSearchParams(location.search);
if (params.get("hf"))
    localStorage.setItem("hfToken", params.get("hf"));
const hfToken = localStorage.getItem("hfToken");

// Web-friendly checkpoints: it026 is the default stable/manifest net;
// the older iterations live in versioned HF directories.
const CKPTS = {
    it026: null,
    it023: "https://huggingface.co/chriszhang01/gonet/resolve/main/web/it023/net-{b}.onnx",
    it010: "https://huggingface.co/chriszhang01/gonet/resolve/main/web/it010/net-{b}.onnx",
    it004: "https://huggingface.co/chriszhang01/gonet/resolve/main/web/it004/net-{b}.onnx",
};
const requestedCkpt = params.get("ckpt");
const ckpt = requestedCkpt &&
    Object.prototype.hasOwnProperty.call(CKPTS, requestedCkpt)
    ? requestedCkpt : "it026";
const netSource = params.get("net") || CKPTS[ckpt];


// WebKit/Safari tabs get much less headroom than Chrome. The 19x19
// attention scores grow as B·heads·361²; a 32-row batch is ~130 MB of
// fp32 scores, while 8 rows is ~33 MB. Keep the batch small there and
// use plain WebGPU (Safari's graph capture path is the fragile one).
const ua = navigator.userAgent;
const isIOS = /iPad|iPhone|iPod/.test(ua) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const isSafari = /Safari/.test(ua) &&
    !/Chrome|Chromium|CriOS|FxiOS|Edg|OPR/i.test(ua);
const lowMem = isIOS || isSafari;

const engineInit = () => {
    engine.postMessage({
        type: "init", b: 9, komi: 7.5, sims: 200,
        walkers: lowMem ? 8 : 32,  // batch = parked walkers
        rows: lowMem ? 16 : 64,    // fewer evals — GPU dispatch overhead
        seed: 1, maxply: 900, // dominates at tiny batches
        net: netSource, hfToken,
    });
};

function newGame() {
    const b = parsedSize();
    if (!netReadyFor(b)) {
        ensureNet(b);
        syncControls();
        return;
    }
    autoStartPending = false;
    const komi = parseFloat($("komi").value);
    humanColor = $("color-b").checked ? 1 : 2;
    const sims = parseInt($("sims").value, 10);
    engine.postMessage({ type: "setSims", sims });
    referee.postMessage({ type: "newGame", b, komi, humanColor });
    moveNo = 0;
    lastMove = null;
    $("moves").textContent = "";
    $("result").textContent = "";
}

// referee → UI
referee.onmessage = (ev) => {
    const m = ev.data;
    switch (m.type) {
    case "board":
        // ghost color: the human's actual color (0 = none), so the
        // hover stone matches whose turn it is to click
        board.set(m.b, m.stones, lastMove,
                  m.side === humanColor ? humanColor : 0);
        myTurn = m.side === humanColor;
        $("ply").textContent = `ply ${m.ply}`;
        break;
    case "new-game":
        board.set(m.b, new Uint8Array(m.b * m.b), null, m.humanColor);
        $("status").textContent = netReadyFor(parsedSize())
            ? "new game" : "new game — loading net…";
        break;
    case "move":
        lastMove = m.move === 0xFFFF ? null : m.move;
        moveNo++;
        $("moves").textContent +=
            `${moveNo}. ${m.by === "human" ? "you" : "ai"} ` +
            `${m.color === 1 ? "B" : "W"} ${m.vertex}\n`;
        $("moves").scrollTop = $("moves").scrollHeight;
        break;
    case "status":
        $("status").textContent = m.text;
        $("board").style.opacity = m.thinking ? 0.7 : 1;
        break;
    case "result":
        $("result").textContent = m.result;
        break;
    case "engine-ready":
        engineReady = true;
        syncControls();
        break;
    case "ep": {
        const b = m.b ?? 9;
        pendingSizes.delete(b);
        ortStarted = true;
        // The engine keeps one ONNX session: whichever size just
        // loaded evicted every other size's session.
        loadedSizes.clear();
        failedSizes.clear();
        if (m.net === "FAILED")
            failedSizes.add(b);
        else
            loadedSizes.add(b);
        if (b === parsedSize()) {
            $("status").textContent = m.net === "FAILED"
                ? `engine ready — net FAILED (${m.error ?? "no web net"}), ` +
                  `plays on dummy eval`
                : `engine ready — net ${m.net}` +
                  (m.checkpoint ? ` (${m.checkpoint})` : "") +
                  `, webgpu ${m.webgpu}`;
        } else {
            ensureNet(parsedSize());
        }
        syncControls();
        maybeAutoStart();
        break;
    }
    case "value":
        // the value net's read, black's perspective: bar + numbers
        $("wrfill").style.width = (m.winrate * 100).toFixed(1) + "%";
        $("wrtext").textContent =
            `B ${(m.winrate * 100).toFixed(1)}% · ` +
            `score ${m.score >= 0 ? "+" : ""}${m.score.toFixed(1)}`;
        break;
    case "net-progress":
        setProgress(m);
        break;
    case "stats":
        // the honest numbers: which EP ran, eval count and net time
        $("perf").textContent =
            `${m.evals} evals · ${m.netMs}ms net (${m.rows} rows, ` +
            `${m.ep}) · ${m.totalMs}ms total`;
        break;
    case "referee-ready":
        refereeReady = true;
        maybeAutoStart();
        break;
    }
};

engine.onmessage = (ev) => {
    if (ev.data.type === "ready") {
        engineReady = true;
        syncControls();
        maybeAutoStart();
    }
    if (ev.data.type === "fatal")
        $("status").textContent = `engine failed: ${ev.data.error}`;
};

$("new").onclick = () => newGame();
$("pass").onclick = () => {
    if (netReadyFor(parsedSize()))
        referee.postMessage({ type: "humanPass" });
};
$("resign").onclick = () => {
    if (netReadyFor(parsedSize()))
        referee.postMessage({ type: "resign" });
};
$("size").onchange = () => {
    // sensible komi defaults per size (the selfplay loop's convention)
    const b = parsedSize();
    $("komi").value = b === 7 ? "9.5" : "7.5";
    // show the new grid immediately; the cell clicks stay locked until
    // the preloaded session lands, then a fresh game starts by itself
    board.set(b, new Uint8Array(b * b), null, 0);
    autoStartPending = true;
    syncControls();
    ensureNet(b);
    maybeAutoStart();
};
// a default empty 9x9 board on load (instead of a black canvas) —
// replaced by the referee's snapshot once a game starts
board.set(9, new Uint8Array(81), null, false);
syncControls();
engineInit();
