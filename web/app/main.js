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
    if (!ready && engineReady)
        $("status").textContent = `loading net for ${b}x${b}…`;
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

const engineInit = () => {
    engine.postMessage({
        type: "init", b: 9, komi: 7.5, sims: 200,
        walkers: 32,          // batch = parked walkers: bigger waves,
        rows: 64,             // fewer evals — GPU dispatch overhead
        seed: 1, maxply: 900, // dominates at tiny batches
        net: params.get("net"), hfToken,
    });
};

function newGame() {
    const b = parsedSize();
    if (!netReadyFor(b)) {
        ensureNet(b);
        syncControls();
        return;
    }
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
        break;
    }
    case "value":
        // the value net's read, black's perspective: bar + numbers
        $("wrfill").style.width = (m.winrate * 100).toFixed(1) + "%";
        $("wrtext").textContent =
            `B ${(m.winrate * 100).toFixed(1)}% · ` +
            `score ${m.score >= 0 ? "+" : ""}${m.score.toFixed(1)}`;
        break;
    case "stats":
        // the honest numbers: which EP ran, eval count and net time
        $("perf").textContent =
            `${m.evals} evals · ${m.netMs}ms net (${m.rows} rows, ` +
            `${m.ep}) · ${m.totalMs}ms total`;
        break;
    case "referee-ready":
        break;                     // the first newGame drives the flow
    }
};

engine.onmessage = (ev) => {
    if (ev.data.type === "ready") {
        engineReady = true;
        syncControls();
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
    syncControls();
    ensureNet(b);
};
// a default empty 9x9 board on load (instead of a black canvas) —
// replaced by the referee's snapshot once a game starts
board.set(9, new Uint8Array(81), null, false);
syncControls();
engineInit();
