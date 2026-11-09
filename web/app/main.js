// main.js — the UI thread: page controls, the board canvas, and the two
// workers (referee + engine) joined by a MessageChannel. All game flow
// goes through the referee; this file only renders and forwards clicks.
import { Board } from "./board.js";

const $ = (id) => document.getElementById(id);

const referee = new Worker(new URL("referee_worker.js", import.meta.url));
const engine = new Worker(new URL("engine_worker.js", import.meta.url));
const channel = new MessageChannel();
referee.postMessage({ type: "boot", port: channel.port1 }, [channel.port1]);
engine.postMessage({ type: "bind", port: channel.port2 }, [channel.port2]);

const board = new Board($("board"), (move) => {
    referee.postMessage({ type: "humanMove", move });
});

let humanColor = 1;
let moveNo = 0;
let lastMove = null;
let myTurn = false;

// The engine inits ONCE (the eval zone is per-process); New Game only
// reconfigures — sims via the export, board/komi via GTP from the referee.
let engineReady = false;
let netReady = false;       // the ONNX session is loaded (or failed)
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
    const b = parseInt($("size").value, 10);
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
        $("status").textContent = netReady ? "new game"
                                               : "new game — loading net…";
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
        $("status").textContent = "engine ready — loading net…";
        break;
    case "ep":
        netReady = true;
        $("status").textContent = m.net === "FAILED"
            ? `engine ready — net FAILED (${m.error ?? "no web net"}), ` +
              `plays on dummy eval`
            : `engine ready — net ${m.net}` +
              (m.checkpoint ? ` (${m.checkpoint})` : "") +
              `, webgpu ${m.webgpu}`;
        break;
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
        $("status").textContent = "engine ready — loading net…";
    }
    if (ev.data.type === "fatal")
        $("status").textContent = `engine failed: ${ev.data.error}`;
};

$("new").onclick = () => newGame();
$("pass").onclick = () => referee.postMessage({ type: "humanPass" });
$("resign").onclick = () => referee.postMessage({ type: "resign" });
$("size").onchange = () => {
    // sensible komi defaults per size (the selfplay loop's convention)
    $("komi").value = parseInt($("size").value, 10) === 7 ? "9.5" : "7.5";
};
// a default empty 9x9 board on load (instead of a black canvas) —
// replaced by the referee's snapshot once a game starts
board.set(9, new Uint8Array(81), null, false);
engineInit();
