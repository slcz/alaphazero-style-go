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
const engineInit = () => {
    engine.postMessage({
        type: "init", b: 9, komi: 7.5, sims: 200,
        walkers: 32,          // batch = parked walkers: bigger waves,
        rows: 64,             // fewer evals — GPU dispatch overhead
        seed: 1, maxply: 900, // dominates at tiny batches
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
        board.set(m.b, m.stones, lastMove, m.side === humanColor);
        myTurn = m.side === humanColor;
        $("ply").textContent = `ply ${m.ply}`;
        break;
    case "new-game":
        board.set(m.b, new Uint8Array(m.b * m.b), null, m.humanColor === 1);
        $("status").textContent = "new game";
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
        $("status").textContent = "engine ready";
        break;
    case "ep":
        $("status").textContent =
            `engine ready — net ${m.net}, webgpu ${m.webgpu}`;
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
        $("status").textContent = "engine ready";
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
board.b = 0;
engineInit();
