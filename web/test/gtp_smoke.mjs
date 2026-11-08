// gtp_smoke.mjs — drive the wasm GTP engine in node with the dummy
// evaluator (no net attached): protocol sanity, a played game, and the
// quit handshake. Run: node web/test/gtp_smoke.mjs
import createGoEngine from "../../engine/build/wasm/go_engine.js";

const lines = [];
const M = await createGoEngine({
    print: (l) => lines.push(l),
});

// 7x7, komi 9.5, 64 sims, walkers 8, zone rows 32, fixed seed
if (M.ccall("gtp_wasm_init", "number",
            ["number", "number", "number", "number", "number",
             "number", "number"],
            [7, 9.5, 150, 64, 8, 32, 12345n]) !== 0)
    throw new Error("gtp_wasm_init failed");

const exec = (line) =>
    M.ccall("gtp_exec", "number", ["string"], [line]);

function ask(line) {
    lines.length = 0;
    const quit = exec(line);
    const out = lines.join("\n");
    return { quit, out };
}

let failed = 0;
function expect(cond, what, got) {
    if (!cond) {
        failed++;
        console.error(`FAIL ${what}\n  got: ${JSON.stringify(got)}`);
    } else {
        console.log(`ok   ${what}`);
    }
}

let r = ask("protocol_version");
expect(r.out.startsWith("=2") && r.quit === 0, "protocol_version", r);

r = ask("name");
expect(r.out.includes("gogo"), "name", r);

r = ask("boardsize 9");
expect(r.out.startsWith("="), "boardsize 9", r);

r = ask("komi 7.5");
expect(r.out.startsWith("="), "komi 7.5", r);

r = ask("play b D5");
expect(r.out.startsWith("="), "play b D5", r);

r = ask("play w E3");
expect(r.out.startsWith("="), "play w E3", r);

r = ask("genmove b");
expect(/^= ?[A-HJ][1-9]|^= ?pass/.test(r.out.trim()), "genmove b", r);

r = ask("illegalcommand");
expect(r.out.startsWith("?"), "unknown command -> error response", r);

r = ask("boardsize 19");
expect(r.out.startsWith("="), "boardsize 19 (engine replants)", r);
r = ask("boardsize 7");
expect(r.out.startsWith("="), "back to 7x7", r);

r = ask("10 genmove b");
expect(r.out.startsWith("=10"), "numeric command id echoed", r);

r = ask("quit");
expect(r.quit === 1, "quit returns 1", r);

// a full tiny game: alternate genmove until double pass / wall. The
// zone is once-per-process, so a fresh game is boardsize + clear_board
// (the same restart path the GUI will use).
ask("boardsize 5");
ask("clear_board");
let plies = 0, passes = 0;
while (plies < 60 && passes < 2) {
    r = ask(`genmove ${plies % 2 ? "w" : "b"}`);
    const mv = r.out.trim().slice(1).trim();
    if (!/^[A-Z][0-9]|pass/.test(mv))
        throw new Error(`bad genmove response: ${r.out}`);
    passes = mv === "pass" ? passes + 1 : 0;
    plies++;
}
expect(passes === 2 || plies === 60, `tiny 5x5 game ended (${plies} plies)`,
       { plies, passes });

console.log(failed ? `\n${failed} FAILED` : "\nall green");
process.exit(failed ? 1 : 0);
