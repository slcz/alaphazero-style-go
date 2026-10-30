/*
 * engine_mux.c — multi-game think server for the gate referee (ported
 * from the chess engine's uci_mux; the Go dialect speaks u16 points).
 * One process = one network + one mux hosting ALL of that network's
 * games, so a phase's thinks batch across every game on the GPU.
 *
 *   engine_mux <games> [rows=0] [device=cuda] [seed]
 *   env: PUCT_CKPT (required), PUCT_WALKERS, PUCT_SAMPLE_PLYS
 *
 * Line protocol on stdin, one reply per queued game:
 *   game <gid> config <b> <komi> <maxply>   replant the game (fresh tree)
 *   game <gid> moves <m> [...]              advance (opponent replies since
 *                                           the last phase); m = point
 *                                           index or "pass" (GTP letters
 *                                           accepted too)
 *   game <gid> go nodes <sims>              queue this game's think
 *   run                                     one mux round over ALL queued
 *   → game <gid> bestmove <m> value <v>     m = point index or "pass";
 *                                           v = mover-view root value
 *                                           (the referee's resign input)
 *   quit                                    exit 0
 *
 * sims <= 1 is the raw-policy player: argmax PRIORS (a think(1) visit
 * would concentrate on one child). PUCT_SAMPLE_PLYS=N samples ∝ visits
 * for the first N plies (self-play-style variety); default 0 =
 * deterministic argmax (the eval protocol). The referee owns correctness:
 * illegal moves / malformed input are fatal (exit 1).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "go.h"
#include "go_nn_eval.h"
#include "go_puct.h"
#include "go_rng.h"

typedef struct {
    PuctGame *g;
    int      pending;
    int      sims;
} Slot;

static Slot *slots;
static int   n_slots;
static int  *queue;
static int   queued;
static int   g_sample_plys;

static void fail(const char *msg) {
    fprintf(stderr, "engine_mux: %s\n", msg);
    exit(1);
}

static Slot *slot_of(int gid) {
    if (gid < 0 || gid >= n_slots)
        fail("bad gid");
    return &slots[gid];
}

/* "pass" = MOVE_PASS; "24" = point index; "D4" = GTP vertex. Failure is
 * signalled via *is_fail (point 0 = A1 is legal, so a bare parse can't
 * distinguish). */
static Move parse_any(const State *s, const char *tok, int *is_fail) {
    *is_fail = 0;
    if (strcmp(tok, "pass") == 0)
        return MOVE_PASS;
    char *end;
    long v = strtol(tok, &end, 10);
    if (end != tok && *end == '\0') {
        if (v >= 0 && v < s->b * s->b)
            return (Move)v;
        *is_fail = 1;
        return MOVE_PASS;
    }
    Move mv;
    if (go_parse_gtp(s, tok, &mv) == 0)
        return mv;
    *is_fail = 1;
    return MOVE_PASS;
}

static void emit_bestmove(int gid, Slot *s, int sims) {
    Move mv;
    float value;
    if (sims <= 1) {
        /* raw-policy player (the ladder baseline): argmax the PRIORS —
         * a think(1) visit would concentrate on one child */
        uint16_t mv2[MAX_MOVES];
        float prior[MAX_MOVES];
        int n = puct_root_children(s->g, mv2, NULL, NULL, prior);
        if (n <= 0)
            fail("no legal move: game was terminal");
        int best = 0;
        for (int i = 1; i < n; i++)
            if (prior[i] > prior[best])
                best = i;
        mv = mv2[best];
    } else {
        mv = puct_best_move(s->g);
        /* PUCT_SAMPLE_PLYS: sample ∝ visits for the first N plies */
        if (g_sample_plys > 0 && puct_state(s->g)->ply < g_sample_plys) {
            uint16_t mv3[MAX_MOVES];
            uint32_t vi[MAX_MOVES];
            double w[MAX_MOVES];
            int n = puct_root_children(s->g, mv3, vi, NULL, NULL);
            if (n <= 0)
                fail("no legal move: game was terminal");
            for (int i = 0; i < n; i++)
                w[i] = (double)vi[i];
            mv = mv3[sample_index(w, n, 1.0)];
        }
    }
    uint32_t root_visits;
    double wsum;
    puct_root_stats(s->g, &root_visits, &wsum);
    value = root_visits ? (float)-(wsum / root_visits) : 0.0f;
    if (mv == MOVE_PASS)
        printf("game %d bestmove pass value %+.3f\n", gid, value);
    else
        printf("game %d bestmove %d value %+.3f\n", gid, (int)mv, value);
    if (puct_play(s->g, mv) != 0)
        fail("puct_play failed on own bestmove");
    puct_set_suspended(s->g, true);
    s->pending = 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <games> [rows=0] [device=cuda] [seed]\n"
                        "env: PUCT_CKPT (required), PUCT_WALKERS, "
                        "PUCT_SAMPLE_PLYS\n", argv[0]);
        return 2;
    }
    n_slots = atoi(argv[1]);
    if (n_slots < 1)
        n_slots = 1;
    int rows = argc > 2 ? atoi(argv[2]) : 0;
    const char *device = argc > 3 ? argv[3] : "cuda";
    uint64_t mseed = argc > 4 ? strtoull(argv[4], NULL, 10) : 0;
    if (!mseed)
        mseed = ((uint64_t)time(NULL) << 32)
              ^ (uint64_t)getpid() * 0x9E3779B97F4A7C15ull;
    const char *ckpt = getenv("PUCT_CKPT");
    if (!ckpt || !*ckpt) {
        fprintf(stderr, "PUCT_CKPT not set\n");
        return 2;
    }
    if (rows <= 0)
        rows = 8 * n_slots;

    go_init();
    PuctGame **handles = calloc((size_t)n_slots, sizeof *handles);
    slots = calloc((size_t)n_slots, sizeof *slots);
    queue = calloc((size_t)n_slots, sizeof *queue);
    if (!handles || !slots || !queue)
        fail("oom");
    /* games start 19x19; the referee configs each slot before use */
    if (puct_mux_new(n_slots, 0, 19, 7.5f, MAX_PLY, handles) != 0)
        fail("mux_new failed");
    if (strcmp(device, "dummy") == 0)
        puct_zone_init(rows);     /* built-in dummy evaluator (no net) */
    else if (puct_attach_nn(ckpt, device, rows) != 0)
        fail("puct_attach_nn failed");
    const char *wenv = getenv("PUCT_WALKERS");
    const char *senv = getenv("PUCT_SAMPLE_PLYS");
    if (senv && *senv) {
        g_sample_plys = atoi(senv);
        rng_seed(mseed);
    }
    for (int i = 0; i < n_slots; i++) {
        slots[i].g = handles[i];
        puct_set_suspended(slots[i].g, true);
        if (wenv && *wenv)
            puct_set_walkers(slots[i].g, atoi(wenv));
    }
    printf("ready\n");
    fflush(stdout);

    char line[65536];
    while (fgets(line, sizeof line, stdin)) {
        char *cmd = strtok(line, " \t\n");
        if (!cmd)
            continue;
        if (!strcmp(cmd, "quit"))
            break;
        if (!strcmp(cmd, "run")) {
            if (!queued)
                continue;
            int sims = slots[queue[0]].sims;
            puct_mux_step(sims);      /* ends fully delivered */
            for (int q = 0; q < queued; q++)
                emit_bestmove(queue[q], &slots[queue[q]], sims);
            queued = 0;
            fflush(stdout);
            continue;
        }
        if (!strcmp(cmd, "game")) {
            char *gid_s = strtok(NULL, " \t\n");
            if (!gid_s)
                fail("missing gid");
            Slot *s = slot_of(atoi(gid_s));
            char *sub = strtok(NULL, " \t\n");
            if (!sub)
                fail("missing subcommand");
            if (!strcmp(sub, "config")) {
                char *b = strtok(NULL, " \t\n");
                char *k = strtok(NULL, " \t\n");
                char *mp = strtok(NULL, " \t\n");
                if (!b || !k || !mp ||
                    puct_game_reconfig(s->g, atoi(b), (float)atof(k),
                                       atoi(mp)) != 0)
                    fail("bad config");
            } else if (!strcmp(sub, "moves")) {
                char *t;
                while ((t = strtok(NULL, " \t\n"))) {
                    int bad;
                    Move mv = parse_any(puct_state(s->g), t, &bad);
                    if (bad)
                        fail("bad move token");
                    MoveList ml;
                    gen_legal(puct_state(s->g), &ml);
                    int ok = 0;
                    for (int i = 0; i < ml.count; i++)
                        if (ml.moves[i] == mv)
                            ok = 1;
                    if (!ok || puct_play(s->g, mv) != 0)
                        fail("illegal move");
                }
            } else if (!strcmp(sub, "go")) {
                char *kw = strtok(NULL, " \t\n");
                char *v = strtok(NULL, " \t\n");
                if (!kw || strcmp(kw, "nodes") || !v)
                    fail("go needs 'nodes <sims>'");
                if (s->pending)
                    fail("game already queued this phase");
                if (queued && atoi(v) != slots[queue[0]].sims)
                    fail("mixed sims in one phase");
                s->pending = 1;
                s->sims = atoi(v);
                puct_set_suspended(s->g, false);
                queue[queued++] = s - slots;
            } else {
                fail("unknown subcommand");
            }
            continue;
        }
        fprintf(stderr, "engine_mux: unknown command '%s'\n", cmd);
        exit(1);
    }
    return 0;
}
