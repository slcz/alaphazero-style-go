/*
 * gtp.c — the GTP front-end (play path; no wasm, no GUI).
 *
 *   gtp [device=mps] [rows=0]      env: PUCT_CKPT (required unless
 *                                       device=dummy), GO_SIMS, GO_MAXPLY
 *
 * Implements the required subset only: protocol_version, name, version,
 * known_command, list_commands, quit, boardsize, clear_board, komi,
 * play, genmove. Numeric command ids are honored; responses are the
 * standard "=[id] ...\n\n" / "?[id] msg\n\n".
 *
 * Search budget: GO_SIMS sims per genmove (default 400). genmove never
 * resigns — resignation is the DRIVER's call (the match referee reads
 * the engine's value through the mux protocol, not GTP). Pass is played
 * when the search's best move is pass.
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

static const char *GTP_COMMANDS[] = {
    "protocol_version", "name", "version", "known_command",
    "list_commands", "quit", "boardsize", "clear_board", "komi",
    "play", "genmove",
};

static PuctGame *g;
static int g_sims = 400;

static void respond(const char *id, const char *fmt, const char *payload) {
    if (id && *id)
        printf("=%s %s\n\n", id, payload);
    else
        printf("=%s\n\n", payload ? payload : "");
    (void)fmt;
    fflush(stdout);
}

static void respond_ok(const char *id) {
    if (id && *id)
        printf("=%s \n\n", id);
    else
        printf("= \n\n");
    fflush(stdout);
}

static void respond_err(const char *id, const char *msg) {
    if (id && *id)
        printf("?%s %s\n\n", id, msg);
    else
        printf("? %s\n\n", msg);
    fflush(stdout);
}

static int cmd_index(const char *cmd) {
    for (size_t i = 0; i < sizeof GTP_COMMANDS / sizeof *GTP_COMMANDS; i++)
        if (strcmp(cmd, GTP_COMMANDS[i]) == 0)
            return (int)i;
    return -1;
}

/* GTP colors are the one-letter forms ("b"/"w"); the long spellings are
 * tolerated. A token is only ever a color if it is not a vertex, and the
 * move parse runs first, so "B4" stays a move. */
static int parse_color(const char *tok) {
    if (!tok || !*tok)
        return -1;
    switch (tok[0]) {
    case 'b': case 'B': return BLACK;
    case 'w': case 'W': return WHITE;
    }
    return -1;
}

/* play <move> | play <color> <move> — the move must be legal; a color
 * token must match the side to move. */
static void do_play(char *args, const char *id) {
    char *tok = strtok(args, " \t\n");
    if (!tok) {
        respond_err(id, "play needs a move");
        return;
    }
    Move mv;
    if (go_parse_gtp(puct_state(g), tok, &mv) != 0) {
        /* maybe a color token first */
        int color = parse_color(tok);
        if (color < 0) {
            respond_err(id, "illegal move");
            return;
        }
        if (color != puct_state(g)->side) {
            respond_err(id, "out of turn");
            return;
        }
        tok = strtok(NULL, " \t\n");
        if (!tok || go_parse_gtp(puct_state(g), tok, &mv) != 0) {
            respond_err(id, "illegal move");
            return;
        }
    }
    MoveList ml;
    gen_legal(puct_state(g), &ml);
    for (int i = 0; i < ml.count; i++)
        if (ml.moves[i] == mv) {
            puct_play(g, mv);
            respond_ok(id);
            return;
        }
    respond_err(id, "illegal move");
}

static void do_genmove(char *args, const char *id) {
    char *tok = args ? strtok(args, " \t\n") : NULL;
    if (tok) {
        int color = parse_color(tok);
        if (color < 0) {
            respond_err(id, "bad color");
            return;
        }
        if (color != puct_state(g)->side) {
            respond_err(id, "out of turn");
            return;
        }
    }
    if (puct_game_over(g)) {
        respond(id, "%s", "pass");       /* game already decided */
        return;
    }
    puct_think(g, g_sims);
    Move mv = puct_best_move(g);
    char v[6];
    move_to_gtp(mv, puct_state(g)->b, v);
    puct_play(g, mv);
    respond(id, "%s", v);
}

int main(int argc, char **argv) {
    const char *device = argc > 1 ? argv[1] : "mps";
    int rows = argc > 2 ? atoi(argv[2]) : 0;
    if (rows <= 0)
        rows = 1024;
    const char *ckpt = getenv("PUCT_CKPT");
    const char *senv = getenv("GO_SIMS");
    const char *menv = getenv("GO_MAXPLY");
    if (senv && *senv)
        g_sims = atoi(senv);
    int maxply = menv && *menv ? atoi(menv) : MAX_PLY;

    go_init();
    rng_seed(((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid()
             * 0x9E3779B97F4A7C15ull);
    g = puct_game_new(0, 19, 7.5f, maxply);
    if (!g) {
        fprintf(stderr, "gtp: game alloc failed\n");
        return 1;
    }
    if (strcmp(device, "dummy") != 0) {
        if (!ckpt || !*ckpt) {
            fprintf(stderr, "gtp: PUCT_CKPT not set\n");
            return 2;
        }
        if (puct_attach_nn(ckpt, device, rows) != 0) {
            fprintf(stderr, "gtp: puct_attach_nn failed\n");
            return 1;
        }
    } else {
        puct_zone_init(rows);
    }

    char line[4096];
    while (fgets(line, sizeof line, stdin)) {
        char *p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '#' || *p == '\n' || !*p)
            continue;                    /* comment / empty */
        char id[16] = {0};
        char *cmd = p;
        /* numeric command id */
        while (*p >= '0' && *p <= '9')
            p++;
        if (p > cmd && *p == ' ') {
            size_t n = (size_t)(p - cmd);
            if (n >= sizeof id)
                n = sizeof id - 1;
            memcpy(id, cmd, n);
            cmd = ++p;
        }
        char *nl = strchr(cmd, '\n');
        if (nl)
            *nl = '\0';
        char *sp = strchr(cmd, ' ');
        char *args = NULL;
        if (sp) {
            *sp = '\0';
            args = sp + 1;
        }
        switch (cmd_index(cmd)) {
        case 0:
            respond(id, "%s", "2");
            break;
        case 1:
            respond(id, "%s", "gogo");
            break;
        case 2:
            respond(id, "%s", "0.1");
            break;
        case 3: {
            char *c = strtok(args, " \t\n");
            if (c && cmd_index(c) >= 0)
                respond(id, "%s", "true");
            else
                respond(id, "%s", "false");
            break;
        }
        case 4: {
            printf("=");
            if (id && *id)
                printf("%s ", id);
            for (size_t i = 0;
                 i < sizeof GTP_COMMANDS / sizeof *GTP_COMMANDS; i++)
                printf("%s\n", GTP_COMMANDS[i]);
            printf("\n");
            fflush(stdout);
            break;
        }
        case 5:
            return 0;
        case 6: {
            char *v = strtok(args, " \t\n");
            int b = v ? atoi(v) : 0;
            if (b < 2 || b > GO_MAX_B ||
                puct_game_reconfig(g, b, puct_state(g)->komi,
                                   puct_state(g)->maxply) != 0) {
                respond_err(id, "unsupported board size");
                break;
            }
            respond_ok(id);
            break;
        }
        case 7:
            puct_game_reset(g);
            respond_ok(id);
            break;
        case 8: {
            char *v = strtok(args, " \t\n");
            if (!v) {
                respond_err(id, "komi needs a value");
                break;
            }
            float komi = (float)atof(v);
            /* mid-game komi changes are pathological under GTP; the tree
             * is cleared only when a terminal was already reached, so a
             * running game keeps its search (terminal values recompute
             * on the next game anyway) */
            puct_game_reconfig(g, puct_state(g)->b, komi,
                               puct_state(g)->maxply);
            respond_ok(id);
            break;
        }
        case 9:
            do_play(args, id);
            break;
        case 10:
            do_genmove(args, id);
            break;
        default:
            respond_err(id, "unknown command");
            break;
        }
    }
    return 0;
}
