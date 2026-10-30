/* ffi.c — the Python-facing shim (go_ffi.h). One malloc'd State per
 * handle; go_push validates against gen_legal (replay-speed, not search). */
#include "go_ffi.h"

#include <stdlib.h>
#include <string.h>

#include "go.h"
#include "go_nn.h"

struct GoGame { State st; };

GoGame *go_game_new(int b, float komi, int maxply) {
    if (b < 2 || b > GO_MAX_B || maxply < 1 || maxply > MAX_PLY)
        return NULL;
    GoGame *g = malloc(sizeof *g);
    if (g)
        go_state_init(&g->st, b, komi, maxply);
    return g;
}

void go_game_free(GoGame *g) { free(g); }

int  go_side(const GoGame *g)      { return g->st.side; }
int  go_ply(const GoGame *g)       { return g->st.ply; }
int  go_board_size(const GoGame *g){ return g->st.b; }
int  go_n_pass(const GoGame *g)    { return g->st.n_pass; }

void go_board(const GoGame *g, uint8_t *out) {
    memcpy(out, g->st.board, (size_t)g->st.b * g->st.b);
}

int go_legal(GoGame *g, uint16_t *out, int max) {
    MoveList ml;
    gen_legal(&g->st, &ml);
    int n = ml.count > max ? max : ml.count;
    memcpy(out, ml.moves, (size_t)n * sizeof *out);
    return ml.count;
}

int go_push(GoGame *g, uint16_t move) {
    if (move != MOVE_PASS &&
        (move >= g->st.b * g->st.b || g->st.board[move] != EMPTY))
        return -1;
    MoveList ml;
    gen_legal(&g->st, &ml);
    for (int i = 0; i < ml.count; i++)
        if (ml.moves[i] == move) {
            make_move(&g->st, move);
            return 0;
        }
    return -1;
}

int go_push_fast(GoGame *g, uint16_t move) {
    if (move == MOVE_PASS) {
        make_move(&g->st, MOVE_PASS);
        return 0;
    }
    if (move >= g->st.b * g->st.b || g->st.board[move] != EMPTY)
        return -1;
    make_move(&g->st, move);
    return 0;
}

int go_termination(const GoGame *g, float *score, float *ownership) {
    return state_termination(&g->st, score, ownership);
}

int go_encode(const GoGame *g, float *out) {
    nn_encode(&g->st, out);
    return 0;
}
