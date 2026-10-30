/*
 * go.h — Tromp-Taylor rules engine.
 *
 * Points: index = y*b + x (x = column, y = row; row 0 is GTP row 1).
 * Positional superko: keys[i] is the Zobrist key of the stone placement
 * after i plies (side to move excluded — TT is positional); a board move
 * is legal iff its resulting placement key never appeared before. Suicide
 * is legal (TT clears own color too); single-stone suicide recreates the
 * current placement, so superko already rejects it. Pass is always legal
 * and keeps the key. Game over: two consecutive passes or ply == maxply,
 * then strict TT scoring (stones + empty regions reaching one color only).
 *
 * No unmake: the search copies States forward-only (chess puct.c does the
 * same); replay snapshots live Python-side.
 */
#ifndef GO_H
#define GO_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GO_MAX_B      19
#define GO_MAX_POINTS (GO_MAX_B * GO_MAX_B)   /* 361 */
#define GO_HIST       5     /* encoder's last-move window (go_nn.h) */

#ifndef MAX_PLY
#define MAX_PLY 1024        /* placement-key history depth inside State */
#endif
#define MAX_MOVES (GO_MAX_POINTS + 1)   /* board points + pass */

enum { EMPTY = 0, BLACK = 1, WHITE = 2 };
static inline int stone_opposite(int c) { return 3 - c; }

typedef uint16_t Move;      /* board point 0..b*b-1, or MOVE_PASS */
#define MOVE_PASS 0xFFFF

typedef struct {
    Move moves[MAX_MOVES];
    int  count;
} MoveList;

typedef struct {
    uint8_t  board[GO_MAX_POINTS];
    uint8_t  b;            /* board size, 2..19 (square) */
    uint8_t  side;         /* color to move */
    uint8_t  n_pass;       /* consecutive passes; game over at 2 */
    uint16_t ply;
    uint16_t maxply;       /* adjudicate (TT-score) at this ply */
    float    komi;         /* white's compensation */
    uint64_t keys[MAX_PLY + 1];
    uint16_t hist[GO_HIST]; /* hist[p % GO_HIST] = move played at ply p */
} State;

extern uint64_t zob_stone[3][GO_MAX_POINTS];

/* Nonzero base for keys[0]: the empty board's placement key must differ
 * from 0 (puct's done-guard uses 0 as its "fresh game" sentinel). */
#define GO_KEY_BASE 0x9E3779B97F4A7C15ULL

void go_init(void);        /* fill zobrist tables; once per process */
void go_state_init(State *s, int b, float komi, int maxply);

/* Advance by a move from gen_legal (or MOVE_PASS). */
void make_move(State *s, Move m);

/* Legal moves: empty points whose placement is new, plus pass. Empty when
 * the game is over. Pure — safe on shared States from any thread. */
void gen_legal(const State *s, MoveList *ml);

uint64_t state_poskey(const State *s);   /* == keys[ply] */

enum { TERM_PLAYING = 0, TERM_OVER = 1 };

/* Over = double pass or ply cap. On TERM_OVER fills *score (black margin:
 * B stones+territory − W stones+territory − komi; positive = black wins)
 * and/or *ownership (per point: 1 black, 0 white, 0.5 dame/unreachable).
 * Either pointer may be NULL. Pure. */
int state_termination(const State *s, float *score, float *ownership);

/* GTP vertex ("D4", "pass"); column letters skip I. 0 = ok, -1 = invalid. */
int  go_parse_gtp(const State *s, const char *v, Move *out);
void move_to_gtp(Move m, int b, char out[4]);

#ifdef __cplusplus
}
#endif

#endif /* GO_H */
