/*
 * go_ffi.h — the C-ABI surface consumed by Python (cffi, ABI mode).
 *
 * An opaque handle per game owning one State (go.h); small plain types
 * across the boundary. Moves are u16 points (or MOVE_PASS) — the same
 * encoding the shard and the policy slots use, never strings. There is
 * no pop: replay is forward-only (the data prep walks once, caching
 * what it needs). The handle is not thread-safe; the zobrist tables are
 * read-only after go_init, so any number of handles may coexist.
 */
#ifndef GO_FFI_H
#define GO_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GO_FFI_NN_TOTAL (361 * 18)   /* == NN_TOTAL; standalone by design */

typedef struct GoGame GoGame;

/* Call once per process before creating games. */
void go_init(void);

/* A game at the empty b x b board. NULL on bad args. */
GoGame *go_game_new(int b, float komi, int maxply);
void    go_game_free(GoGame *g);

int  go_side(const GoGame *g);          /* BLACK / WHITE */
int  go_ply(const GoGame *g);
int  go_board_size(const GoGame *g);
int  go_n_pass(const GoGame *g);
void go_board(const GoGame *g, uint8_t *out);   /* b*b entries */

/* Legal moves (u16, gen_legal order: points ascending, pass last) into
 * `out` (caller gives >= 362 slots). Returns the count. */
int  go_legal(GoGame *g, uint16_t *out, int max);

/* Play a move. 0 = played, -1 = illegal/malformed. */
int  go_push(GoGame *g, uint16_t move);

/* Play a TRUSTED move (shard replay): occupancy check + apply, no
 * gen_legal scan. Only for moves recorded by the engine itself — an
 * illegal fast push corrupts the key history silently. */
int  go_push_fast(GoGame *g, uint16_t move);

/* state_termination: 0 playing, 1 over. On over, *score (black margin)
 * and *ownership (361 floats, black view {0,0.5,1}) are filled; either
 * pointer may be NULL. */
int  go_termination(const GoGame *g, float *score, float *ownership);

/* Feature tensor of the current position: GO_FFI_NN_TOTAL floats,
 * token-major, first b*b tokens active (layout: go_nn.h). Returns 0. */
int  go_encode(const GoGame *g, float *out);

#ifdef __cplusplus
}
#endif

#endif /* GO_FFI_H */
