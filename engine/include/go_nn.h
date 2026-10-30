/*
 * go_nn.h — input-feature encoder for the Go net (mirrors chess_nn.h).
 *
 * THE ENCODED TENSOR (authoritative spec; the C encoder is the single
 * source of truth — the trainer rebuilds features by replaying games
 * through THIS encoder via cffi, so there is no production Python encoder)
 * =====================================================================
 * NN_TOKENS tokens, one per board point in index order (y*b+x). A board
 * uses the first b*b tokens; the row stride stays NN_TOTAL so the landing
 * zone's pinned buffers are size-static. Token dim layout, token-major:
 *
 *   [0]      own stone   (the side to move's color — Go needs no flip)
 *   [1]      opponent stone
 *   [2, 7)   last 5 move locations, one-hot: dim 2+h marks the point of
 *            the move h+1 plies ago (h = 0 newest). A pass ply leaves the
 *            plane empty (its flag is a scalar below); pre-history empty.
 *   [7, 11)  distance to the left / right / top / bottom edge, / b
 *   [11, 16) which of the previous 5 moves were passes (broadcast)
 *   [16]     komi / 15.0                        (broadcast)
 *   [17]     board size / 19.0                  (broadcast)
 *
 * History lives in State.hist (go.h) — no separate history object.
 */
#ifndef GO_NN_H
#define GO_NN_H

#include "go.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NN_TOKENS    GO_MAX_POINTS         /* max board; b*b active       */
#define NN_TOKEN_DIM 18
#define NN_TOTAL     (NN_TOKENS * NN_TOKEN_DIM)

enum {
    GNN_OWN = 0,        /* +0 own stone plane                              */
    GNN_OPP = 1,        /* +1 opponent stone plane                         */
    GNN_LASTM = 2,      /* +h(0..4) last-move one-hot                      */
    GNN_EDGE = 7,       /* +0..3 left/right/top/bottom distance / b        */
    GNN_PASS = 11,      /* +h(0..4) the h-th previous move was a pass      */
    GNN_KOMI = 16,
    GNN_SIZE = 17
};

/* Encode the current position of `s` into out (caller allocates
 * NN_TOTAL floats). Pure: reads only. Zeroes the b*b active tokens. */
void nn_encode(const State *s, float out[NN_TOTAL]);

#ifdef __cplusplus
}
#endif

#endif /* GO_NN_H */
