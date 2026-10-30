/* go_match.h — driver-side move picking + wall clock (selfplay, gate,
 * GTP front-end). Ported from the chess engine's chess_match.h. */
#ifndef GO_MATCH_H
#define GO_MATCH_H

#include <stdint.h>

#include "go.h"

/* Seconds since an arbitrary origin (CLOCK_MONOTONIC). */
double now_s(void);

/* Pick a move to PLAY: sampled ∝ `prior` (prior_side) or ∝ visits for the
 * first `sample_plys` plies (temp applied), deterministic argmax after.
 * boost shifts visits by a constant (unused by the current drivers). */
Move match_pick_move(const uint16_t *moves, const uint32_t *visits,
                     const float *prior, int n, int ply, int sample_plys,
                     bool prior_side, double temp, uint32_t boost);

#endif /* GO_MATCH_H */
