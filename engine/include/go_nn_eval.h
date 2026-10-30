/*
 * go_nn_eval.h — embedded-Python NN evaluator seam. nn_eval.c (ported
 * with the go_ai.net.nn_worker module) initializes the landing zone and
 * swaps in the real evaluator; until then nn_stub.c answers -1.
 */
#ifndef GO_NN_EVAL_H
#define GO_NN_EVAL_H

#include "go_puct.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Load `ckpt` on `device`, size the landing zone (rows per bank), point
 * the banks at the worker's pinned host tensors, install the evaluator.
 * The active board size is inferred per batch from the staged masks.
 * 0 = ok, -1 = failure. */
int puct_attach_nn(const char *ckpt, const char *device, int rows);

#ifdef __cplusplus
}
#endif

#endif /* GO_NN_EVAL_H */
