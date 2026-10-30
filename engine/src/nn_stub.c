/* nn_stub.c — placeholder puct_attach_nn for the Python-free build
 * (build/selfplay: device "dummy" only). The real evaluator lives in
 * nn_eval.c, linked into the NN binaries (selfplay_nn, gtp, engine_mux). */
#include "go_nn_eval.h"

int puct_attach_nn(const char *ckpt, const char *device, int rows) {
    (void)ckpt; (void)device; (void)rows;
    return -1;
}
