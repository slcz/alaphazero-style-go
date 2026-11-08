/* wasm_glue.c — the web build's net evaluator (compiled only under
 * __EMSCRIPTEN__; the Makefile's wasm target adds it).
 *
 * JS is the evaluator: EM_ASYNC_JS calls suspend the wasm stack under
 * Asyncify while the worker runs onnxruntime-web, so the evaluator is
 * "synchronous" from C's point of view — exactly the contract
 * go_puct.h documents for sync evaluators (everything at PUSH, PULL and
 * WAIT no-ops). This file also provides puct_nn_raw for gtp.c's
 * showstate/analyze, which nn_eval.c owns in native builds.
 */
#ifdef __EMSCRIPTEN__

#include <emscripten.h>
#include <stdint.h>
#include <string.h>

#include "go.h"
#include "go_nn.h"
#include "go_nn_eval.h"
#include "go_puct.h"

/* Batched zone eval: n rows of planes [NN_TOTAL] (token-major, board
 * size readable from token dim GNN_SIZE) + mask bytes; JS writes
 * logp [n, PUCT_POLICY_SIZE] (log-probs), wdl [n, 2] (softmax probs)
 * and score [n] back into the wasm heap. */
EM_ASYNC_JS(void, js_net_run, (int n, int planesP, int maskP,
                               int logpP, int wdlP, int scoreP), {
    return Module.netRun(n, planesP, maskP, logpP, wdlP, scoreP);
});

/* One position, all five heads: JS writes policy [b*b+1] (probs),
 * wdl [2] (probs), own [b*b] (mover view, [0,1]) and score [1]. */
EM_ASYNC_JS(int, js_raw_run, (int b, int planesP, int maskP, int polP,
                              int wdlP, int ownP, int scoreP), {
    return Module.rawRun(b, planesP, maskP, polP, wdlP, ownP, scoreP);
});

static void wasm_eval(void *ctx, PuctEvalOp op, int bank, int n,
                      const float *planes, const uint8_t *masks,
                      float *logp, float *wdl, float *score) {
    (void)ctx;
    (void)bank;
    (void)masks;
    if (op != PUCT_EVAL_PUSH)           /* sync evaluator: PULL/WAIT noop */
        return;
    js_net_run(n, (int)(uintptr_t)planes, (int)(uintptr_t)masks,
               (int)(uintptr_t)logp, (int)(uintptr_t)wdl,
               (int)(uintptr_t)score);
}

void gtp_wasm_attach_net(void) {
    puct_set_eval(wasm_eval, NULL);
}

/* Mirror of nn_eval.c's raw eval, minus CPython: encode the position,
 * run the net once through JS, hand the heads to the caller. */
int puct_nn_raw(const State *s, float *policy, float *wdl, float *own,
                float *score) {
    int b = s->b, n = b * b;
    float planes[NN_TOTAL];
    uint8_t mask[PUCT_POLICY_SIZE];
    MoveList ml;

    nn_encode(s, planes);
    gen_legal(s, &ml);
    memset(mask, 0, sizeof mask);
    for (int i = 0; i < ml.count; i++)
        mask[go_policy_slot(s, ml.moves[i])] = 1;

    float pol[PUCT_POLICY_SIZE], w[2], own_out[GO_MAX_POINTS], sc;
    if (js_raw_run(b, (int)(uintptr_t)planes, (int)(uintptr_t)mask,
                   (int)(uintptr_t)pol, (int)(uintptr_t)w,
                   (int)(uintptr_t)own_out, (int)(uintptr_t)&sc) != 0)
        return -1;
    if (policy)
        memcpy(policy, pol, (size_t)(n + 1) * sizeof(float));
    if (wdl)
        memcpy(wdl, w, 2 * sizeof(float));
    if (own)
        memcpy(own, own_out, (size_t)n * sizeof(float));
    if (score)
        *score = sc;
    return 0;
}

#endif  /* __EMSCRIPTEN__ */
