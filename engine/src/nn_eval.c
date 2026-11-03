/* nn_eval.c — embedded-Python evaluator (go_nn_eval.h). Ported from the
 * chess engine: CPython is embedded, the io thread drives the worker's
 * three-phase protocol, and the pinned host tensors are handed to the
 * landing zone by raw pointer. */
#include "go_nn_eval.h"

#include <Python.h>
#include <stdlib.h>
#include <string.h>

#include "go_nn.h"
#include "go_puct.h"

typedef struct {
    PyObject *worker;
} NnCtx;

/* Python is only up once puct_attach_nn ran; the dummy device never gets
 * there, so every entry point must check this before touching Py_*. */
static int nn_ready = 0;

static void nn_eval(void *ctx, PuctEvalOp op, int bank, int n,
                    const float *planes, const uint8_t *masks, float *logp,
                    float *wdl) {
    (void)planes;
    (void)masks;
    (void)logp;
    (void)wdl;
    NnCtx *c = ctx;
    const char *meth = op == PUCT_EVAL_PUSH ? "push"
                     : op == PUCT_EVAL_PULL ? "pull" : "wait_ready";
    PyGILState_STATE st = PyGILState_Ensure();
    PyObject *r = PyObject_CallMethod(c->worker, (char *)meth, "(ii)",
                                      bank, n);
    if (!r)
        PyErr_Print();
    Py_XDECREF(r);
    PyGILState_Release(st);
}

static int python_init(void) {
    static int initialized = 0;
    if (initialized)
        return 0;
    Py_Initialize();
    PyRun_SimpleString(
        "import sys\n"
        "sys.path.insert(0, '" PUCT_REPO_ROOT "')\n"
        "sys.path.insert(0, '" PUCT_PURELIB "')\n");
    PyEval_SaveThread();       /* the interpreter is now up for grabs */
    initialized = 1;
    return 0;
}

/* One evaluator per system: size the landing zone (rows per bank), swap in
 * the worker's pinned host tensors — one set per bank — and set the eval
 * fn. */
int puct_attach_nn(const char *ckpt, const char *device, int rows) {
    python_init();
    puct_zone_init(rows);

    PyGILState_STATE st = PyGILState_Ensure();
    PyObject *mod = PyImport_ImportModule("go_ai.net.nn_worker");
    if (!mod) {
        PyErr_Print();
        PyGILState_Release(st);
        return -1;
    }
    PyObject *res =
        PyObject_CallMethod(mod, "setup", "(ssii)", ckpt, device, rows,
                            PUCT_BANKS);
    if (!res) {
        PyErr_Print();
        Py_DECREF(mod);
        PyGILState_Release(st);
        return -1;
    }
    for (int b = 0; b < PUCT_BANKS; b++) {
        PyObject *pv = PyDict_GetItemString(res, "planes");
        PyObject *mv = PyDict_GetItemString(res, "mask");
        PyObject *lv = PyDict_GetItemString(res, "logp");
        PyObject *wv = PyDict_GetItemString(res, "wdl");
        if (!pv || !mv || !lv || !wv ||
            !PyList_Check(pv) || PyList_Size(pv) != PUCT_BANKS) {
            Py_DECREF(res);
            Py_DECREF(mod);
            PyGILState_Release(st);
            return -1;
        }
        puct_zone_set_buffers(b,
                              (float *)PyLong_AsUnsignedLongLong(
                                  PyList_GetItem(pv, b)),
                              (uint8_t *)PyLong_AsUnsignedLongLong(
                                  PyList_GetItem(mv, b)),
                              (float *)PyLong_AsUnsignedLongLong(
                                  PyList_GetItem(lv, b)),
                              (float *)PyLong_AsUnsignedLongLong(
                                  PyList_GetItem(wv, b)));
    }
    NnCtx *c = malloc(sizeof *c);
    c->worker = mod;
    puct_set_eval(nn_eval, c);
    Py_DECREF(res);
    PyGILState_Release(st);
    nn_ready = 1;
    return 0;
}

/* ---- raw one-position evaluation (GTP analysis) ------------------------- */

static PyObject *raw_fn(void) {
    static PyObject *fn;
    if (fn)
        return fn;
    if (!nn_ready)
        return NULL;
    PyObject *mod = PyImport_ImportModule("go_ai.net.nn_worker");
    if (!mod) {
        PyErr_Print();
        return NULL;
    }
    fn = PyObject_GetAttrString(mod, "raw");
    Py_DECREF(mod);
    if (!fn)
        PyErr_Print();
    return fn;
}

int puct_nn_raw(const State *s, float *policy, float *wdl, float *own,
                float *score) {
    int b = s->b, n = b * b;
    if (!nn_ready)
        return -1;
    float planes[NN_TOTAL];
    uint8_t mask[PUCT_POLICY_SIZE];
    MoveList ml;

    nn_encode(s, planes);
    gen_legal(s, &ml);
    memset(mask, 0, sizeof mask);
    for (int i = 0; i < ml.count; i++)
        mask[go_policy_slot(s, ml.moves[i])] = 1;

    PyGILState_STATE st = PyGILState_Ensure();
    PyObject *fn = raw_fn();
    if (!fn) {
        PyGILState_Release(st);
        return -1;
    }
    PyObject *r = fn ? PyObject_CallFunction(fn, "y#y#i",
                                             (char *)planes,
                                             (Py_ssize_t)sizeof planes,
                                             (char *)mask,
                                             (Py_ssize_t)sizeof mask, b)
                     : NULL;
    int rc = -1;
    if (r && PyTuple_Check(r) && PyTuple_GET_SIZE(r) == 4) {
        if (policy)
            memcpy(policy, PyBytes_AsString(PyTuple_GET_ITEM(r, 0)),
                   (size_t)(n + 1) * sizeof(float));
        if (wdl)
            memcpy(wdl, PyBytes_AsString(PyTuple_GET_ITEM(r, 1)),
                   2 * sizeof(float));
        if (own)
            memcpy(own, PyBytes_AsString(PyTuple_GET_ITEM(r, 2)),
                   (size_t)n * sizeof(float));
        if (score)
            memcpy(score, PyBytes_AsString(PyTuple_GET_ITEM(r, 3)),
                   3 * sizeof(float));
        rc = 0;
    } else if (!r) {
        PyErr_Print();
    }
    Py_XDECREF(r);
    PyGILState_Release(st);
    return rc;
}
