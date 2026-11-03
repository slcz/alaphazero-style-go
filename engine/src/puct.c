/* puct.c — RPO MCTS + multi-game batching (go_puct.h), ported from the
 * chess engine: pi-bar search (Grill et al. 2020), mmap'd node blocks
 * with compaction, double-banked landing zone + io thread, walkers, and
 * the multi-game mux. Game-specific seams only: go.h rules, go_nn.h
 * encoder, go_policy_slot, TT terminals. No wasm paths — native only. */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include "go_puct.h"
#include "go_rng.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* A walker = one concurrent descent. Its scratch State is copied fresh
 * from the game root at every sim and doubles as the parked leaf's
 * position at delivery (only the side to move is read). idx points at
 * the parked leaf; a parked leaf is expanded on the spot, so no two
 * walkers ever park on one node. */
typedef struct PuctWalker {
    State      st;
    uint32_t   idx;      /* current leaf; remapped by compaction */
    int        row;      /* staged row while parked, -1 while walking */
    bool       dead;
} PuctWalker;

enum { WALKERS_DEFAULT = 8 };   /* walkers per game unless puct_set_walkers */

/* Bank life cycle (guarded by Z.mu): FREE → QUEUED (zone_submit) →
 * IN_QUEUE (pushed) → PULLED (D2H enqueued) → READY (published) → FREE
 * (delivered). Only a FREE bank is staged into. */
#define ZONE_LOCK()   pthread_mutex_lock(&Z.mu)
#define ZONE_UNLOCK() pthread_mutex_unlock(&Z.mu)

#define BANK_FREE 0
#define BANK_QUEUED 1
#define BANK_IN_QUEUE 2
#define BANK_PULLED 3
#define BANK_READY 4

/* Permanent row storage: one mapping, partitioned into the four arrays ×
 * banks, sized once by puct_zone_init (pages materialize on touch);
 * puct_zone_set_buffers can instead point a bank at pinned tensors —
 * this storage is then bypassed, never freed. */
static void *z_mem;        /* also the zone's initialized flag */
static int   z_rows;       /* rows per bank */

/* One row's storage: planes + logp + 2 value floats, then mask bytes. */
#define ZONE_ROW_BYTES \
    (((size_t)NN_TOTAL + PUCT_POLICY_SIZE + 3) * sizeof(float) \
     + PUCT_POLICY_SIZE)

/* The landing zone — exactly one per system, two banks (double buffer):
 * walkers stage rows into the active bank while the io thread evaluates
 * the other. Row ids are global across banks (bank = row / z_rows), so
 * delivery needs no row→game table. */
static struct {
    struct {
        float   *planes;   /* cap × NN_TOTAL, into z_mem or pinned */
        uint8_t *mask;     /* cap × PUCT_POLICY_SIZE */
        float   *logp;     /* cap × PUCT_POLICY_SIZE */
        float   *wdl;      /* cap × 2 */
        float   *score;    /* cap × 1: predicted margin, mover view */
        int      used;     /* rows staged; frozen once the bank leaves FREE */
        int      state;    /* BANK_FREE .. BANK_READY, guarded by Z.mu */
        bool     flush;    /* pull this bank now, not after the peer's push
                              (starved waker) */
    } banks[PUCT_BANKS];
    int     active;       /* bank the walkers stage into (waker only) */
    PuctEvalFn eval;
    void   *eval_ctx;
    pthread_t       io_thread;
    pthread_mutex_t mu;
    pthread_cond_t  io_cv, walker_cv;
    bool            io_quit;
} Z;

struct PuctGame {
    State      state;
    uint8_t    b;             /* the game's config — */
    float      komi;          /* puct_game_reset replants */
    uint16_t   maxply;        /* an empty board with these */
    PuctNode  *nodes;
    size_t     block_bytes;
    size_t     cap;      /* node slots in the block */
    uint32_t   used;
    uint32_t   root;
    float      cpuct;
    float     *noise;      /* MAX_MOVES: cached Dirichlet draw for the root */
    uint64_t   noise_key;  /* poskey the draw belongs to (0 = none) */
    uint64_t   rng;        /* per-game sampling stream, re-keyed at every
                              fresh think */
    uint32_t   oom;
    uint32_t   compactions;
    PuctWalker *walkers;    /* parked idx fields are ground truth;
                               compaction remaps them */
    int        n_walkers;
    bool       think_done; /* completed (awaiting the driver's play) or
                              never-thought; see puct_walk */
    bool       suspended;  /* skip in walks: the game waits on something
                              external (e.g. the opponent's GTP reply) */
    uint8_t    step_state; /* WALK_*: this step's last pass result */
    uint64_t   done_key;   /* position key at completion; 0 = fresh game */
    int        sims_done;
    uint32_t   parks, terminal_hits;
    uint32_t   pending_parks;  /* parks that co-slept on a pending row */
    uint32_t   desc_sum, desc_max, descents;  /* per-think sim depth */
};

/* Remap walkers whose pre-copy idx (snap) was old_idx to new_idx; the
 * snapshot keeps each walker a single hop — matching the live fields
 * can double-remap mid-copy. */
static void sleep_remap(PuctGame *g, const uint32_t *snap,
                        uint32_t old_idx, uint32_t new_idx) {
    for (int i = 0; i < g->n_walkers; i++)
        if (snap[i] == old_idx)
            g->walkers[i].idx = new_idx;
}

static void *map_block(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}
static void unmap_block(void *p, size_t bytes) { (void)bytes; munmap(p, bytes); }

/* PUCT_TREE_STATS=1: shape of the live tree at each compaction —
 * width (fanout, visited children) and height (depth stats plus a
 * per-depth histogram of expanded nodes; the last bucket is depth >=
 * DH, visit-weighted mean depth is over all visited nodes). */
static void tree_stats(const PuctNode *nb, uint32_t root, uint32_t used) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("PUCT_TREE_STATS");
        on = e && *e && strcmp(e, "0");
    }
    if (!on)
        return;
    enum { DH = 48 };
    uint64_t expanded = 0, multi = 0, anyv = 0, nodes_v = 0, nodes_v1 = 0;
    uint64_t kids = 0, kids_v = 0, kids_term = 0;
    uint64_t dsum = 0, vsum = 0, vwt = 0;
    uint32_t dmax = 0;
    uint32_t hist[DH + 1] = {0};
    size_t qcap = used / sizeof(PuctNode) + 2;
    const PuctNode **q = malloc(qcap * sizeof *q);
    uint16_t *qd = malloc(qcap * sizeof *qd);
    if (!q || !qd) {
        free(q);
        free(qd);
        return;
    }
    size_t hd = 0, tl = 0;
    q[tl] = (const PuctNode *)((const char *)nb + root);
    qd[tl++] = 0;
    while (hd < tl) {
        const PuctNode *n = q[hd];
        uint16_t d = qd[hd++];
        if (n->visits) {
            nodes_v++;
            if (n->visits == 1)
                nodes_v1++;
            vsum += (uint64_t)n->visits * d;
            vwt += n->visits;
        }
        if (!n->n_children)
            continue;
        expanded++;
        dsum += d;
        if (d > dmax)
            dmax = d;
        hist[d < DH ? d : DH]++;
        const PuctEdge *E =
            (const PuctEdge *)((const char *)nb + n->first_child);
        uint32_t dv = 0;
        for (uint32_t c = 0; c < n->n_children; c++) {
            kids++;
            if (E[c].child) {
                const PuctNode *ch =
                    (const PuctNode *)((const char *)nb + E[c].child);
                if (ch->visits) {
                    kids_v++;
                    dv++;
                }
                if (ch->flags & PUCT_TERMINAL)
                    kids_term++;
                if (tl < qcap) {
                    q[tl] = ch;
                    qd[tl++] = (uint16_t)(d + 1);
                }
            }
        }
        if (dv >= 1)
            anyv++;
        if (dv >= 2)
            multi++;
    }
    free(q);
    free(qd);
    fprintf(stderr,
            "TREESTATS nodes=%llu expanded=%llu kids=%llu "
            "kidvisits=%llu term=%llu | leaves/expanded=%.1f "
            "visitedkids/expanded=%.2f p(>=1vis)=%.3f p(>=2vis)=%.3f "
            "1visit=%.3f | dmean=%.1f dmax=%u dvmean=%.1f hist=",
            (unsigned long long)nodes_v, (unsigned long long)expanded,
            (unsigned long long)kids, (unsigned long long)kids_v,
            (unsigned long long)kids_term,
            expanded ? (double)(kids - kids_v) / expanded : 0.0,
            expanded ? (double)kids_v / expanded : 0.0,
            expanded ? (double)anyv / expanded : 0.0,
            expanded ? (double)multi / expanded : 0.0,
            nodes_v ? (double)nodes_v1 / nodes_v : 0.0,
            expanded ? (double)dsum / expanded : 0.0, dmax,
            vwt ? (double)vsum / vwt : 0.0);
    for (int i = 0; i <= DH; i++)
        fprintf(stderr, "%s%u", i ? "," : "", hist[i]);
    fputc('\n', stderr);
}

/* The block is a byte arena: nodes (20B) and edge arrays (8B/edge)
 * interleaved; handles are byte offsets, 0 = the sentinel node = none.
 * Allocation may compact, which relocates everything — callers keep
 * their cursor in a walker idx field (remapped by sleep_remap) and
 * re-derive raw pointers after any alloc. */
static PuctNode *node_at(PuctGame *g, uint32_t off) {
    return (PuctNode *)((char *)g->nodes + off);
}
static const PuctNode *node_at_c(const PuctGame *g, uint32_t off) {
    return (const PuctNode *)((const char *)g->nodes + off);
}
static PuctEdge *kids_of(PuctGame *g, const PuctNode *n) {
    return (PuctEdge *)((char *)g->nodes + n->first_child);
}
static const PuctEdge *kids_of_c(const PuctGame *g, const PuctNode *n) {
    return (const PuctEdge *)((const char *)g->nodes + n->first_child);
}

/* Pack the root-reachable subtree into a fresh block (BFS over a
 * malloc'd offset queue), remapping parent/first_child/child handles
 * and walker idx fields; the old block is munmap'd. */
static bool compact(PuctGame *g) {
    char *nb = map_block(g->block_bytes);
    if (!nb)
        return false;
    const char *ob = (const char *)g->nodes;
    /* pre-copy snapshot (see sleep_remap); never malloc(0) */
    uint32_t *snap = malloc((size_t)(g->n_walkers ? g->n_walkers : 1)
                            * sizeof *snap);
    size_t qcap = g->used / sizeof(PuctNode) + 2;
    uint32_t *q = malloc(qcap * sizeof *q);
    if (!snap || !q) {
        free(snap);
        free(q);
        unmap_block(nb, g->block_bytes);
        return false;
    }
    for (int i = 0; i < g->n_walkers; i++)
        snap[i] = g->walkers[i].idx;
    size_t hd = 0, tl = 0;
    uint32_t off = (uint32_t)sizeof(PuctNode);   /* 0 = sentinel */
    memcpy(nb + off, ob + g->root, sizeof(PuctNode));
    sleep_remap(g, snap, g->root, off);
    ((PuctNode *)(nb + off))->parent = 0;
    g->root = off;
    q[tl++] = off;
    off += (uint32_t)sizeof(PuctNode);
    while (hd < tl) {
        PuctNode *cn = (PuctNode *)(nb + q[hd++]);
        uint32_t n = cn->n_children;
        if (n == 0)
            continue;
        uint32_t old_fc = cn->first_child;
        uint32_t es = off;
        memcpy(nb + es, ob + old_fc, (size_t)n * sizeof(PuctEdge));
        cn->first_child = es;
        off += n * (uint32_t)sizeof(PuctEdge);
        PuctEdge *E = (PuctEdge *)(nb + es);
        for (uint32_t i = 0; i < n; i++) {
            if (E[i].child == 0)
                continue;
            memcpy(nb + off, ob + E[i].child, sizeof(PuctNode));
            sleep_remap(g, snap, E[i].child, off);
            ((PuctNode *)(nb + off))->parent = q[hd - 1];
            E[i].child = off;
            q[tl++] = off;
            off += (uint32_t)sizeof(PuctNode);
        }
    }
    free(snap);
    free(q);
    tree_stats((const PuctNode *)nb, g->root, off);
    unmap_block(g->nodes, g->block_bytes);
    g->nodes = (PuctNode *)nb;
    g->used = off;
    g->compactions++;
    return true;
}

static uint32_t alloc_bytes(PuctGame *g, size_t n) {
    if (n > g->cap ||
        ((size_t)g->used + n > g->cap &&
         (!compact(g) || (size_t)g->used + n > g->cap))) {
        g->oom++;
        return 0;
    }
    uint32_t off = g->used;
    memset((char *)g->nodes + off, 0, n);
    g->used += (uint32_t)n;
    return off;
}

/* Terminal WDL from the strict TT score of the leaf position, in the
 * leaf mover's view (score 0 — possible only with integer komi from
 * GTP — counts as a draw value 0). */
static void set_terminal(PuctNode *n, const State *st) {
    float score;
    state_termination(st, &score, NULL);
    n->flags |= PUCT_TERMINAL;
    if (score != 0.0f)
        n->flags |= (st->side == BLACK) == (score > 0.0f)
                        ? PUCT_OUTC_WIN : PUCT_OUTC_LOSS;
}

/* Static score term: the TT margin squeezed into [-1,1] by a tanh scaled
 * with the board width, so "two points" means the same thing to the search
 * on 7x7 and on 19x19. */
static double score_term(double margin, int b) {
    return tanh(margin / (2.0 * (double)b));
}

/* Search utility = lambda * win/loss + (1 - lambda) * score, both in the
 * mover's view and both in [-1,1]. With lambda = 0.5 the search trades a
 * point of expected result against a point of margin; at lambda = 1 it is
 * plain AlphaZero value. */
static double g_util_lambda = 0.5;

void puct_set_utility_lambda(double l) {
    g_util_lambda = l < 0.0 ? 0.0 : l > 1.0 ? 1.0 : l;
}

double puct_utility_lambda(void) {
    return g_util_lambda;
}

static float blend(float wdl, double margin, int b) {
    return (float)(g_util_lambda * (double)wdl
                   + (1.0 - g_util_lambda) * score_term(margin, b));
}

/* Terminal backup value, mover's view: the exact TT margin, blended. */
static float terminal_value(const PuctNode *n, const State *st) {
    float wdl = (n->flags & PUCT_OUTC_WIN) ? 1.0f :
                (n->flags & PUCT_OUTC_LOSS) ? -1.0f : 0.0f;
    float score = 0.0f;
    state_termination(st, &score, NULL);
    double margin = (st->side == BLACK) ? score : -score;
    return blend(wdl, margin, st->b);
}

static double prior_of_q(uint16_t q) {   /* prior_q decode */
    return (double)q / 65535.0;
}

/* ---- landing zone -------------------------------------------------------- */

static void *io_main(void *arg);
static void walker_deliver(PuctGame *g, PuctWalker *w);

/* Default evaluator until puct_set_eval swaps in a real one: uniform
 * logp + balanced value. */
static void dummy_eval(void *ctx, PuctEvalOp op, int bank, int n,
                       const float *planes, const uint8_t *masks,
                       float *logp, float *wdl, float *score) {
    (void)ctx; (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;
    memset(score, 0, (size_t)n * sizeof(float));
    for (int r = 0; r < n; r++) {
        memset(logp + (size_t)r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + (size_t)r * 2;
        w[0] = 0.5f; w[1] = 0.5f;
    }
}

void puct_zone_init(int max_rows) {
    assert(!z_mem);
    if (max_rows < 1)
        max_rows = 1;
    size_t rows = (size_t)max_rows;
    z_mem = map_block((size_t)PUCT_BANKS * rows * ZONE_ROW_BYTES);
    assert(z_mem);
    float *planes = z_mem;
    float *logp = planes + (size_t)PUCT_BANKS * rows * NN_TOTAL;
    float *wdl = logp + (size_t)PUCT_BANKS * rows * PUCT_POLICY_SIZE;
    float *score = wdl + (size_t)PUCT_BANKS * rows * 2;
    uint8_t *mask = (uint8_t *)(score + (size_t)PUCT_BANKS * rows);
    for (int b = 0; b < PUCT_BANKS; b++)
        puct_zone_set_buffers(b,
                              planes + (size_t)b * rows * NN_TOTAL,
                              mask + (size_t)b * rows * PUCT_POLICY_SIZE,
                              logp + (size_t)b * rows * PUCT_POLICY_SIZE,
                              wdl + (size_t)b * rows * 2,
                              score + (size_t)b * rows);
    z_rows = max_rows;
    const char *lenv = getenv("GO_UTIL_LAMBDA");
    if (lenv && *lenv)
        puct_set_utility_lambda(atof(lenv));
    if (!Z.eval)
        Z.eval = dummy_eval;
    pthread_mutex_init(&Z.mu, NULL);
    pthread_cond_init(&Z.io_cv, NULL);
    pthread_cond_init(&Z.walker_cv, NULL);
    assert(pthread_create(&Z.io_thread, NULL, io_main, &Z) == 0);
}

void puct_zone_set_buffers(int bank, float *planes, uint8_t *mask,
                           float *logp, float *wdl, float *score) {
    assert(bank >= 0 && bank < PUCT_BANKS);
    Z.banks[bank].planes = planes;
    Z.banks[bank].mask = mask;
    Z.banks[bank].logp = logp;
    Z.banks[bank].wdl = wdl;
    Z.banks[bank].score = score;
}

void puct_set_eval(PuctEvalFn fn, void *ctx) {
    Z.eval = fn;
    Z.eval_ctx = ctx;
}

/* Work the io thread can act on: a queued or pulled bank, or an
 * in-flight one flagged flush. */
static bool io_pending(void) {
    for (int b = 0; b < PUCT_BANKS; b++) {
        int st = Z.banks[b].state;
        if (st == BANK_QUEUED || st == BANK_PULLED ||
            (st == BANK_IN_QUEUE && Z.banks[b].flush))
            return true;
    }
    return false;
}

static void bank_eval(PuctEvalOp op, int b, int n) {
    assert(Z.eval);
    Z.eval(Z.eval_ctx, op, b, n, Z.banks[b].planes, Z.banks[b].mask,
           Z.banks[b].logp, Z.banks[b].wdl, Z.banks[b].score);
}

/* Advance every drivable bank one phase (push → pull → publish); mutex
 * HELD, released around each evaluator call. Results are delivered by
 * the waker (zone_pump), never here. */
static void io_drive(void) {
    /* 1. push (async) */
    for (int b = 0; b < PUCT_BANKS; b++) {
        if (Z.banks[b].state != BANK_QUEUED)
            continue;
        Z.banks[b].state = BANK_IN_QUEUE;
        int n = Z.banks[b].used;
        ZONE_UNLOCK();
        bank_eval(PUCT_EVAL_PUSH, b, n);
        ZONE_LOCK();
        pthread_cond_broadcast(&Z.walker_cv);
    }
    /* 2. pull — after the peer is pushed too, or flush */
    for (int b = 0; b < PUCT_BANKS; b++) {
        if (Z.banks[b].state != BANK_IN_QUEUE)
            continue;
        if (!Z.banks[b].flush && Z.banks[b ^ 1].state == BANK_FREE)
            continue;
        Z.banks[b].state = BANK_PULLED;
        Z.banks[b].flush = false;
        int n = Z.banks[b].used;
        ZONE_UNLOCK();
        bank_eval(PUCT_EVAL_PULL, b, n);
        ZONE_LOCK();
    }
    /* 3. publish: block on the D2H, in order */
    for (int b = 0; b < PUCT_BANKS; b++) {
        if (Z.banks[b].state != BANK_PULLED)
            continue;
        int n = Z.banks[b].used;
        ZONE_UNLOCK();
        bank_eval(PUCT_EVAL_WAIT, b, n);
        ZONE_LOCK();
        Z.banks[b].state = BANK_READY;
        pthread_cond_broadcast(&Z.walker_cv);
    }
}

/* io thread: wait for work, then io_drive it. */
static void *io_main(void *arg) {
    (void)arg;
    ZONE_LOCK();
    for (;;) {
        while (!Z.io_quit && !io_pending())
            pthread_cond_wait(&Z.io_cv, &Z.mu);
        if (Z.io_quit)
            break;
        io_drive();
    }
    ZONE_UNLOCK();
    return NULL;
}

static bool bank_busy(int b) {
    ZONE_LOCK();
    bool busy = Z.banks[b].state != BANK_FREE;
    ZONE_UNLOCK();
    return busy;
}

/* Submit a bank for evaluation; its rows stay staged for delivery. */
static void zone_submit(int b) {
    ZONE_LOCK();
    assert(Z.banks[b].used > 0 && Z.banks[b].state == BANK_FREE);
    Z.banks[b].state = BANK_QUEUED;
    pthread_cond_signal(&Z.io_cv);
    ZONE_UNLOCK();
}

/* Block until bank b is READY; a starved waker flags flush to pull it
 * without the peer's push. */
static void zone_wait(int b) {
    ZONE_LOCK();
    while (Z.banks[b].state != BANK_READY) {
        if (Z.banks[b].state == BANK_IN_QUEUE && !Z.banks[b].flush) {
            Z.banks[b].flush = true;
            pthread_cond_signal(&Z.io_cv);
        }
        pthread_cond_wait(&Z.walker_cv, &Z.mu);
    }
    ZONE_UNLOCK();
}

/* Deliver bank b's rows into their games; the bank becomes the active
 * one. */
static int deliver_bank(int b, PuctGame **games, int n_games) {
    for (int i = 0; i < n_games; i++) {
        PuctGame *g = games[i];
        for (int k = 0; k < g->n_walkers; k++) {
            PuctWalker *w = &g->walkers[k];
            if (w->row >= 0 && w->row / z_rows == b)
                walker_deliver(g, w);
        }
    }
    ZONE_LOCK();
    assert(Z.banks[b].state == BANK_READY);
    int rows = Z.banks[b].used;
    Z.banks[b].state = BANK_FREE;
    Z.banks[b].used = 0;
    Z.active = b;
    ZONE_UNLOCK();
    return rows;
}

/* ---- game ---------------------------------------------------------------- */

/* Fresh game in place at the CURRENT config: tree and walkers reset; the
 * mmap'd block and the lifetime stats are kept. */
static void game_fresh(PuctGame *g) {
    memset(g->nodes, 0, 2 * sizeof(PuctNode));
    g->used = 2 * (uint32_t)sizeof(PuctNode);   /* sentinel + root */
    g->root = (uint32_t)sizeof(PuctNode);
    g->think_done = true;
    g->done_key = 0;
    for (int i = 0; i < g->n_walkers; i++)
        g->walkers[i].row = -1;
}

PuctGame *puct_game_new(size_t block_bytes, int b, float komi, int maxply) {
    if (block_bytes == 0)
        block_bytes = PUCT_DEFAULT_BLOCK;
    PuctGame *g = calloc(1, sizeof *g);
    if (!g)
        return NULL;
    g->nodes = map_block(block_bytes);
    if (!g->nodes) {
        free(g);
        return NULL;
    }
    g->block_bytes = block_bytes;
    g->cap = block_bytes;            /* byte arena */
    g->cpuct = 1.25f;
    if (b < 2 || b > GO_MAX_B || maxply < 1 || maxply > MAX_PLY ||
        g->cap < 2 * sizeof(PuctNode) + MAX_MOVES * sizeof(PuctEdge)) {
        unmap_block(g->nodes, g->block_bytes);
        free(g);
        return NULL;
    }
    g->b = (uint8_t)b;
    g->komi = komi;
    g->maxply = (uint16_t)maxply;
    go_state_init(&g->state, b, komi, maxply);
    game_fresh(g);   /* walkers stay NULL until set (or defaulted in walk) */
    return g;
}

void puct_game_free(PuctGame *g) {
    if (g) {
        unmap_block(g->nodes, g->block_bytes);
        free(g->walkers);
        free(g);
    }
}

/* Fresh empty-board game in place under the SAME config (see game_fresh). */
void puct_game_reset(PuctGame *g) {
    go_state_init(&g->state, g->b, g->komi, g->maxply);
    game_fresh(g);
}

/* Replant under a new config (GTP boardsize); call with no walker in
 * flight. 0 = ok, -1 on bad args. */
int puct_game_reconfig(PuctGame *g, int b, float komi, int maxply) {
    if (b < 2 || b > GO_MAX_B || maxply < 1 || maxply > MAX_PLY)
        return -1;
    g->b = (uint8_t)b;
    g->komi = komi;
    g->maxply = (uint16_t)maxply;
    go_state_init(&g->state, b, komi, maxply);
    game_fresh(g);
    return 0;
}

void puct_set_walkers(PuctGame *g, int n) {
    if (n < 1)
        n = 1;
    assert(!g->walkers);   /* set once, before the first think */
    g->walkers = calloc((size_t)n, sizeof *g->walkers);
    assert(g->walkers);    /* walker setup is a startup decision */
    for (int i = 0; i < n; i++)
        g->walkers[i].row = -1;   /* row 0 is a valid row id */
    g->n_walkers = n;
}

void puct_set_suspended(PuctGame *g, bool on) { g->suspended = on; }

int go_policy_slot(const State *s, Move m) {
    return m == MOVE_PASS ? s->b * s->b : m;
}

typedef double v4d __attribute__((vector_size(32)));   /* gcc and clang */

static int node_pi_bar(const PuctGame *g, const PuctNode *r,
                       uint16_t *moves, float *pi, int cap);

/* One uniform in [0,1) from a caller-owned splitmix64 stream (the same
 * math as rng_next_stream, which only returns the raw word). */
static double stream_uniform(uint64_t *st) {
    return (double)(rng_next_stream(st) >> 11) * (1.0 / 9007199254740992.0);
}

/* Gamma(alpha) by Marsaglia-Tsang, with the alpha<1 boost. Only used for
 * the root Dirichlet draw, so the per-sample cost is irrelevant. */
static double rng_gamma(uint64_t *st, double alpha) {
    if (alpha < 1.0) {
        double u = stream_uniform(st);
        return rng_gamma(st, alpha + 1.0) * pow(u, 1.0 / alpha);
    }
    double d = alpha - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        /* Box-Muller from two uniforms: no shared state to contend on */
        double u1 = stream_uniform(st), u2 = stream_uniform(st);
        if (u1 < 1e-300)
            u1 = 1e-300;
        double r = sqrt(-2.0 * log(u1));
        double x = r * cos(6.283185307179586 * u2);
        double v = 1.0 + c * x;
        if (v <= 0.0)
            continue;
        v = v * v * v;
        double u = stream_uniform(st);
        if (u < 1.0 - 0.0331 * x * x * x * x)
            return d * v;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
            return d * v;
    }
}

/* Dirichlet noise mixed into the ROOT priors: Symmetric alphaZero-style
 * exploration. eps = 0 disables. Selfplay-only, like the root
 * temperature. */
static double g_noise_eps = 0.0, g_noise_alpha = 0.3;

void puct_set_root_noise(double eps, double alpha) {
    g_noise_eps = eps < 0.0 ? 0.0 : eps > 1.0 ? 1.0 : eps;
    if (alpha > 0.0)
        g_noise_alpha = alpha;
}

double puct_root_noise_eps(void) { return g_noise_eps; }

/* SEARCH (Grill et al. 2020): the descending move is sampled from
 * pi-bar — no deterministic PUCT path, no root noise. Returns the
 * sampled EDGE index into n's child array. */
static uint32_t select_child(PuctGame *g, const PuctNode *n) {
    uint16_t mv[MAX_MOVES];
    float pi[MAX_MOVES];
    if (n->n_children < 2)
        return 0;
    int m = node_pi_bar(g, n, mv, pi, MAX_MOVES);
    if (m <= 1)
        return 0;
    double w[MAX_MOVES];
    for (int i = 0; i < m; i++)
        w[i] = (double)pi[i];
    return (uint32_t)sample_index_stream(&g->rng, w, m, 1.0);
}

/* Fill the parked leaf's real priors: softmax over the row's logp at
 * each child edge's policy slot. */
/* Mix Dirichlet(alpha) noise into a node's stored priors, aZero style:
 * p <- (1-eps) p + eps x. Applied to the ROOT only, once per root, so the
 * exploration is fresh on every ply without touching deeper priors. */
static void noise_priors(PuctGame *g, PuctNode *n) {
    if (g_noise_eps <= 0.0 || n->n_children < 2)
        return;
    PuctEdge *E = kids_of(g, n);
    int nc = (int)n->n_children;
    double x[MAX_MOVES], sum = 0.0;
    for (int i = 0; i < nc; i++) {
        x[i] = rng_gamma(&g->rng, g_noise_alpha);
        sum += x[i];
    }
    if (!(sum > 0.0))
        return;
    double tot = 0.0, p[MAX_MOVES];
    for (int i = 0; i < nc; i++) {
        p[i] = (1.0 - g_noise_eps) * prior_of_q(E[i].prior_q)
             + g_noise_eps * (x[i] / sum);
        tot += p[i];
    }
    if (!(tot > 0.0))
        return;
    for (int i = 0; i < nc; i++)          /* renormalize after the blend */
        E[i].prior_q = (uint16_t)(p[i] / tot * 65535.0 + 0.5);
}

static void fill_priors(PuctGame *g, uint32_t idx, const float *logp,
                        const State *st) {
    PuctNode *n = node_at(g, idx);
    PuctEdge *E = kids_of(g, n);
    int slots[MAX_MOVES];
    double max = -INFINITY, sum = 0.0;
    for (uint32_t i = 0; i < n->n_children; i++) {
        slots[i] = go_policy_slot(st, E[i].move);
        if (logp[slots[i]] > max)
            max = logp[slots[i]];
    }
    for (uint32_t i = 0; i < n->n_children; i++)
        sum += exp((double)logp[slots[i]] - max);
    for (uint32_t i = 0; i < n->n_children; i++)
        E[i].prior_q =
            (uint16_t)(exp((double)logp[slots[i]] - max) / sum * 65535.0
                       + 0.5);
    n->flags &= ~(uint8_t)PUCT_PENDING;
    if (idx == g->root)                  /* fresh root: explore it */
        noise_priors(g, n);
}

/* Backup: W and one visit land on every path node; nothing is counted
 * during the descent. */
static void backup(PuctGame *g, uint32_t leaf, float v) {
    double vs = -(double)v;
    for (uint32_t i = leaf; i != 0; ) {
        PuctNode *n = node_at(g, i);
        n->value_sum += vs;
        n->visits++;
        vs = -vs;
        i = n->parent;
    }
}

/* One descent attempt: pi-bar sampling from the root to a terminal
 * backup or a park (leaf's edge array built on the spot, eval row
 * staged). The first selection of an edge materializes its node (one
 * alloc, in place — a pack inside relocates the tree, so the cursor
 * lives in w->idx, which sleep_remap keeps current). Returns false
 * only when the active bank is full. */
static bool walker_step(PuctGame *g, PuctWalker *w) {
    w->st = g->state;
    w->idx = g->root;
    PuctNode *n = node_at(g, w->idx);
    while (n->n_children > 0 && !(n->flags & PUCT_PENDING) &&
           w->st.ply < w->st.maxply) {
        PuctEdge *E = kids_of(g, n);
        uint32_t e = select_child(g, n);
        if (E[e].child == 0) {
            uint32_t c = alloc_bytes(g, sizeof(PuctNode));
            if (c == 0) {
                w->dead = true;
                return true;
            }
            n = node_at(g, w->idx);   /* a pack may have relocated it */
            E = kids_of(g, n);
            node_at(g, c)->parent = w->idx;
            E[e].child = c;
        }
        make_move(&w->st, E[e].move);
        w->idx = E[e].child;
        n = node_at(g, w->idx);
    }
    {   /* every exit below ends a descent of this depth */
        uint32_t d = w->st.ply - g->state.ply;
        g->desc_sum += d;
        g->descents++;
        if (d > g->desc_max)
            g->desc_max = d;
    }
    if (n->flags & PUCT_PENDING) {
        /* expanded but unevaluated: co-sleep on its in-flight row */
        w->row = -1;
        for (int i = 0; i < g->n_walkers; i++)
            if (g->walkers[i].row >= 0 && g->walkers[i].idx == w->idx) {
                w->row = g->walkers[i].row;
                break;
            }
        assert(w->row >= 0);
        g->parks++;
        g->pending_parks++;
        return true;
    }
    if (n->n_children > 0) {
        /* ply wall at an evaluated node: no value will ever arrive */
        w->dead = true;
        return true;
    }
    if (n->flags & PUCT_TERMINAL ||
        state_termination(&w->st, NULL, NULL) != TERM_PLAYING) {
        if (!(n->flags & PUCT_TERMINAL))
            set_terminal(n, &w->st);
        backup(g, w->idx, terminal_value(n, &w->st));
        g->parks++;
        g->terminal_hits++;
        g->sims_done++;
        return true;
    }
    /* park: reserve a row, build the edge array here, stage into it. */
    ZONE_LOCK();
    int b = Z.active;
    if (Z.banks[b].used >= z_rows) {
        ZONE_UNLOCK();
        return false;
    }
    int off = Z.banks[b].used++;
    ZONE_UNLOCK();
    MoveList ml;
    gen_legal(&w->st, &ml);
    uint32_t es =
        alloc_bytes(g, (size_t)ml.count * sizeof(PuctEdge));
    if (es == 0) {
        w->dead = true;
        return true;
    }
    PuctNode *pn = node_at(g, w->idx);
    PuctEdge *E = (PuctEdge *)((char *)g->nodes + es);
    uint16_t uq = (uint16_t)(65535.0 / ml.count + 0.5);
    for (int i = 0; i < ml.count; i++) {
        E[i].move = ml.moves[i];
        E[i].prior_q = uq;      /* child stays 0 (zeroed alloc) */
    }
    pn->flags |= PUCT_PENDING;
    pn->first_child = es;
    pn->n_children = (uint16_t)ml.count;
    w->row = b * z_rows + off;
    nn_encode(&w->st, Z.banks[b].planes + (size_t)off * NN_TOTAL);
    uint8_t *m = Z.banks[b].mask + (size_t)off * PUCT_POLICY_SIZE;
    memset(m, 0, PUCT_POLICY_SIZE);
    for (int i = 0; i < ml.count; i++)
        m[go_policy_slot(&w->st, ml.moves[i])] = 1;
    g->parks++;
    return true;
}

/* Resume a parked walker: fill priors, back the value up; the root's
 * row fills priors only. */
static void walker_deliver(PuctGame *g, PuctWalker *w) {
    int b = w->row / z_rows;
    int off = w->row - b * z_rows;
    assert(node_at(g, w->idx)->n_children > 0);
    /* first delivery only: co-sleepers share the row */
    if (node_at(g, w->idx)->flags & PUCT_PENDING)
        fill_priors(g, w->idx,
                    Z.banks[b].logp + (size_t)off * PUCT_POLICY_SIZE, &w->st);
    if (w->idx != g->root) {
        const float *wd = Z.banks[b].wdl + (size_t)off * 2;
        backup(g, w->idx,
               blend(wd[0] - wd[1], (double)Z.banks[b].score[off], g->b));
        g->sims_done++;
    }
    w->row = -1;
}

/* Why a walk pass stopped: DONE (think complete), STUCK (all walkers
 * parked — only a delivery unblocks), FULL (active bank out of rows). */
enum { WALK_STUCK = 0, WALK_DONE = 1, WALK_FULL = 2 };

static int puct_walk(PuctGame *g, int sims) {
    if (g->suspended)
        return WALK_DONE;
    if (g->think_done) {
        /* done-guard: a completed think stays complete until the driver
         * plays a move (puct_play clears done_key) */
        if (g->done_key == state_poskey(&g->state))
            return WALK_DONE;
        g->think_done = false;
        g->sims_done = 0;
        g->parks = g->terminal_hits = g->pending_parks = 0;
        g->desc_sum = g->desc_max = g->descents = 0;
        g->rng = rng_next();
        for (int i = 0; i < g->n_walkers; i++) {
            g->walkers[i].dead = false;
            g->walkers[i].row = -1;
        }
    }
    if (node_at(g, g->root)->flags & PUCT_TERMINAL) {
        g->think_done = true;
        g->done_key = state_poskey(&g->state);
        return WALK_DONE;
    }
    if (!g->walkers)
        puct_set_walkers(g, WALKERS_DEFAULT);
    assert(z_mem);
    for (;;) {
        bool any_live = false, any_parked = false, all_parked = true;
        for (int i = 0; i < g->n_walkers; i++) {
            const PuctWalker *w = &g->walkers[i];
            if (w->dead)
                continue;
            any_live = true;
            if (w->row >= 0) any_parked = true; else all_parked = false;
        }
        if (!any_live || (g->sims_done >= sims && !any_parked)) {
            g->think_done = true;
            g->done_key = state_poskey(&g->state);
            return WALK_DONE;
        }
        if (all_parked || g->sims_done >= sims)
            return WALK_STUCK;
        for (int i = 0; i < g->n_walkers; i++) {
            PuctWalker *w = &g->walkers[i];
            if (w->dead || w->row >= 0)
                continue;
            if (!walker_step(g, w))
                return WALK_FULL;
        }
    }
}

/* Ship the active bank (FREE on entry), or wait out the in-flight peer
 * and deliver its rows. */
static int zone_pump(PuctGame **games, int n_games) {
    if (Z.banks[Z.active].used > 0) {
        int ship = Z.active;
        zone_submit(ship);
        ZONE_LOCK();
        Z.active = ship ^ 1;
        ZONE_UNLOCK();
        if (bank_busy(Z.active)) {
            zone_wait(Z.active);
            return deliver_bank(Z.active, games, n_games);
        }
        return 0;
    }
    assert(bank_busy(Z.active ^ 1));
    zone_wait(Z.active ^ 1);
    return deliver_bank(Z.active ^ 1, games, n_games);
}

bool puct_think_done(const PuctGame *g) { return g->think_done; }

/* True when the position is decided (double pass / ply cap). */
bool puct_game_over(const PuctGame *g) {
    return state_termination(&g->state, NULL, NULL) != TERM_PLAYING;
}

void puct_think(PuctGame *g, int sims) {
    uint64_t key0 = state_poskey(&g->state);
    g->think_done = true;
    g->done_key = 0;
    while (puct_walk(g, sims) != WALK_DONE) {
        PuctGame *gg = g;
        zone_pump(&gg, 1);
    }
    assert(state_poskey(&g->state) == key0);
}

/* Play m (must be legal): a materialized root child keeps its subtree;
 * any other legal move gets a fresh root. */
int puct_play(PuctGame *g, Move m) {
    PuctNode *r = node_at(g, g->root);
    PuctEdge *E = kids_of(g, r);
    uint32_t nr = 0;
    for (uint32_t i = 0; i < r->n_children; i++)
        if (E[i].move == m) {
            nr = E[i].child;
            break;
        }
    if (nr == 0)
        nr = alloc_bytes(g, sizeof(PuctNode));
    if (nr == 0)
        return -1;
    make_move(&g->state, m);
    g->root = nr;
    node_at(g, nr)->parent = 0;
    g->done_key = 0;
    noise_priors(g, node_at(g, nr));     /* the new root is a new position */
    return 0;
}

Move puct_best_move(const PuctGame *g) {
    const PuctNode *r = node_at_c(g, g->root);
    if (r->n_children == 0)
        return MOVE_PASS;
    const PuctEdge *E = kids_of_c(g, r);
    uint32_t best = 0, bv = 0;
    for (uint32_t i = 0; i < r->n_children; i++) {
        const PuctNode *c =
            E[i].child ? node_at_c(g, E[i].child) : NULL;
        if (c && c->visits > bv) {
            bv = c->visits;
            best = i;
        }
    }
    return E[best].move;
}

uint32_t puct_oom_count(const PuctGame *g) { return g->oom; }

uint32_t puct_compact_count(const PuctGame *g) { return g->compactions; }

/* Per-think leaf-arrival telemetry (reset by the next fresh think). */
void puct_park_stats(const PuctGame *g, uint32_t *parks,
                     uint32_t *terminal_hits, uint32_t *pending_parks) {
    if (parks)
        *parks = g->parks;
    if (terminal_hits)
        *terminal_hits = g->terminal_hits;
    if (pending_parks)
        *pending_parks = g->pending_parks;
}

/* Per-think root→leaf descent depth (all walker exits: eval parks,
 * co-sleeps, terminals, ply walls). mean of 0 = no descent yet. */
void puct_depth_stats(const PuctGame *g, double *mean, uint32_t *max) {
    if (mean)
        *mean = g->descents ? (double)g->desc_sum / g->descents : 0.0;
    if (max)
        *max = g->desc_max;
}

const State *puct_state(const PuctGame *g) { return &g->state; }

int puct_root_children(const PuctGame *g, uint16_t *moves, uint32_t *visits,
                       double *q, float *prior) {
    const PuctNode *r = node_at_c(g, g->root);
    const PuctEdge *E = kids_of_c(g, r);
    for (uint32_t i = 0; i < r->n_children; i++) {
        const PuctNode *c =
            E[i].child ? node_at_c(g, E[i].child) : NULL;
        if (moves)
            moves[i] = E[i].move;
        if (visits)
            visits[i] = c ? c->visits : 0;
        if (q)
            q[i] = (c && c->visits) ? (double)c->value_sum / c->visits
                                    : 0.0;
        if (prior)
            prior[i] = (float)prior_of_q(E[i].prior_q);
    }
    return (int)r->n_children;
}

void puct_root_stats(const PuctGame *g, uint32_t *visits, double *value_sum) {
    const PuctNode *r = node_at_c(g, g->root);
    if (visits)
        *visits = r->visits;
    if (value_sum)
        *value_sum = r->value_sum;
}

/* ---- multi-game multiplexer ----------------------------------------------
 * ONE mux per process, like the zone. mux_games[0..mux_games_max) is
 * allocated up front (games_max IS the memory budget) and handed to the
 * caller once; idle slots are idled via puct_set_suspended. puct_mux_step
 * runs in rounds: give every eligible game one puct_walk pass, pump,
 * repeat until every live think is complete (ending fully delivered).
 *
 * Locking: only Z.mu and the pool's handshake mutex; the walk path is
 * lock-free — games are claimed off an atomic scan cursor (each game one
 * pass per round ⇒ one writer of its step_state byte), ordered against
 * the driver's resets by the round barrier. Walk tasks are opt-in
 * (PUCT_TASKS; none by default — the driver pulls inline). */

static PuctGame **mux_games;
static int        mux_games_max;

static atomic_int  mux_scan;      /* round cursor: one offset per claim */
static atomic_bool mux_full;      /* a pass found the active bank full */

/* the walk-task pool — thread plumbing only */
static struct {
    pthread_t       *tids;
    int              n_tasks;     /* 0 = inline */
    int              quit;
    int              sims;        /* the current step's budget */
    int              epoch;       /* bumps per round: tasks walk when
                                     behind; active/epoch under mu */
    int              active;      /* tasks still inside the round */
    pthread_mutex_t  mu;          /* the condvar's companion, nothing else */
    pthread_cond_t   cv;
} P;

/* The last round this task ran: sleep until P.epoch moves past it. */
static _Thread_local int t_epoch;

/* One task's round (and, with no tasks, the driver's): claim scan
 * offsets until dry, giving each eligible game one puct_walk pass. */
static void mux_pull(void) {
    for (;;) {
        int off = atomic_fetch_add_explicit(&mux_scan, 1,
                                            memory_order_relaxed);
        if (off >= mux_games_max ||
            atomic_load_explicit(&mux_full, memory_order_relaxed))
            return;
        PuctGame *g = mux_games[off];
        if (g->suspended || g->step_state == WALK_DONE)
            continue;
        if ((g->step_state = puct_walk(g, P.sims)) == WALK_FULL) {
            atomic_store_explicit(&mux_full, true,
                                  memory_order_relaxed);
            return;
        }
    }
}

static void *mux_task_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&P.mu);
    for (;;) {
        while (!P.quit && P.epoch == t_epoch)
            pthread_cond_wait(&P.cv, &P.mu);
        if (P.quit)
            break;
        t_epoch = P.epoch;
        pthread_mutex_unlock(&P.mu);
        mux_pull();
        pthread_mutex_lock(&P.mu);
        if (--P.active == 0)
            pthread_cond_broadcast(&P.cv);  /* round drained */
    }
    pthread_mutex_unlock(&P.mu);
    return NULL;
}

/* The game array is handed to the caller once; size the zone first —
 * the mux does not. 0 = ok, -1 = a game failed to allocate. */
int puct_mux_new(int games_max, size_t block_bytes, int b, float komi,
                 int maxply, PuctGame **games) {
    if (games_max < 1)
        games_max = 1;
    mux_games = calloc((size_t)games_max, sizeof *mux_games);
    assert(mux_games);           /* startup allocation */
    mux_games_max = games_max;
    int n = 0;
    const char *env = getenv("PUCT_TASKS");
    if (env && *env) {
        int v = atoi(env);
        if (v >= 1)
            n = v;
    }
    P.tids = calloc((size_t)(n > 0 ? n : 1), sizeof *P.tids);
    assert(P.tids);
    pthread_mutex_init(&P.mu, NULL);
    pthread_cond_init(&P.cv, NULL);
    for (int i = 0; i < games_max; i++) {
        mux_games[i] = puct_game_new(block_bytes, b, komi, maxply);
        if (!mux_games[i]) {
            puct_mux_free();
            return -1;
        }
        games[i] = mux_games[i];
    }
    for (int i = 0; i < n; i++)
        assert(pthread_create(&P.tids[i], NULL, mux_task_main,
                              NULL) == 0);
    P.n_tasks = n;
    return 0;
}

void puct_mux_free(void) {
    pthread_mutex_lock(&P.mu);
    P.quit = 1;
    pthread_cond_broadcast(&P.cv);
    pthread_mutex_unlock(&P.mu);
    for (int i = 0; i < P.n_tasks; i++)
        pthread_join(P.tids[i], NULL);
    P.n_tasks = 0;
    P.quit = 0;
    P.epoch = 0;    /* fresh tasks (t_epoch 0) must sleep until a step */
    pthread_mutex_destroy(&P.mu);
    pthread_cond_destroy(&P.cv);
    for (int i = 0; i < mux_games_max; i++)
        puct_game_free(mux_games[i]);
    free(mux_games);
    mux_games = NULL;
    mux_games_max = 0;
    free(P.tids);
    P.tids = NULL;
}

void puct_mux_step(int sims) {
    assert(z_mem);
    for (int i = 0; i < mux_games_max; i++)
        mux_games[i]->step_state = WALK_STUCK;
    P.sims = sims;
    for (;;) {
        atomic_store_explicit(&mux_scan, 0, memory_order_relaxed);
        atomic_store_explicit(&mux_full, false, memory_order_relaxed);
        if (P.n_tasks > 0) {
            pthread_mutex_lock(&P.mu);
            P.active = P.n_tasks;
            P.epoch++;
            pthread_cond_broadcast(&P.cv);
            while (P.active > 0)
                pthread_cond_wait(&P.cv, &P.mu);
            pthread_mutex_unlock(&P.mu);
        } else {
            mux_pull();
        }
        bool all_done = true;
        for (int i = 0; i < mux_games_max; i++)
            if (mux_games[i]->step_state != WALK_DONE &&
                !mux_games[i]->suspended) {
                all_done = false;
                break;
            }
        if (all_done)
            break;
        zone_pump(mux_games, mux_games_max);
    }
    for (int b = 0; b < PUCT_BANKS; b++)
        assert(Z.banks[b].used == 0 && !bank_busy(b));
}

float puct_cpuct(const PuctGame *g) { return g->cpuct; }

/* pi-bar (Grill et al. 2020): the exact solution of
 * argmax_y [q^T y - lambda KL[pi_theta, y]], i.e.
 *     pi(a) = lambda*pi_theta(a) / (alpha - q(a)),
 * lambda = cpuct*sqrt(N)/(|A|+N), alpha the zero of sum pi = 1 by Newton
 * from the right. Unvisited children take the node's own value. */
static int node_pi_bar(const PuctGame *g, const PuctNode *r,
                       uint16_t *moves, float *pi, int cap) {
    int n = (int)r->n_children;
    if (n <= 0 || cap <= 0)
        return 0;
    if (n > cap)
        n = cap;
    const PuctEdge *E = kids_of_c(g, r);
    double N = (double)r->visits;
    double qmax = -INFINITY, sum = 0.0;
    double lam = N > 0.0 ? (double)g->cpuct * sqrt(N)
                               / (N + (double)r->n_children) : 0.0;
    if (lam == 0.0) {
        /* nothing visited: the regularized policy is the prior itself */
        double ps = 0.0;
        for (int i = 0; i < n; i++)
            ps += prior_of_q(E[i].prior_q);
        for (int i = 0; i < n; i++) {
            if (moves)
                moves[i] = E[i].move;
            pi[i] = (float)(prior_of_q(E[i].prior_q) / ps);
        }
        return n;
    }
    /* W is parent-view, children's q are mover-view: the fallback is the
     * node's own mean, negated */
    double v_node = N > 0.0 ? -(r->value_sum / N) : 0.0;
    v4d q4[MAX_MOVES / 4 + 1], lp4[MAX_MOVES / 4 + 1];
    double *const q = (double *)q4, *const lp = (double *)lp4;
    int nv = (n + 3) / 4;
    for (int i = 0; i < nv * 4; i++) {
        if (i < n) {
            const PuctNode *c =
                E[i].child ? node_at_c(g, E[i].child) : NULL;
            double qi = (c && c->visits) ? c->value_sum / (double)c->visits
                                         : v_node;
            q[i] = qi;
            lp[i] = lam * prior_of_q(E[i].prior_q);
            if (qi > qmax)
                qmax = qi;
        } else {
            /* pad q from a real child: 0.0 could NaN via 0*inf */
            q[i] = q[n - 1];
            lp[i] = 0.0;
        }
    }
    /* safe start: sum pi <= 1 already at qmax + lam */
    double a = qmax + lam;
    for (int it = 0; it < 64; it++) {
        double f = -1.0, df = 0.0;
        /* Newton on sum lp/(a-q) = 1 */
        v4d av = {a, a, a, a}, fv = {0.0, 0.0, 0.0, 0.0};
        v4d dfv = {0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < nv; k++) {
            v4d rr = 1.0 / (av - q4[k]);
            fv += lp4[k] * rr;
            dfv -= lp4[k] * rr * rr;
        }
        f += fv[0] + fv[1] + fv[2] + fv[3];
        df += dfv[0] + dfv[1] + dfv[2] + dfv[3];
        double step = f / df;
        double next = a - step;
        a = next > qmax ? next : 0.5 * (a + qmax);
        if (step < 0.0 ? -step < 1e-15 : step < 1e-15)
            break;
    }
    for (int i = 0; i < n; i++) {
        if (moves)
            moves[i] = E[i].move;
        pi[i] = (float)(lp[i] / (a - q[i]));
        sum += pi[i];
    }
    for (int i = 0; i < n; i++)
        pi[i] = (float)((double)pi[i] / sum);
    return n;
}

int puct_learn_pi(const PuctGame *g, uint16_t *moves, float *pi, int cap) {
    int n = node_pi_bar(g, node_at_c(g, g->root), moves, pi, cap);
    if (getenv("PUCT_DEBUG_PI")) {          // dump root children once per ply
        static int calls = 0;
        const PuctNode *r = node_at_c(g, g->root);
        const PuctEdge *E = kids_of_c(g, r);
        char v[6];
        int b = puct_state(g)->b;
        fprintf(stderr, "PIBAR call=%d rootvisits=%u children=%u\n",
                calls, r->visits, r->n_children);
        for (int i = 0; i < n; i++) {
            const PuctNode *c =
                E[i].child ? node_at_c(g, E[i].child) : NULL;
            move_to_gtp(E[i].move, b, v);
            fprintf(stderr,
                    "PIBAR call=%d mv=%s prior=%.4f visits=%u "
                    "qsum/visits=%+.4f pi=%.4f\n",
                    calls, E[i].move == MOVE_PASS ? "pass" : v,
                    (float)prior_of_q(E[i].prior_q), c ? c->visits : 0,
                    c && c->visits
                        ? (double)c->value_sum / c->visits : 0.0,
                    pi[i]);
        }
        calls++;
    }
    return n;
}
