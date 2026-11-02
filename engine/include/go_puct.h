/*
 * go_puct.h — RPO MCTS + multi-game batching (ported from the chess
 * engine's chess_puct.h; the search is game-agnostic — only the seams
 * changed: go.h rules, go_nn.h encoder, 2-way win/loss value, no wasm).
 *
 * Single-thread, single-game search (the mux below walks its games on a
 * small pool of tasks). One PuctGame owns a State + one mmap'd node
 * block (64 MB default) that is bump-allocated and packed into a fresh
 * block whenever it fills.
 *
 * Node layout: W (value_sum) is stored in the PARENT-mover's view, so
 * selection is sign-free and backup flips the sign once per ply. A
 * node's children are a contiguous edge array created eagerly at
 * expansion; each edge becomes a node on first selection (see
 * PuctEdge). The tree persists across moves: puct_play makes the
 * played child the new root and keeps its subtree.
 *
 * Eval seam: requests are built entirely C-side (planes via nn_encode,
 * mask via gen_legal + go_policy_slot); the callback only runs the
 * forward. Until puct_set_eval installs one, a built-in dummy runs:
 * uniform priors, value 0.
 */
#ifndef GO_PUCT_H
#define GO_PUCT_H

#include <stdbool.h>
#include <stddef.h>

#include "go.h"
#include "go_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PUCT_POLICY_SIZE (GO_MAX_POINTS + 1)  /* board points + pass slot   */
#define PUCT_DEFAULT_BLOCK (64u << 20)   /* 64 MB node block per game      */
#define PUCT_BANKS 2                 /* landing-zone banks (double buffer) */

/* Children are ALWAYS a contiguous PuctEdge array (byte offset
 * first_child in the game's block). An edge is a placeholder until a
 * walker first selects it, at which point it is materialized: one
 * PuctNode is allocated and the edge's child handle set. Nodes never
 * move or merge after materialization (only compaction relocates
 * them). Unvisited edges carry no stats — pi-bar falls back to the
 * parent's value for them, as before. */
typedef struct PuctEdge {    /* 8 bytes */
    uint16_t move;
    uint16_t prior_q;        /* P × 65535 (fixed point) */
    uint32_t child;          /* byte offset of the materialized node; 0 = never visited */
} PuctEdge;

typedef struct PuctNode {    /* 20 bytes; u32 handles only (pack-safe) */
    uint32_t parent;         /* byte offset; 0 = none (root's parent) */
    uint32_t first_child;    /* byte offset of the PuctEdge array; 0 = childless */
    uint32_t visits;         /* N */
    float    value_sum;      /* W, parent-mover's view */
    uint16_t n_children;     /* number of edges in the child array */
    uint8_t  flags;
    uint8_t  pad;
} PuctNode;

_Static_assert(sizeof(PuctNode) == 20 && sizeof(PuctEdge) == 8, "layout");

#define PUCT_TERMINAL 1
#define PUCT_PENDING 2           /* expand-at-park: children exist with
                                    uniform placeholder priors, closed for
                                    descent until the row delivers (walkers
                                    co-sleep; the real softmax fills in then) */
#define PUCT_OUTC_WIN  4         /* terminal outcome, mover's view (both */
#define PUCT_OUTC_LOSS 8         /* clear = draw; see set_terminal)     */

typedef struct PuctGame PuctGame;

/* planes: bank-relative rows, n x NN_TOTAL. masks/logp: n x
 * PUCT_POLICY_SIZE (1 = legal); only the first b*b+1 slots of a row are
 * active. wdl: n x 2 out (mover's view, softmax [pW, pL]). score: n x 1
 * out, the score head's mean margin in the mover's view. bank = the
 * batch's landing-zone bank. */
typedef enum {
    PUCT_EVAL_PUSH,   /* H2D + launch compute; async — must not block on
                         the GPU (the io thread keeps serving the other
                         bank so computes run back to back) */
    PUCT_EVAL_PULL,   /* enqueue D2H for the bank's results; async */
    PUCT_EVAL_WAIT    /* block until the bank's results are readable in
                         logp/wdl (the only GPU-blocking op) */
} PuctEvalOp;

/* Synchronous evaluators (dummy, MPS, scripted tests) do the whole
 * forward at PUSH and treat PULL/WAIT as no-ops. */
typedef void (*PuctEvalFn)(void *ctx, PuctEvalOp op, int bank, int n,
                           const float *planes, const uint8_t *masks,
                           float *logp, float *wdl, float *score);

/* block_bytes = 0 selects PUCT_DEFAULT_BLOCK. The game starts at the
 * empty b x b board with komi and the ply wall maxply. Call go_init()
 * once per process first. NULL on OOM. */
PuctGame *puct_game_new(size_t block_bytes, int b, float komi, int maxply);
void      puct_game_free(PuctGame *g);
void      puct_set_eval(PuctEvalFn fn, void *ctx);   /* the one evaluator */

/* Suspend: puct_think / puct_mux_step treat the game as idle (returning
 * done) until resumed — for games waiting on something outside the
 * search. Never suspend a game with rows in flight. */
void      puct_set_suspended(PuctGame *g, bool on);

/* Recycle in place: fresh empty-board game with the SAME b/komi/maxply
 * — state, tree and walkers reset; the mmap'd block and stats are kept.
 * This is how the mux starts a new game without freeing memory. */
void      puct_game_reset(PuctGame *g);

/* The boardsize/komi/maxply override (GTP boardsize): replants the game
 * at a fresh empty board under the new config. Call with no walker in
 * flight. 0 = ok, -1 on bad args. */
int       puct_game_reconfig(PuctGame *g, int b, float komi, int maxply);

/* ---- the landing zone (global) ---------------------------------------------
 * Eval rows shared by every game in the process, in TWO banks (double
 * buffer): the walker thread stages into the active bank while the io
 * thread evaluates the other. Row ids are global across banks. Plus the
 * evaluator and its single io thread; a walker claims a row when it
 * parks and remembers its row id, so delivery needs no row→game table. */
void puct_zone_init(int max_rows);   /* rows PER BANK; ONE-TIME process
                                        setup — asserts if the zone exists */
void puct_zone_set_buffers(int bank, float *planes, uint8_t *mask,
                           float *logp, float *wdl,
                           float *score);  /* nn_eval.c seam:
                                             point the bank at caller-owned
                                             pinned tensors (e.g. sized
                                             rows = zone cap) */

/* Weight of the static score term in the search utility: the value
 * backed up is w[0]-w[1] + w * tanh(margin / (2b)), where the margin is
 * the score head's mean at a net leaf and the exact TT score at a
 * terminal. Default 0.10 (KataGo's staticScoreUtilityFactor); 0 restores
 * pure win/loss. Process-global. */
void   puct_set_score_utility(double w);
double puct_score_utility(void);

/* Move → policy slot: the board point, or b*b for pass. */
int  go_policy_slot(const State *s, Move m);

/* Run PUCT simulations from the root until `sims` backups are counted;
 * returns with the game position exactly as it was. W walkers (default
 * 8, see puct_set_walkers) descend the tree concurrently on the calling
 * thread; a walker that reaches a childless leaf expands it in place with
 * uniform placeholder priors, claims an eval row and parks. Only the
 * calling thread touches the tree (in the mux: the one task the driver
 * handed this walk pass). Gives up per-walker (oom counter) if the block
 * cannot be packed further. */
void puct_think(PuctGame *g, int sims);

/* True between a completed think and the next played move — the mux
 * driver's done-guard. */
bool puct_think_done(const PuctGame *g);

/* The game's position is decided (double pass or ply cap): a terminal
 * root thinks instantly — drivers end the game here. */
bool puct_game_over(const PuctGame *g);

/* Number of concurrent walkers / in-flight evals (>= 1; set ONCE, before
 * the first think — default 8 if never called). */
void puct_set_walkers(PuctGame *g, int n);

/* Argmax visits among root children; MOVE_PASS if the root has no
 * children (a terminal root — drivers check puct_game_over first). */
Move puct_best_move(const PuctGame *g);

/* Advance the game by `m` (must be legal). If `m` is a root child the
 * tree persists (subtree kept); otherwise a fresh root is allocated. */
int  puct_play(PuctGame *g, Move m);

uint32_t    puct_oom_count(const PuctGame *g);
uint32_t    puct_compact_count(const PuctGame *g);

/* Per-think leaf-arrival telemetry (reset each fresh think); any output
 * pointer may be NULL. parks = every sim ending at a childless node (eval
 * parks + terminal arrivals); terminal_hits = parks backed up from a
 * terminal; pending_parks = parks that co-slept on another walker's
 * in-flight eval row. Eval rows staged = parks - terminal_hits -
 * pending_parks. */
void puct_park_stats(const PuctGame *g, uint32_t *parks,
                     uint32_t *terminal_hits, uint32_t *pending_parks);

/* Per-think search depth: mean/max root→leaf descent depth (all walker
 * exits; reset by the next fresh think). */
void puct_depth_stats(const PuctGame *g, double *mean, uint32_t *max);

const State *puct_state(const PuctGame *g);

/* Root children in generation order; any output pointer may be NULL.
 * q is W/N in the root mover's view (0 while N = 0). Returns the count. */
int puct_root_children(const PuctGame *g, uint16_t *moves, uint32_t *visits,
                       double *q, float *prior);
int puct_learn_pi(const PuctGame *g, uint16_t *moves, float *pi, int cap);
float puct_cpuct(const PuctGame *g);

/* The root node's own visit count and value sum (parent-mover's view). */
void puct_root_stats(const PuctGame *g, uint32_t *visits, double *value_sum);

/* ---- multi-game multiplexer ----------------------------------------------
 * ONE per process, like the zone. Owns exactly games_max games, ALL
 * allocated up front at the empty board under (b, komi, maxply) —
 * games_max is the memory budget. The fixed game array is handed to the
 * caller ONCE; idle slots are skipped via puct_set_suspended.
 *
 * Every step walks every NON-SUSPENDED game with the same sims budget:
 * their walkers park into the batch, full or stuck banks ship to the io
 * thread asynchronously — batched across every live game — and the step
 * ends fully delivered so the driver can play moves and recycle games
 * between steps. Walk threads are opt-in: PUCT_TASKS=N runs each round
 * on N tasks; with none the driver walks the round itself. The zone is
 * NOT sized here — call puct_zone_init first. 0 = ok; -1 on failure. */
int  puct_mux_new(int games_max, size_t block_bytes, int b, float komi,
                  int maxply, PuctGame **games);
void puct_mux_free(void);
void puct_mux_step(int sims);   /* one fill → eval → deliver */

#ifdef __cplusplus
}
#endif

#endif /* GO_PUCT_H */
