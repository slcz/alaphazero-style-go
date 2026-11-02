/* test_puct.c — gate for the ported RPO search (go_puct.h), adapted from
 * the chess suite: bookkeeping, value-sign convention, prior delivery,
 * learn-pi stationary point, TT-terminal wall, mux parity/pressure/
 * recycle, compaction, OOM give-up. Chess-specific items (mate pins,
 * FEN replant) are replaced by their Go analogs (ply-wall terminals,
 * board reconfig). Run with "bench" / "benchmux" for stub-eval rates, or
 * PUCT_TASKS=N in the environment to drive the mux through walk tasks. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "go.h"
#include "go_nn.h"
#include "go_puct.h"
#include "go_rng.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (cond) {                                                         \
            g_pass++;                                                       \
        } else {                                                            \
            g_fail++;                                                       \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                   \
    } while (0)

/* B 7x7 komi 9.5 maxply 150 — the suite's standard game */
static PuctGame *std_game(size_t block) {
    return puct_game_new(block, 7, 9.5f, 150);
}

static bool near_d(double a, double b, double eps) {
    return fabs(a - b) < eps;
}

/* ---- scripted evaluators ---------------------------------------------- */

typedef struct { int calls; } CountCtx;

static void eval_count(void *ctx, PuctEvalOp op, int bank, int n,
                       const float *planes, const uint8_t *masks,
                       float *logp, float *wdl, float *score) {
    (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;                     /* PULL/WAIT are no-ops for sync evals */
    CountCtx *c = ctx;
    c->calls += n;
    for (int r = 0; r < n; r++) {
        memset(logp + (size_t)r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + (size_t)r * 2;
        w[0] = 0.5f; w[1] = 0.5f;
        score[r] = 0.0f;
    }
}

static void eval_mover_wins(void *ctx, PuctEvalOp op, int bank, int n,
                            const float *planes, const uint8_t *masks,
                            float *logp, float *wdl, float *score) {
    (void)ctx; (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;
    for (int r = 0; r < n; r++) {
        memset(logp + (size_t)r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + (size_t)r * 2;
        w[0] = 1.0f; w[1] = 0.0f;
        score[r] = 0.0f;
    }
}

static void eval_pattern(void *ctx, PuctEvalOp op, int bank, int n,
                         const float *planes, const uint8_t *masks,
                         float *logp, float *wdl, float *score) {
    (void)ctx; (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;
    for (int r = 0; r < n; r++) {
        float *lp = logp + (size_t)r * PUCT_POLICY_SIZE;
        for (int s = 0; s < PUCT_POLICY_SIZE; s++)
            lp[s] = (s % 7 == 0) ? 1.5f : 0.0f;
        float *w = wdl + (size_t)r * 2;
        w[0] = 0.5f; w[1] = 0.5f;
        score[r] = 0.0f;
    }
}

/* ---- 1. policy slot ------------------------------------------------------ */

/* wdl = +0.5 and score mean = +14 for every row, at b = 7: the leaf
 * utility is 0.5 + w*tanh(14/14). */
static void eval_fixed_score(void *ctx, PuctEvalOp op, int bank, int n,
                             const float *planes, const uint8_t *masks,
                             float *logp, float *wdl, float *score) {
    (void)ctx; (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;
    for (int r = 0; r < n; r++) {
        memset(logp + (size_t)r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + (size_t)r * 2;
        w[0] = 0.75f; w[1] = 0.25f;
        score[r] = 14.0f;
    }
}

static void test_leaf_score_utility(void) {
    /* The score head's mean must reach the leaf backup through the eval
     * row: utility = 0.5 + w*tanh(14/(2*7)), seen negated at the root.
     * One playout only, so the child carries just its own evaluation (a
     * deeper tree mixes in sign-flipped backups from the ply below). */
    puct_set_score_utility(0.10);
    PuctGame *g = puct_game_new(0, 7, 9.5f, 400);
    CHECK(g != NULL);
    puct_set_eval(eval_fixed_score, NULL);
    puct_think(g, 1);

    double want = -(0.5 + 0.10 * tanh(14.0 / 14.0));
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, NULL);
    CHECK(n == 50);
    int leaves = 0;             /* children with no subtree of their own */
    for (int i = 0; i < n; i++) {
        if (visits[i] != 1)
            continue;           /* deeper means sign-flipped backups, too */
        leaves++;
        CHECK(near_d(q[i], want, 1e-6));
    }
    CHECK(leaves >= 1);
    puct_game_free(g);
}

static void test_policy_slot(void) {
    State s7, s9;
    go_state_init(&s7, 7, 9.5f, 150);
    go_state_init(&s9, 9, 7.5f, 300);
    CHECK(go_policy_slot(&s7, 0) == 0);
    CHECK(go_policy_slot(&s7, 48) == 48);
    CHECK(go_policy_slot(&s7, MOVE_PASS) == 49);
    CHECK(go_policy_slot(&s9, MOVE_PASS) == 81);
    CHECK(go_policy_slot(&s9, 80) == 80);
}

/* ---- 2. bookkeeping ------------------------------------------------------ */

static void test_bookkeeping(void) {
    PuctGame *g = std_game(0);
    CHECK(g != NULL);
    CountCtx cc = {0};
    puct_set_eval(eval_count, &cc);
    puct_think(g, 300);
    CHECK(cc.calls > 0);

    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    float prior[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, prior);
    CHECK(n == 50);                    /* 49 points + pass on 7x7 */

    double prior_sum = 0, w_children = 0;
    uint32_t n_children_sum = 0;
    for (int i = 0; i < n; i++) {
        prior_sum += prior[i];
        w_children += q[i] * visits[i];
        n_children_sum += visits[i];
    }
    CHECK(near_d(prior_sum, 1.0, n / 65535.0));

    uint32_t root_visits;
    double root_w;
    puct_root_stats(g, &root_visits, &root_w);
    CHECK(root_visits == n_children_sum);
    CHECK(near_d(root_w, -w_children, 1e-9));

    /* pi-bar exists and is dense */
    float pi[MAX_MOVES];
    int m = puct_learn_pi(g, NULL, pi, MAX_MOVES);
    CHECK(m == n);
    double sum = 0;
    for (int i = 0; i < m; i++) {
        CHECK(pi[i] > 0.0f);
        sum += pi[i];
    }
    CHECK(near_d(sum, 1.0, 1e-5));
    puct_game_free(g);
}

/* ---- 3. value sign convention -------------------------------------------- */

static void test_value_sign(void) {
    PuctGame *g = std_game(0);
    puct_set_walkers(g, 1);
    puct_set_eval(eval_mover_wins, NULL);
    puct_think(g, 1);

    /* one sim: the walker first expands the root (its row fills priors
     * only, no sim), then descends into the first root child — uniform
     * placeholder priors, no visits yet — expands it, and its row backs
     * "mover at the leaf wins" up: Q = -1 in the root mover's view. */
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, NULL);
    CHECK(n == 50);
    int hot = 0;
    for (int i = 0; i < n; i++) {
        if (visits[i] > 0) {
            hot++;
            CHECK(visits[i] == 1);
            CHECK(near_d(q[i], -1.0, 1e-9));
        }
    }
    CHECK(hot == 1);
    uint32_t root_visits;
    double root_w;
    puct_root_stats(g, &root_visits, &root_w);
    CHECK(root_visits == 1);
    CHECK(near_d(root_w, 1.0, 1e-9));
    puct_game_free(g);
}

/* ---- 3b. terminal utility includes the static score term --------------- */

static void test_terminal_score_utility(void) {
    /* Same position as test_wall_terminal with KataGo's default weight:
     * a terminal's utility is ±1 plus 0.10 * tanh(margin / (2*b)).
     * pass:  white to move, margin -9.5  -> -1 - 0.10*tanh(9.5/14)
     * point: white to move, margin -39.5 -> -1 - 0.10*tanh(39.5/14)
     * and the root sees both negated (root is black). */
    puct_set_score_utility(0.10);
    PuctGame *g = puct_game_new(0, 7, 9.5f, 1);
    CHECK(g != NULL);
    CountCtx cc = {0};
    puct_set_eval(eval_count, &cc);
    puct_think(g, 200);

    double q_pass = -1.0 - 0.10 * tanh(9.5 / 14.0);
    double q_point = 1.0 + 0.10 * tanh(39.5 / 14.0);
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, NULL);
    CHECK(n == 50);
    for (int i = 0; i < n; i++) {
        if (!visits[i])
            continue;
        double want = moves[i] == MOVE_PASS ? q_pass : q_point;
        CHECK(near_d(q[i], want, 1e-6));
    }
    puct_game_free(g);
    puct_set_score_utility(0.10);              /* the documented default */
}

/* ---- 4. TT terminal wall: ply cap ends the game in-search ---------------- */

static void test_wall_terminal(void) {
    /* maxply 1: every root child lands on the ply wall and terminates.
     * Strict TT: one stone on an empty board reaches the whole empty
     * region, so every point move OWNS the board — 49 − 9.5 = +39.5, a
     * proven win — while pass loses by komi. Both signs proven in one
     * position, and the ONLY eval row is the root expansion.
     *
     * Pure win/loss here (weight 0); test_terminal_score_utility pins the
     * blended values. */
    puct_set_score_utility(0.0);
    PuctGame *g = puct_game_new(0, 7, 9.5f, 1);
    CHECK(g != NULL);
    CountCtx cc = {0};
    puct_set_eval(eval_count, &cc);
    puct_think(g, 300);

    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, NULL);
    CHECK(n == 50);
    uint32_t visited = 0;
    for (int i = 0; i < n; i++) {
        if (visits[i] > 0)
            visited++;
        if (moves[i] == MOVE_PASS) {
            CHECK(visits[i] >= 1);
            CHECK(near_d(q[i], -1.0, 1e-9));   /* pass loses by komi */
        } else {
            if (visits[i] > 0)
                CHECK(near_d(q[i], 1.0, 1e-9));    /* point owns the board */
        }
    }
    CHECK(visited > 40);
    CHECK(cc.calls == 1);              /* the root row only */

    uint32_t parks, terms, pend;
    puct_park_stats(g, &parks, &terms, &pend);
    CHECK(terms > 0);
    CHECK(parks == 1 + terms + pend);
    puct_game_free(g);
}

/* ---- 5. determinism ------------------------------------------------------- */

static void test_determinism(void) {
    PuctGame *g1 = std_game(0), *g2 = std_game(0);
    rng_seed(1234);              /* selection samples: seed each think */
    puct_think(g1, 400);
    rng_seed(1234);
    puct_think(g2, 400);
    uint16_t m1[MAX_MOVES], m2[MAX_MOVES];
    uint32_t v1[MAX_MOVES], v2[MAX_MOVES];
    int n1 = puct_root_children(g1, m1, v1, NULL, NULL);
    int n2 = puct_root_children(g2, m2, v2, NULL, NULL);
    CHECK(n1 == n2 && n1 == 50);
    for (int i = 0; i < n1; i++) {
        CHECK(m1[i] == m2[i]);
        CHECK(v1[i] == v2[i]);
    }
    puct_game_free(g1);
    puct_game_free(g2);
}

/* ---- 6. prior delivery ---------------------------------------------------- */

static void test_prior_pattern(void) {
    PuctGame *g = std_game(0);
    puct_set_eval(eval_pattern, NULL);
    puct_think(g, 1);

    State root;
    go_state_init(&root, 7, 9.5f, 150);
    MoveList ml;
    gen_legal(&root, &ml);
    CHECK(ml.count == 50);

    uint16_t moves[MAX_MOVES];
    float prior[MAX_MOVES];
    int n = puct_root_children(g, moves, NULL, NULL, prior);
    CHECK(n == ml.count);

    double z[MAX_MOVES], sum = 0;
    for (int i = 0; i < n; i++) {
        int slot = go_policy_slot(&root, ml.moves[i]);
        z[i] = (slot % 7 == 0) ? 1.5 : 0.0;
        sum += exp(z[i]);
    }
    for (int i = 0; i < n; i++)
        CHECK(near_d(prior[i], exp(z[i]) / sum, 1.0 / 65535));
    puct_game_free(g);
}

/* ---- 7. slot fuzz: injective over every legal list ------------------------ */

static void test_slot_fuzz(void) {
    srand(7);
    State s;
    go_state_init(&s, 7, 9.5f, 150);
    int positions = 0;
    for (int ply = 0; ply < 150; ply++) {
        if (state_termination(&s, NULL, NULL) != TERM_PLAYING)
            break;
        MoveList ml;
        gen_legal(&s, &ml);
        CHECK(ml.count > 0);

        bool seen[PUCT_POLICY_SIZE] = {false};
        for (int i = 0; i < ml.count; i++) {
            int slot = go_policy_slot(&s, ml.moves[i]);
            CHECK(slot >= 0 && slot < PUCT_POLICY_SIZE);
            CHECK(!seen[slot]);
            seen[slot] = true;
        }
        positions++;
        /* board moves only (pass is last in the list): random passes end
         * the game early by double-pass */
        if (ml.count == 1) {
            make_move(&s, MOVE_PASS);
            break;
        }
        make_move(&s, ml.moves[rand() % (ml.count - 1)]);
    }
    CHECK(positions > 100);
}

/* ---- 8. OOM give-up ------------------------------------------------------- */

static void test_oom(void) {
    PuctGame *g = puct_game_new(sizeof(PuctNode) * 256, 7, 9.5f, 150);
    CHECK(g != NULL);
    puct_think(g, 1000);
    CHECK(puct_oom_count(g) >= 1);
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, NULL, NULL);
    CHECK(n > 0);
    int best = 0;
    for (int i = 1; i < n; i++)
        if (visits[i] > visits[best])
            best = i;
    CHECK(puct_play(g, moves[best]) == 0);
    puct_think(g, 100);            /* the game is still playable */
    CHECK(puct_state(g)->ply == 1);
    puct_game_free(g);
}

/* ---- 9. compaction -------------------------------------------------------- */

static void test_compaction(void) {
    PuctGame *g = puct_game_new(sizeof(PuctNode) * 3000, 7, 9.5f, 150);
    CHECK(g != NULL);
    for (int round = 0; round < 12; round++) {
        puct_think(g, 100);
        if (puct_game_over(g))
            break;
        uint16_t moves[MAX_MOVES];
        uint32_t visits[MAX_MOVES];
        int n = puct_root_children(g, moves, visits, NULL, NULL);
        CHECK(n > 0);
        int best = 0;
        for (int i = 1; i < n; i++)
            if (visits[i] > visits[best])
                best = i;
        CHECK(puct_play(g, moves[best]) == 0);
    }
    /* the block is far smaller than 12 rounds of search: only packing on
     * fill keeps this going without a single OOM */
    CHECK(puct_compact_count(g) > 0);
    CHECK(puct_oom_count(g) == 0);
    puct_game_free(g);
}

/* ---- 10. mux parity: shared zone == single-game think --------------------- */

static void test_mux_parity(void) {
    PuctGame *ref = std_game(0);
    puct_think(ref, 60);
    uint16_t rm[MAX_MOVES];
    int rn = puct_root_children(ref, rm, NULL, NULL, NULL);
    CHECK(rn == 50);

    /* 3 games, batch cap 8×3 = 24: no mid-game zone splits, so each game
     * must reproduce the reference think bit-for-bit */
    PuctGame *games[3];
    CHECK(puct_mux_new(3, 0, 7, 9.5f, 150, games) == 0);
    int done_games = 0;
    for (int step = 0; step < 100 && done_games < 3; step++) {
        puct_mux_step(60);
        done_games = 0;
        for (int i = 0; i < 3; i++) {
            PuctGame *g = games[i];
            if (!puct_think_done(g))
                continue;
            done_games++;
            uint16_t mv[MAX_MOVES];
            int n = puct_root_children(g, mv, NULL, NULL, NULL);
            CHECK(n == rn);
            for (int k = 0; k < n; k++)
                CHECK(mv[k] == rm[k]);   /* children match; counts need not:
                                          * sampling draws from a shared stream
                                          * whose order the mux interleaves */
        }
    }
    CHECK(done_games == 3);
    puct_mux_free();
    puct_game_free(ref);
}

/* ---- 11. zone pressure: batch cap below walkers×games -------------------- */

static void test_mux_pressure(void) {
    PuctGame *games[3];
    CHECK(puct_mux_new(3, 0, 7, 9.5f, 150, games) == 0);
    for (int i = 0; i < 3; i++)
        puct_set_walkers(games[i], 16);  /* 3x16 walkers vs the 24-row
                                            zone: rounds split */
    uint32_t inherited[3] = {0};
    bool expect[3] = {false, false, false};
    int checks = 0;
    uint64_t parks_sum = 0;
    for (int step = 0; step < 500 && checks < 20; step++) {
        puct_mux_step(37);
        for (int i = 0; i < 3; i++) {
            PuctGame *g = games[i];
            if (!puct_think_done(g))
                continue;
            uint32_t parks, terms, root_visits;
            double root_w;
            puct_park_stats(g, &parks, &terms, NULL);
            parks_sum += parks;
            puct_root_stats(g, &root_visits, &root_w);
            if (puct_game_over(g) ||
                puct_root_children(g, NULL, NULL, NULL, NULL) == 0) {
                expect[i] = false;           /* terminal: game over */
                puct_game_reset(g);
                continue;
            }
            /* the played child becomes the root with its visits intact;
             * every park of the next think then backs the root up exactly
             * once, so root visits == inherited + parks */
            if (expect[i]) {
                CHECK(root_visits == parks + inherited[i]);
                checks++;
            }
            uint16_t mv[MAX_MOVES];
            uint32_t vi[MAX_MOVES];
            int n = puct_root_children(g, mv, vi, NULL, NULL);
            uint32_t bv = 0;
            Move best = MOVE_PASS;
            for (int k = 0; k < n; k++)
                if (vi[k] > bv) { bv = vi[k]; best = mv[k]; }
            inherited[i] = bv;
            expect[i] = true;
            CHECK(puct_play(g, best) == 0);
        }
    }
    CHECK(checks >= 20);
    CHECK(parks_sum > 0);                /* rows staged: telemetry via parks */
    puct_mux_free();
}

/* ---- 12. recycling + run quota -------------------------------------------- */

static void test_mux_recycle(void) {
    PuctGame *games[2];
    CHECK(puct_mux_new(2, 0, 7, 9.5f, 150, games) == 0);
    int finished = 0;
    /* run until 4 games complete (each pool game recycled once): the
     * driver plays a move per completed think and adjudicates at 12 ply */
    for (int step = 0; step < 4000 && finished < 4; step++) {
        puct_mux_step(24);
        for (int i = 0; i < 2; i++) {
            PuctGame *g = games[i];
            if (!puct_think_done(g))
                continue;
            if (puct_state(g)->ply >= 12 || puct_game_over(g)) {
                finished++;
                CHECK(puct_state(g)->ply >= 1);
                puct_game_reset(g);
                CHECK(puct_state(g)->ply == 0);   /* fresh empty-board game */
                continue;
            }
            Move mv = puct_best_move(g);
            CHECK(puct_play(g, mv) == 0);
        }
    }
    CHECK(finished == 4);

    /* keep recycling well past the quota: a few thousand more thinks */
    for (int step = 0; step < 400; step++) {
        puct_mux_step(24);
        for (int i = 0; i < 2; i++) {
            PuctGame *g = games[i];
            if (!puct_think_done(g))
                continue;
            if (puct_state(g)->ply >= 12 || puct_game_over(g)) {
                puct_game_reset(g);
                continue;
            }
            CHECK(puct_play(g, puct_best_move(g)) == 0);
        }
    }
    puct_mux_free();
}

/* ---- 13. board reconfig (GTP boardsize) ----------------------------------- */

static void test_reconfig(void) {
    PuctGame *g = std_game(0);
    puct_game_reset(g);
    MoveList ml;
    gen_legal(puct_state(g), &ml);
    CHECK(ml.count == 50);
    CHECK(puct_state(g)->ply == 0);

    CHECK(puct_game_reconfig(g, 9, 7.5f, 300) == 0);
    gen_legal(puct_state(g), &ml);
    CHECK(ml.count == 82);
    CHECK(puct_state(g)->b == 9 && puct_state(g)->komi == 7.5f);
    puct_think(g, 16);                    /* search works at the new root */
    CHECK(puct_root_children(g, NULL, NULL, NULL, NULL) == 82);

    puct_game_reset(g);                   /* reset keeps the new config */
    gen_legal(puct_state(g), &ml);
    CHECK(ml.count == 82);

    CHECK(puct_game_reconfig(g, 20, 7.5f, 300) == -1);   /* too big */
    CHECK(puct_game_reconfig(g, 7, 9.5f, 0) == -1);      /* bad maxply */
    gen_legal(puct_state(g), &ml);        /* rejected reconfig is a no-op */
    CHECK(ml.count == 82);
    puct_game_free(g);
}

/* ---- 14. pass re-think: the done-guard vs an unchanged poskey ------------- */

/* A pass keeps the placement key, so the position after it repeats the
 * prior poskey — exactly the chess knight-shuffle repetition case. The
 * next think must still run (puct_play clears the done-guard), or the
 * driver would silently read out a stale tree. */
static void test_pass_rethink(void) {
    PuctGame *games[1];
    CHECK(puct_mux_new(1, 0, 7, 9.5f, 150, games) == 0);
    PuctGame *g = games[0];

    puct_mux_step(60);
    uint32_t v0;
    puct_root_stats(g, &v0, NULL);
    CHECK(v0 > 0);

    CHECK(puct_play(g, MOVE_PASS) == 0);
    puct_mux_step(60);
    uint32_t v1;
    puct_root_stats(g, &v1, NULL);
    CHECK(v1 > v0);

    CHECK(puct_play(g, 24) == 0);         /* and after a stone */
    puct_mux_step(60);
    uint32_t v2;
    puct_root_stats(g, &v2, NULL);
    CHECK(v2 > 0);
    puct_mux_free();
}

/* ---- 15. learn-pi: stationary point of the regularized objective ---------- */

static void test_learn_pi(void) {
    uint16_t moves[MAX_MOVES], lmoves[MAX_MOVES];
    uint32_t visits[MAX_MOVES], rv;
    double q[MAX_MOVES], wsum;
    float prior[MAX_MOVES], pi[MAX_MOVES];
    for (int sims = 20; sims <= 200; sims *= 10) {
        PuctGame *g = std_game(0);
        CHECK(g != NULL);
        CountCtx cc = {0};
        puct_set_eval(eval_count, &cc);
        puct_think(g, sims);
        int n = puct_root_children(g, moves, visits, q, prior);
        puct_root_stats(g, &rv, &wsum);
        CHECK(rv > 0);
        int m = puct_learn_pi(g, lmoves, pi, MAX_MOVES);
        CHECK(m == n && m > 1);
        double sum = 0.0, res0 = 0.0, v_root = -(wsum / rv);   /* root W is parent-view */
        double lam = (double)puct_cpuct(g) * sqrt((double)rv) / ((double)rv + n);
        for (int i = 0; i < m; i++) {
            CHECK(pi[i] > 0.0f);                       /* dense: every child */
            sum += pi[i];
            double res = (visits[i] ? q[i] : v_root)
                         + lam * (double)prior[i] / (double)pi[i];
            if (i == 0)
                res0 = res;
            else
                CHECK(fabs(res - res0) < 1e-4);
        }
        CHECK(fabs(sum - 1.0) < 1e-5);
        puct_game_free(g);
    }

    /* Sign-sensitive: with an eval that always says "mover at the leaf
     * wins", every child's q is -1 in the root mover's view (visited AND
     * unvisited — the fallback must agree), so pi_bar collapses to the
     * prior. A sign-flipped v_root puts the unvisited tail at +1 and
     * breaks this. */
    PuctGame *g = std_game(0);
    CHECK(g != NULL);
    puct_set_eval(eval_mover_wins, NULL);
    puct_think(g, 8);                  /* fewer sims than legal moves, so the
                                        * unvisited fallback is exercised */
    int n2 = puct_root_children(g, moves, visits, q, prior);
    CHECK(n2 == 50);
    int unseen = 0;
    for (int i = 0; i < n2; i++)
        if (visits[i] == 0)
            unseen++;
    CHECK(unseen > 0);
    int m2 = puct_learn_pi(g, lmoves, pi, MAX_MOVES);
    CHECK(m2 == n2);
    for (int i = 0; i < m2; i++)
        CHECK(fabs((double)pi[i] - 1.0 / m2) < 2e-3);
    puct_game_free(g);
}

/* ---- benches -------------------------------------------------------------- */

static void bench(int walkers) {
    PuctGame *g = std_game(0);
    puct_set_walkers(g, walkers);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    puct_think(g, 30000);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("bench: 30000 sims in %.3fs = %.0f sims/s (walkers=%d oom=%u)\n",
           dt, 30000.0 / dt, walkers, puct_oom_count(g));
    puct_game_free(g);
}

static void bench_mux(int games_max, int sims) {
    PuctGame *games[games_max];
    CHECK(puct_mux_new(games_max, 0, 7, 9.5f, 150, games) == 0);
    uint64_t row_quota = 200000, eval_rows = 0;
    int finished = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (eval_rows < row_quota) {
        puct_mux_step(sims);
        for (int i = 0; i < games_max; i++) {
            PuctGame *g = games[i];
            if (!puct_think_done(g))
                continue;
            uint32_t parks, terms, pend;
            puct_park_stats(g, &parks, &terms, &pend);
            eval_rows += parks - terms - pend;   /* rows actually staged */
            if (puct_state(g)->ply >= 60 || puct_game_over(g)) {
                finished++;
                puct_game_reset(g);
                continue;
            }
            Move mv = puct_best_move(g);
            puct_play(g, mv);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("bench_mux: %llu rows in %.2fs = %.0f rows/s, %d games finished "
           "(games=%d sims=%d)\n",
           (unsigned long long)eval_rows, dt,
           (double)eval_rows / dt, finished,
           games_max, sims);
    puct_mux_free();
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    go_init();
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        puct_zone_init(8);
        int ws[4] = {1, 2, 4, 8};
        for (int i = 0; i < 4; i++)
            bench(ws[i]);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "benchmux") == 0) {
        puct_zone_init(256);
        bench_mux(64, 200);
        bench_mux(256, 200);
        return 0;
    }
    puct_zone_init(24);   /* one zone for the whole suite: 3 games x 8
                             walkers fit a bank without mid-think splits */
    test_policy_slot();
    test_bookkeeping();
    test_value_sign();
    test_wall_terminal();
    test_terminal_score_utility();
    test_leaf_score_utility();
    test_determinism();
    test_prior_pattern();
    test_slot_fuzz();
    test_oom();
    test_compaction();
    test_mux_parity();
    test_mux_pressure();
    test_mux_recycle();
    test_reconfig();
    test_pass_rethink();
    test_learn_pi();
    printf("test_puct: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
