/*
 * selfplay.c — self-play training-data generation (ported from the chess
 * engine; writes GOSHARD1 shards via sp_dump.c).
 *
 *   selfplay <ckpt> <out_dir> <games> [b=7] [komi=9.5] [sims=200]
 *            [pool=1024] [rows=2048] [sample_plys=10] [max_plys=150]
 *            [block_mb=4] [device=mps] [seed=1] [fast_sims=0] [fast_p=0.75]
 *
 * device "dummy" runs the stub evaluator (uniform priors, value 0) — the
 * engine-only test path; no net or checkpoint is needed (pass any ckpt).
 *
 * A pool of games thinks in lockstep on one mux — every live game gets
 * one sims-budget think per puct_mux_step, with the leaf evals of all
 * games batched. The stored target and the played move are the
 * regularized policy pi-bar (Grill et al. 2020, LEARN + ACT): sample it
 * for the first sample_plys plies, argmax after (sample_plys=0 samples
 * every ply). No root Dirichlet noise — the search itself samples pi-bar
 * at every node.
 *
 * Playout cap randomization (KataGo, Wu 2020 §4): with fast_sims > 0,
 * each step draws ONE budget for the whole pool — fast_sims with
 * probability fast_p, else sims. Per-step (not per-move) because the mux
 * is lockstep; each ply independently hits the fast budget at the fast_p
 * marginal while batches stay full-pool. Every ply records its actual
 * budget; the trainer keeps only plies at the slow cap.
 *
 * Game end: double pass or the ply wall (max_plys, set as the game's
 * maxply) — either way the position is TT-scored; the game record carries
 * the score and per-point ownership (the ownership/score training
 * labels), and the shard's π order contract holds: n_pi == the legal
 * count, probs in gen_legal order.
 *
 * Recording: everything comes from the standard readouts — π̄ from
 * puct_learn_pi, the mover's root value from puct_root_stats (-(W/N);
 * the root's W is parent-view). Moves are recorded as u16 points (or
 * MOVE_PASS), never strings. One shard-0.spbin + DONE.json per run; a
 * killed run leaves a truncated last record the reader drops.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "go.h"
#include "go_match.h"
#include "go_nn_eval.h"
#include "go_puct.h"
#include "go_rng.h"
#include "go_sp_dump.h"

/* Compact per-slot game record: parallel arrays up to max_plys, plus a
 * variable-length pool of π̄ probs (4 B per f32, appended in ply order —
 * dumped sequentially as f16). */
typedef struct Slot {
    int      live;
    int      ply;
    uint32_t game_index;
    float    ownership[GO_MAX_POINTS];   /* game-end label */
    /* parallel arrays allocated to max_plys; prob pool grows by
     * doubling (one f32 per legal move, appended in ply order) */
    uint8_t *mover;
    uint16_t *move;
    double  *outlook;
    uint16_t *n_pi;
    uint16_t *ply_sims;
    float    *pi_prob_pool;
    size_t    probs_used, probs_cap;
} Slot;

static Slot   *slots;
static int     max_plys;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "selfplay: oom\n");
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) {
        fprintf(stderr, "selfplay: oom\n");
        exit(1);
    }
    return p;
}

static void slot_arrays_alloc(Slot *s) {
    s->mover = xmalloc((size_t)max_plys * sizeof *s->mover);
    s->move = xmalloc((size_t)max_plys * sizeof *s->move);
    s->outlook = xmalloc((size_t)max_plys * sizeof *s->outlook);
    s->n_pi = xmalloc((size_t)max_plys * sizeof *s->n_pi);
    s->ply_sims = xmalloc((size_t)max_plys * sizeof *s->ply_sims);
    s->probs_cap = 4096;
    s->probs_used = 0;
    s->pi_prob_pool = xmalloc(s->probs_cap * sizeof *s->pi_prob_pool);
}

/* Append one ply's π̄ probs (legal-move order, engine movegen); returns 0
 * ok, -1 when max_plys is exceeded (the caller fails the run — the ply
 * arrays are sized by it). */
static int slot_record_ply(Slot *s, uint8_t mover, uint16_t move,
                           double outlook, uint16_t n_pi, uint16_t ply_sims,
                           const float *pprobs) {
    if (s->ply >= max_plys)
        return -1;
    if (s->probs_used + n_pi > s->probs_cap) {
        while (s->probs_used + n_pi > s->probs_cap)
            s->probs_cap *= 2;
        s->pi_prob_pool = realloc(s->pi_prob_pool,
                                  s->probs_cap * sizeof *s->pi_prob_pool);
        if (!s->pi_prob_pool) {
            fprintf(stderr, "selfplay: oom\n");
            exit(1);
        }
    }
    int p = s->ply;
    s->mover[p] = mover;
    s->move[p] = move;
    s->outlook[p] = outlook;
    s->n_pi[p] = n_pi;
    s->ply_sims[p] = ply_sims;
    memcpy(&s->pi_prob_pool[s->probs_used], pprobs, n_pi * sizeof *pprobs);
    s->probs_used += n_pi;
    s->ply++;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 16) {
        fprintf(stderr,
                "usage: %s <ckpt> <out_dir> <games> [b=7] [komi=9.5]\n"
                "          [sims=200] [pool=1024] [rows=2048] [sample_plys=10]\n"
                "          [max_plys=150] [block_mb=4] [device=mps] [seed=1]\n"
                "          [fast_sims=0] [fast_p=0.75]\n"
                "       device \"dummy\" runs the stub evaluator (no net)\n",
                argv[0]);
        return 2;
    }
    const char *ckpt = argv[1];
    const char *out_dir = argv[2];
    int games = atoi(argv[3]);
    int b = argc > 4 ? atoi(argv[4]) : 7;
    float komi = argc > 5 ? (float)atof(argv[5]) : 9.5f;
    int sims = argc > 6 ? atoi(argv[6]) : 200;
    int pool = argc > 7 ? atoi(argv[7]) : 1024;
    int rows = argc > 8 ? atoi(argv[8]) : 2048;
    int sample_plys = argc > 9 ? atoi(argv[9]) : 10;
    if (sample_plys < 0)
        sample_plys = 0;
    max_plys = argc > 10 ? atoi(argv[10]) : 150;
    int block_mb = argc > 11 ? atoi(argv[11]) : 4;
    const char *device = argc > 12 ? argv[12] : "mps";
    uint64_t seed = argc > 13 ? strtoull(argv[13], NULL, 10) : 1;
    int fast_sims = argc > 14 ? atoi(argv[14]) : 0;
    double fast_p = argc > 15 ? atof(argv[15]) : 0.75;
    if (games < 1 || sims < 1 || pool < 1 || b < 2 || b > GO_MAX_B) {
        fprintf(stderr, "selfplay: bad args\n");
        return 2;
    }
    if (max_plys < 1 || max_plys > MAX_PLY) {
        fprintf(stderr, "selfplay: max_plys must be 1..%d\n", MAX_PLY);
        return 2;
    }
    if (fast_sims < 0 || fast_sims >= sims ||
        (fast_sims > 0 && !(fast_p > 0.0 && fast_p < 1.0))) {
        fprintf(stderr, "selfplay: need 0 <= fast_sims < sims (%d) and, "
                        "when fast_sims > 0, 0 < fast_p < 1\n", sims);
        return 2;
    }

    go_init();
    rng_seed(seed);
    PuctGame **handles = xmalloc((size_t)pool * sizeof *handles);
    if (puct_mux_new(pool, (size_t)block_mb << 20, b, komi, max_plys,
                     handles) != 0) {
        fprintf(stderr, "selfplay: mux_new failed\n");
        return 1;
    }
    /* walkers per game: PUCT_WALKERS, default 8 (same env as engine_mux) */
    const char *wenv = getenv("PUCT_WALKERS");
    int walkers = wenv ? atoi(wenv) : 8;
    for (int i = 0; i < pool; i++)
        puct_set_walkers(handles[i], walkers);
    if (strcmp(device, "dummy") == 0)
        puct_zone_init(rows);        /* keep the built-in dummy evaluator */
    else {
        char nb[16];
        snprintf(nb, sizeof nb, "%d", b);
        setenv("NN_B", nb, 1);       // static-compile hint: nn_worker
    }
    if (strcmp(device, "dummy") != 0 && puct_attach_nn(ckpt, device, rows) != 0) {
        fprintf(stderr, "selfplay: puct_attach_nn failed\n");
        return 1;
    }    slots = xcalloc((size_t)pool, sizeof *slots);   /* live/ply/game_index
                                                       must start at 0 */
    for (int i = 0; i < pool; i++) {
        slot_arrays_alloc(&slots[i]);
        puct_set_suspended(handles[i], true);
    }

    int fd = sp_dump_open_shard(out_dir, 0, seed);
    if (fd < 0) {
        fprintf(stderr, "selfplay: cannot open shard in %s\n", out_dir);
        return 1;
    }

    uint32_t finished = 0, next_game = 0;
    double depth_sum = 0;          /* global search-depth aggregate */
    uint32_t depth_thinks = 0, depth_max = 0;
    double t0 = now_s(), t_report = t0;
    while (finished < (uint32_t)games) {
        /* start (recycle) slots before the step so their first think is
         * this step, not a wasted one at a stale position */
        for (int i = 0; i < pool; i++)
            if (!slots[i].live && next_game < games) {
                Slot *s = &slots[i];
                puct_game_reset(handles[i]);
                puct_set_suspended(handles[i], false);
                s->live = 1;
                s->ply = 0;
                s->game_index = next_game++;
                s->probs_used = 0;
            }
        /* One budget draw per step for the whole pool (PCR; header
         * comment). fast_sims = 0 → uniform sims every step, off. */
        int step_sims = sims;
        if (fast_sims && rng_uniform() < fast_p)
            step_sims = fast_sims;
        puct_mux_step(step_sims);

        for (int i = 0; i < pool; i++) {
            Slot *s = &slots[i];
            if (!s->live)
                continue;
            PuctGame *g = handles[i];
            if (!puct_think_done(g))
                continue;        /* defensive: step ends fully delivered */

            uint32_t root_visits;
            double wsum;
            puct_root_stats(g, &root_visits, &wsum);
            double outlook = root_visits ? -(wsum / root_visits) : 0.0;
            {
                double dmean;
                uint32_t dmax;
                puct_depth_stats(g, &dmean, &dmax);
                depth_sum += dmean;
                depth_thinks++;
                if (dmax > depth_max)
                    depth_max = dmax;
            }

            /* π̄ is both the stored target and the play policy: sample it
             * for the first sample_plys plies, argmax after. puct_learn_pi
             * returns the moves in root-child order == engine movegen
             * order, which is exactly the on-disk π order contract. */
            float pprobs[SP_PI_MAX];
            uint16_t pmoves[SP_PI_MAX];
            int n_pi = puct_learn_pi(g, pmoves, pprobs, SP_PI_MAX);

            int window = sample_plys > 0 ? sample_plys : max_plys;
            Move mv = match_pick_move(pmoves, NULL, pprobs, n_pi, s->ply,
                                      window, true, 1.0, 0);
            if (slot_record_ply(s, (uint8_t)puct_state(g)->side,
                                mv, outlook, (uint16_t)n_pi,
                                (uint16_t)step_sims, pprobs) != 0) {
                fprintf(stderr, "selfplay: ply overflow at game %u\n",
                        s->game_index);
                return 1;
            }
            if (puct_play(g, mv) != 0) {
                fprintf(stderr, "selfplay: puct_play failed\n");
                return 1;
            }

            float score = 0.0f;
            int over = state_termination(puct_state(g), &score, s->ownership)
                       != TERM_PLAYING;
            uint8_t end_kind;
            if (over) {
                /* TT-scored: double pass or the ply wall. */
                end_kind = puct_state(g)->n_pass >= 2
                             ? SP_END_NATURAL : SP_END_ADJUDICATED;
            } else {
                continue;
            }
            SpGameRec gr = {.game_index = s->game_index,
                            .n_plies = (uint32_t)s->ply,
                            .b = (uint8_t)b,
                            .end_kind = end_kind,
                            .n_sims = (uint16_t)sims,
                            .komi = komi,
                            .score = score,
                            .ownership = s->ownership,
                            .plies = NULL};
            if (sp_dump_game_head(fd, &gr) != 0) {
                fprintf(stderr, "selfplay: shard write failed\n");
                return 1;
            }
            size_t off = 0;
            for (int p = 0; p < s->ply; p++) {
                SpPlyRec pr = {.mover = s->mover[p],
                               .move = s->move[p],
                               .n_pi = s->n_pi[p],
                               .sims = s->ply_sims[p],
                               .outlook = (float)s->outlook[p]};
                memcpy(pr.pi_prob, &s->pi_prob_pool[off],
                       s->n_pi[p] * sizeof *pr.pi_prob);
                off += s->n_pi[p];
                if (sp_dump_ply(fd, &pr) != 0) {
                    fprintf(stderr, "selfplay: shard write failed\n");
                    return 1;
                }
            }
            printf("GAME idx=%u plies=%d end=%s score=%+.1f\n",
                   s->game_index, s->ply,
                   end_kind == SP_END_NATURAL ? "natural" : "cap", score);
            fflush(stdout);
            finished++;
            s->live = 0;
            if (next_game >= games)
                puct_set_suspended(g, true);   /* winding down: stop
                                                  thinking at the corpse */
        }
        if (now_s() - t_report >= 60.0) {      /* live block health */
            uint64_t om = 0, cp = 0;
            for (int i = 0; i < pool; i++) {
                om += puct_oom_count(handles[i]);
                cp += puct_compact_count(handles[i]);
            }
            double dt = now_s() - t0;
            printf("STATUS t=%.0fs games=%u/%d rate=%.2f/s oom=%llu "
                   "compact=%llu\n", dt, finished, games,
                   dt > 0 ? finished / dt : 0.0,
                   (unsigned long long)om, (unsigned long long)cp);
            fflush(stdout);
            t_report = now_s();
        }
    }
    close(fd);
    if (sp_dump_write_done(out_dir, 1, seed, finished) != 0) {
        fprintf(stderr, "selfplay: DONE.json write failed\n");
        return 1;
    }
    double dt = now_s() - t0;
    uint64_t oom = 0, compact = 0;
    for (int i = 0; i < pool; i++) {
        oom += puct_oom_count(handles[i]);
        compact += puct_compact_count(handles[i]);
    }
    printf("SUMMARY games=%u b=%d komi=%.1f sims=%d fast_sims=%d "
           "fast_p=%.2f pool=%d rows=%d elapsed=%.1fs rate=%.2f games/s "
           "oom=%llu compact=%llu depth_mean=%.1f depth_max=%u\n",
           finished, b, komi, sims, fast_sims, fast_p, pool, rows, dt,
           dt > 0 ? finished / dt : 0.0,
           (unsigned long long)oom, (unsigned long long)compact,
           depth_thinks ? depth_sum / depth_thinks : 0.0, depth_max);
    return 0;
}
