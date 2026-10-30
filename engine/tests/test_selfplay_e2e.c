/* test_selfplay_e2e.c — end-to-end selfplay gate (no net): run the
 * driver with the stub evaluator on a small pool, then replay the shard
 * through the rules engine and validate every record — the π̄ order
 * contract, legality, mover, per-ply budgets, and the TT score/ownership
 * labels. Must run from engine/ (make check does). */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "go.h"
#include "go_sp_dump.h"

static int fails, checks;

#define CHECK(cond) do {                                          \
        checks++;                                                 \
        if (!(cond)) {                                            \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            fails++;                                              \
        }                                                         \
    } while (0)

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0)
            bits = sign;
        else {
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400u));
            bits = sign | ((uint32_t)(127 - 14 + e - 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static int read_exact(FILE *f, void *p, size_t n) {
    return fread(p, 1, n, f) == n ? 0 : -1;
}

int main(void) {
    const char *run_dir = "build/test_sp_run";
    const int games = 4, b = 7, sims = 8, fast_sims = 4, max_plys = 40;
    go_init();   /* zobrist tables — without them every point looks
                    superko-illegal to the replay */
    char cmd[512];
    snprintf(cmd, sizeof cmd,
             "mkdir -p %s && ./build/selfplay none %s %d %d 9.5 %d 4 16 6 "
             "%d 1 dummy 7 %d 0.5 > %s/selfplay.log 2>&1",
             run_dir, run_dir, games, b, sims, max_plys, fast_sims,
             run_dir);
    printf("running: %s\n", cmd);
    CHECK(system(cmd) == 0);

    char path[512];
    snprintf(path, sizeof path, "%s/shard-0.spbin", run_dir);
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    if (!f)
        return 1;
    char magic[8];
    struct { uint32_t version, worker; uint64_t s; } hdr;
    CHECK(read_exact(f, magic, 8) == 0 && memcmp(magic, "GOSHARD1", 8) == 0);
    CHECK(read_exact(f, &hdr, sizeof hdr) == 0);
    CHECK(hdr.version == SP_VERSION && hdr.worker == 0 && hdr.s == 7);

    for (int gi = 0; gi < games; gi++) {
        uint32_t game_index, n_plies;
        uint8_t hb, end_kind;
        uint16_t n_sims;
        float komi, score;
        CHECK(read_exact(f, &game_index, 4) == 0);
        CHECK(read_exact(f, &n_plies, 4) == 0);
        CHECK(read_exact(f, &hb, 1) == 0);
        CHECK(read_exact(f, &end_kind, 1) == 0);
        CHECK(read_exact(f, &n_sims, 2) == 0);
        CHECK(read_exact(f, &komi, 4) == 0);
        CHECK(read_exact(f, &score, 4) == 0);
        CHECK(game_index == (uint32_t)gi);
        CHECK(hb == b && komi == 9.5f && n_sims == sims);
        uint16_t own16[GO_MAX_POINTS];
        CHECK(read_exact(f, own16, (size_t)b * b * 2) == 0);

        State s;
        go_state_init(&s, b, komi, max_plys);
        uint16_t played[SP_PI_MAX];
        for (uint32_t p = 0; p < n_plies; p++) {
            uint8_t mover;
            uint16_t move, n_pi, ply_sims;
            float outlook;
            CHECK(read_exact(f, &mover, 1) == 0);
            CHECK(read_exact(f, &move, 2) == 0);
            CHECK(read_exact(f, &n_pi, 2) == 0);
            CHECK(read_exact(f, &ply_sims, 2) == 0);
            CHECK(read_exact(f, &outlook, 4) == 0);
            uint16_t p16[SP_PI_MAX];
            CHECK(read_exact(f, p16, (size_t)n_pi * 2) == 0);

            /* the π̄ order contract + legality + bookkeeping */
            MoveList ml;
            gen_legal(&s, &ml);
            CHECK(state_termination(&s, NULL, NULL) == TERM_PLAYING);
            CHECK((int)n_pi == ml.count);
            CHECK(mover == s.side);
            CHECK(ply_sims == sims || ply_sims == fast_sims);
            bool legal = false;
            for (int i = 0; i < ml.count; i++)
                if (ml.moves[i] == move)
                    legal = true;
            CHECK(legal);
            double sum = 0;
            for (int i = 0; i < n_pi; i++) {
                float pr = f16_to_f32(p16[i]);
                CHECK(pr > 0.0f && pr <= 1.0f);
                sum += pr;
            }
            CHECK(sum > 0.98 && sum < 1.02);   /* f16-rounded dense π̄ */
            played[p] = move;
            make_move(&s, move);
        }

        /* game over exactly at the last ply; TT labels match the replay */
        float rscore, rown[GO_MAX_POINTS];
        CHECK(state_termination(&s, &rscore, rown) == TERM_OVER);
        CHECK(rscore == score);
        for (int i = 0; i < b * b; i++)
            CHECK(f16_to_f32(own16[i]) == rown[i]);
        bool dbl_pass = n_plies >= 2 && played[n_plies - 1] == MOVE_PASS &&
                        played[n_plies - 2] == MOVE_PASS;
        CHECK((end_kind == SP_END_NATURAL) == dbl_pass);
        CHECK(end_kind != SP_END_NATURAL || s.n_pass >= 2);
        CHECK(end_kind != SP_END_ADJUDICATED || s.ply == max_plys);
        CHECK(n_plies <= (uint32_t)max_plys);
    }
    fclose(f);

    /* DONE.json gate */
    snprintf(path, sizeof path, "%s/DONE.json", run_dir);
    FILE *d = fopen(path, "r");
    CHECK(d != NULL);
    if (d) {
        char buf[256];
        CHECK(fgets(buf, sizeof buf, d) != NULL);
        int done_games = -1;
        CHECK(sscanf(buf, "{\"n_workers\": %*d, \"unit_seed\": %*llu, "
                          "\"games_done\": %d}", &done_games) == 1);
        CHECK(done_games == games);
        fclose(d);
    }

    printf("%s: %d checks, %d failures\n",
           fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
