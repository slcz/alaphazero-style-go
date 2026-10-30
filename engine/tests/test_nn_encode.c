/* test_nn_encode.c — encoder spec tests (hand-computed) + replay fuzz.
 *
 * The C encoder is the single source of truth; these tests pin its spec
 * with hand-computed values and replay invariants (no second encoder —
 * a twin implementation would only test itself). */
#include "go_nn.h"

#include <stdio.h>
#include <string.h>

static int fails, checks;

#define CHECK(cond) do {                                          \
        checks++;                                                 \
        if (!(cond)) {                                            \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            fails++;                                              \
        }                                                         \
    } while (0)

static int pt(int b, int x, int y) { return y * b + x; }

static void test_empty_board(void) {
    State s;
    float t[NN_TOTAL];
    go_state_init(&s, 7, 9.5f, 150);
    nn_encode(&s, t);

    float *a1 = t + (size_t)pt(7, 0, 0) * NN_TOKEN_DIM;
    CHECK(a1[GNN_OWN] == 0 && a1[GNN_OPP] == 0);
    CHECK(a1[GNN_EDGE] == 0.0f && a1[GNN_EDGE + 2] == 0.0f);
    CHECK(a1[GNN_EDGE + 1] == 6.0f / 7 && a1[GNN_EDGE + 3] == 6.0f / 7);

    float *center = t + (size_t)pt(7, 3, 3) * NN_TOKEN_DIM;
    CHECK(center[GNN_EDGE] == 3.0f / 7 && center[GNN_EDGE + 1] == 3.0f / 7);
    CHECK(center[GNN_EDGE + 2] == 3.0f / 7 && center[GNN_EDGE + 3] == 3.0f / 7);

    for (int h = 0; h < GO_HIST; h++) {
        CHECK(a1[GNN_LASTM + h] == 0 && a1[GNN_PASS + h] == 0);
    }
    CHECK(a1[GNN_KOMI] == 9.5f / 15.0f);
    CHECK(a1[GNN_SIZE] == 7.0f / 19.0f);
}

static void test_mover_perspective_and_last_move(void) {
    State s;
    float t[NN_TOTAL];
    go_state_init(&s, 7, 9.5f, 150);
    make_move(&s, 24);                  /* B D4 */
    nn_encode(&s, t);                   /* white to move */

    float *d4 = t + (size_t)24 * NN_TOKEN_DIM;
    CHECK(d4[GNN_OWN] == 0 && d4[GNN_OPP] == 1);    /* black stone, white mover */
    CHECK(d4[GNN_LASTM + 0] == 1);
    CHECK(d4[GNN_LASTM + 1] == 0);
    CHECK(d4[GNN_PASS + 0] == 0);
    float *other = t;                   /* a1: everything off */
    CHECK(other[GNN_LASTM + 0] == 0);

    make_move(&s, MOVE_PASS);           /* W pass */
    nn_encode(&s, t);                   /* black to move */
    CHECK(d4[GNN_OWN] == 1 && d4[GNN_OPP] == 0);    /* same stone, black mover */
    CHECK(d4[GNN_LASTM + 0] == 0);                  /* newest move is the pass */
    CHECK(d4[GNN_LASTM + 1] == 1);                  /* D4 now one ply older */
    CHECK(d4[GNN_PASS + 0] == 1);
    CHECK(d4[GNN_PASS + 1] == 0);
}

static void test_history_window(void) {
    State s;
    float t[NN_TOTAL];
    go_state_init(&s, 5, 5.5f, 100);
    int moves[7] = {0, 6, 12, 18, 24, 2, 3};
    for (int i = 0; i < 7; i++)
        make_move(&s, (Move)moves[i]);
    nn_encode(&s, t);
    /* window = moves[2..6] as h=4..0 */
    for (int h = 0; h < GO_HIST; h++) {
        int want = moves[7 - 1 - h];
        float *row = t + (size_t)want * NN_TOKEN_DIM;
        CHECK(row[GNN_LASTM + h] == 1);
        CHECK(row[GNN_PASS + h] == 0);
    }
    CHECK(t[(size_t)moves[0] * NN_TOKEN_DIM + GNN_LASTM + 0] == 0);
    CHECK(t[(size_t)moves[1] * NN_TOKEN_DIM + GNN_LASTM + 0] == 0);
    /* exactly one one-hot per last-move plane */
    for (int h = 0; h < GO_HIST; h++) {
        int sum = 0;
        for (int p = 0; p < 25; p++)
            sum += t[(size_t)p * NN_TOKEN_DIM + GNN_LASTM + h] == 1;
        CHECK(sum == 1);
    }
}

static uint64_t rng = 0x13198A2E03707344ULL;
static uint64_t rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

/* replay invariants over random games */
static void fuzz(int b, int games, int maxply, float komi) {
    for (int g = 0; g < games; g++) {
        State s;
        MoveList ml;
        Move played[600];
        go_state_init(&s, b, komi, maxply);
        float t[NN_TOTAL];
        for (;;) {
            nn_encode(&s, t);
            for (int p = 0; p < b * b; p++) {
                int c = s.board[p];
                float *row = t + (size_t)p * NN_TOKEN_DIM;
                CHECK((row[GNN_OWN] == 1) == (c == s.side));
                CHECK((row[GNN_OPP] == 1) == (c != EMPTY && c != s.side));
                int x = p % b, y = p / b;
                CHECK(row[GNN_EDGE] == (float)x / b);
                CHECK(row[GNN_EDGE + 1] == (float)(b - 1 - x) / b);
                CHECK(row[GNN_EDGE + 2] == (float)y / b);
                CHECK(row[GNN_EDGE + 3] == (float)(b - 1 - y) / b);
                CHECK(row[GNN_KOMI] == komi / 15.0f);
                CHECK(row[GNN_SIZE] == (float)b / 19.0f);
            }
            for (int h = 0; h < GO_HIST; h++) {   /* window vs replayed moves */
                int ply = s.ply - 1 - h;
                bool want_pass = ply >= 0 && played[ply] == MOVE_PASS;
                int hits = 0;
                for (int p = 0; p < b * b; p++) {
                    if (t[(size_t)p * NN_TOKEN_DIM + GNN_LASTM + h] == 1)
                        hits++;
                    if (ply >= 0 && !want_pass)
                        CHECK((t[(size_t)p * NN_TOKEN_DIM + GNN_LASTM + h] == 1)
                              == (p == (int)played[ply]));
                }
                CHECK(hits == (ply >= 0 && !want_pass));
                for (int p = 0; p < b * b; p++)
                    CHECK((t[(size_t)p * NN_TOKEN_DIM + GNN_PASS + h] == 1)
                          == want_pass);
            }
            if (state_termination(&s, NULL, NULL) == TERM_OVER || s.ply >= 600)
                break;
            gen_legal(&s, &ml);
            Move m = (rnd() % 16 == 0) ? MOVE_PASS : ml.moves[rnd() % ml.count];
            make_move(&s, m);
            played[s.ply - 1] = m;
        }
    }
}

int main(void) {
    go_init();
    test_empty_board();
    test_mover_perspective_and_last_move();
    test_history_window();
    fuzz(7, 60, 150, 9.5f);
    fuzz(9, 20, 200, 7.5f);
    printf("%s: %d checks, %d failures\n",
           fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
