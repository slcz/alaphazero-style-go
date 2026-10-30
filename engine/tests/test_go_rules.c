/* test_go_rules.c — TT rules unit tests + random-play fuzz.
 *
 * Point index = y*b + x. Crafted positions bypass make_move, so sync_key()
 * refreshes keys[ply] before gen_legal is trusted. */
#include "go.h"

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

/* ---- helpers ------------------------------------------------------------- */

static void put(State *s, int x, int y, int c) { s->board[y * s->b + x] = (uint8_t)c; }
static int  at(const State *s, int x, int y)  { return s->board[y * s->b + x]; }
static int  pt(const State *s, int x, int y)  { return y * s->b + x; }

static uint64_t manual_key(const State *s) {
    uint64_t k = GO_KEY_BASE;
    for (int i = 0; i < s->b * s->b; i++)
        if (s->board[i] != EMPTY)
            k ^= zob_stone[s->board[i]][i];
    return k;
}
static void sync_key(State *s) { s->keys[s->ply] = manual_key(s); }

static bool in_list(const MoveList *ml, Move m) {
    for (int i = 0; i < ml->count; i++)
        if (ml->moves[i] == m)
            return true;
    return false;
}

/* every group on the board must have a liberty (post-clearing invariant) */
static bool all_groups_alive(const State *s) {
    const int b = s->b, n = b * b;
    uint8_t seen[GO_MAX_POINTS] = {0};
    int stack[GO_MAX_POINTS];
    for (int p = 0; p < n; p++) {
        if (s->board[p] == EMPTY || seen[p])
            continue;
        int top = 0;
        bool lib = false;
        seen[p] = 1;
        stack[top++] = p;
        while (top > 0) {
            int q = stack[--top];
            int x = q % b, y = q / b;
#define T(r) do {                                              \
            if (s->board[r] == EMPTY) lib = true;              \
            else if (s->board[r] == s->board[p] && !seen[r]) { \
                seen[r] = 1; stack[top++] = r;                 \
            }                                                  \
        } while (0)
            if (x > 0) T(q - 1);
            if (x < b - 1) T(q + 1);
            if (y > 0) T(q - b);
            if (y < b - 1) T(q + b);
#undef T
        }
        if (!lib)
            return false;
    }
    return true;
}

/* independent TT scorer (BFS over empty regions) — cross-checks tt_score */
static float manual_score(const State *s) {
    const int b = s->b, n = b * b;
    uint8_t seen[GO_MAX_POINTS] = {0};
    int queue[GO_MAX_POINTS];
    int bs = 0, ws = 0;
#define EDGE(r) do {                                                     \
        if (s->board[r] == EMPTY) {                                      \
            if (!seen[r]) { seen[r] = 1; queue[tail++] = r; }            \
        } else {                                                         \
            rb |= s->board[r] == BLACK;                                  \
            rw |= s->board[r] == WHITE;                                  \
        }                                                                \
    } while (0)
    for (int p = 0; p < n; p++) {
        if (s->board[p] == BLACK) bs++;
        else if (s->board[p] == WHITE) ws++;
        else if (!seen[p]) {
            int head = 0, tail = 0, nr = 0;
            bool rb = false, rw = false;
            seen[p] = 1;
            queue[tail++] = p;
            while (head < tail) {
                int q = queue[head++];
                nr++;
                int x = q % b, y = q / b;
                if (x > 0) EDGE(q - 1);
                if (x < b - 1) EDGE(q + 1);
                if (y > 0) EDGE(q - b);
                if (y < b - 1) EDGE(q + b);
            }
            if (rb && !rw) bs += nr;
            if (rw && !rb) ws += nr;
        }
    }
#undef EDGE
    return (float)(bs - ws) - s->komi;
}

static uint64_t rng = 0x243F6A8885A308D3ULL;
static uint64_t rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

/* ---- unit tests ---------------------------------------------------------- */

static void test_empty_board(void) {
    State s;
    MoveList ml;
    go_state_init(&s, 7, 9.5f, 150);
    gen_legal(&s, &ml);
    CHECK(ml.count == 50);              /* 49 points + pass */
    CHECK(ml.moves[49] == MOVE_PASS);
    for (int i = 0; i < 49; i++)
        CHECK(ml.moves[i] == i);        /* every point legal, index order */
    CHECK(state_termination(&s, NULL, NULL) == TERM_PLAYING);
    CHECK(s.side == BLACK);
}

static void test_make_move_and_key(void) {
    State s;
    go_state_init(&s, 7, 9.5f, 150);
    make_move(&s, 24);                  /* D4 */
    CHECK(at(&s, 3, 3) == BLACK);
    CHECK(s.ply == 1 && s.side == WHITE && s.n_pass == 0);
    CHECK(state_poskey(&s) == manual_key(&s));
    CHECK(s.hist[0] == 24);

    make_move(&s, MOVE_PASS);
    CHECK(s.ply == 2 && s.side == BLACK && s.n_pass == 1);
    CHECK(state_poskey(&s) == s.keys[1]);   /* pass keeps the placement */
    CHECK(s.hist[1] == MOVE_PASS);
    CHECK(state_termination(&s, NULL, NULL) == TERM_PLAYING);

    make_move(&s, 10);
    CHECK(s.n_pass == 0);
    CHECK(s.hist[(s.ply - 1) % GO_HIST] == 10);
    CHECK(s.hist[(s.ply - 3) % GO_HIST] == 24);
}

static void test_capture(void) {
    State s;
    go_state_init(&s, 5, 5.5f, 100);
    make_move(&s, pt(&s, 1, 0));   /* B b1 */
    make_move(&s, pt(&s, 0, 0));   /* W a1 */
    make_move(&s, pt(&s, 4, 4));   /* B e5 */
    make_move(&s, pt(&s, 2, 0));   /* W c1 */
    make_move(&s, pt(&s, 4, 3));   /* B e4 */
    make_move(&s, pt(&s, 1, 1));   /* W b2 captures B b1 */
    CHECK(at(&s, 1, 0) == EMPTY);
    CHECK(at(&s, 0, 0) == WHITE && at(&s, 2, 0) == WHITE && at(&s, 1, 1) == WHITE);
    CHECK(state_poskey(&s) == manual_key(&s));
    CHECK(all_groups_alive(&s));

    MoveList ml;
    gen_legal(&s, &ml);
    CHECK(!in_list(&ml, pt(&s, 1, 0)));  /* replaying into the eye is
                                            single-stone suicide: the
                                            placement recurs -> superko */
}

static void test_suicide(void) {
    State s;
    MoveList ml;

    /* single-stone suicide recreates the placement -> superko rejects it */
    go_state_init(&s, 5, 5.5f, 100);
    put(&s, 1, 0, WHITE); put(&s, 0, 1, WHITE);
    put(&s, 2, 1, WHITE); put(&s, 1, 2, WHITE);
    sync_key(&s);
    s.side = BLACK;
    gen_legal(&s, &ml);
    CHECK(!in_list(&ml, pt(&s, 1, 1)));
    CHECK(in_list(&ml, MOVE_PASS));

    /* multi-stone suicide is legal: the own group is cleared */
    go_state_init(&s, 5, 5.5f, 100);
    put(&s, 3, 0, WHITE); put(&s, 3, 1, WHITE); put(&s, 3, 2, WHITE);
    put(&s, 4, 3, WHITE);
    put(&s, 4, 0, BLACK); put(&s, 4, 1, BLACK);
    sync_key(&s);
    s.side = BLACK;
    gen_legal(&s, &ml);
    CHECK(in_list(&ml, pt(&s, 4, 2)));    /* fills the group's last liberty */
    make_move(&s, pt(&s, 4, 2));
    CHECK(at(&s, 4, 0) == EMPTY && at(&s, 4, 1) == EMPTY && at(&s, 4, 2) == EMPTY);
    CHECK(at(&s, 3, 1) == WHITE);        /* no opponent stones were taken */
    CHECK(all_groups_alive(&s));
    CHECK(state_poskey(&s) == manual_key(&s));
}

static void test_ko(void) {
    State s;
    MoveList ml;
    go_state_init(&s, 5, 5.5f, 100);
    /* white (2,1) has one liberty (3,1); after black takes it, white's
     * recapture would recreate the pre-capture placement */
    put(&s, 2, 1, WHITE); put(&s, 3, 0, WHITE); put(&s, 3, 2, WHITE); put(&s, 4, 1, WHITE);
    put(&s, 1, 1, BLACK); put(&s, 2, 0, BLACK); put(&s, 2, 2, BLACK);
    sync_key(&s);
    s.side = BLACK;

    make_move(&s, pt(&s, 3, 1));         /* captures white (2,1) */
    CHECK(at(&s, 2, 1) == EMPTY);
    CHECK(at(&s, 3, 1) == BLACK);
    CHECK(all_groups_alive(&s));

    gen_legal(&s, &ml);                  /* white to move */
    CHECK(!in_list(&ml, pt(&s, 2, 1)));  /* recapture recreates keys[0] */
    CHECK(in_list(&ml, MOVE_PASS));

    make_move(&s, pt(&s, 0, 3));         /* white elsewhere */
    make_move(&s, pt(&s, 0, 4));         /* black elsewhere */
    gen_legal(&s, &ml);
    CHECK(in_list(&ml, pt(&s, 2, 1)));   /* placement differs: legal now */
    make_move(&s, pt(&s, 2, 1));         /* captures black (3,1) */
    CHECK(at(&s, 3, 1) == EMPTY);
    CHECK(at(&s, 2, 1) == WHITE);
    CHECK(all_groups_alive(&s));
}

static void test_scoring(void) {
    State s;
    float score, own[GO_MAX_POINTS];

    /* empty board, both pass: white by komi, all points neutral */
    go_state_init(&s, 5, 5.5f, 100);
    make_move(&s, MOVE_PASS);
    make_move(&s, MOVE_PASS);
    CHECK(state_termination(&s, &score, own) == TERM_OVER);
    CHECK(score == -5.5f);
    for (int i = 0; i < 25; i++)
        CHECK(own[i] == 0.5f);

    /* row0 black, row2 white, one white stone at (3,4): row1 dame, rows 3-4
     * white territory. B = 5 stones; W = 6 stones + 9 territory = 15 */
    go_state_init(&s, 5, 0.5f, 100);
    for (int x = 0; x < 5; x++) {
        put(&s, x, 0, BLACK);
        put(&s, x, 2, WHITE);
    }
    put(&s, 3, 4, WHITE);
    sync_key(&s);
    CHECK(state_termination(&s, &score, own) == TERM_PLAYING);  /* not over */
    make_move(&s, MOVE_PASS);
    make_move(&s, MOVE_PASS);
    CHECK(state_termination(&s, &score, own) == TERM_OVER);
    CHECK(score == (5.0f - 15.0f) - 0.5f);
    CHECK(own[pt(&s, 0, 0)] == 1.0f);    /* black stone */
    CHECK(own[pt(&s, 0, 1)] == 0.5f);    /* dame row */
    CHECK(own[pt(&s, 0, 2)] == 0.0f);    /* white stone */
    CHECK(own[pt(&s, 0, 3)] == 0.0f);    /* white territory */
    CHECK(own[pt(&s, 4, 4)] == 0.0f);
    CHECK(manual_score(&s) == score);

    /* strict TT: a lone "dead" black stone in white's area is still black,
     * and its region reaches both colors -> dame */
    go_state_init(&s, 5, 0.5f, 100);
    for (int x = 0; x < 5; x++) {
        put(&s, x, 0, BLACK);
        put(&s, x, 2, WHITE);
    }
    put(&s, 2, 4, BLACK);
    sync_key(&s);
    make_move(&s, MOVE_PASS);
    make_move(&s, MOVE_PASS);
    CHECK(state_termination(&s, &score, own) == TERM_OVER);
    CHECK(score == (6.0f - 5.0f) - 0.5f);
    CHECK(own[pt(&s, 2, 4)] == 1.0f);
    CHECK(own[pt(&s, 2, 3)] == 0.5f);    /* region touches both colors */
    CHECK(manual_score(&s) == score);
}

static void test_maxply(void) {
    State s;
    MoveList ml;
    go_state_init(&s, 5, 5.5f, 3);
    make_move(&s, 0);
    make_move(&s, 1);
    CHECK(state_termination(&s, NULL, NULL) == TERM_PLAYING);
    make_move(&s, 2);
    CHECK(state_termination(&s, NULL, NULL) == TERM_OVER);
    gen_legal(&s, &ml);
    CHECK(ml.count == 0);
}

static void test_gtp(void) {
    State s19, s7;
    go_state_init(&s19, 19, 7.5f, 500);
    go_state_init(&s7, 7, 9.5f, 150);
    char v[4];
    Move m;

    move_to_gtp(0, 19, v);
    CHECK(strcmp(v, "A1") == 0);
    move_to_gtp(pt(&s19, 8, 18), 19, v);
    CHECK(strcmp(v, "J19") == 0);        /* column I skipped */
    move_to_gtp(MOVE_PASS, 19, v);
    CHECK(strcmp(v, "pass") == 0);

    for (int y = 0; y < 19; y++)
        for (int x = 0; x < 19; x++) {
            Move p = (Move)(y * 19 + x);
            move_to_gtp(p, 19, v);
            CHECK(go_parse_gtp(&s19, v, &m) == 0 && m == p);
        }
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++) {
            Move p = (Move)(y * 7 + x);
            move_to_gtp(p, 7, v);
            CHECK(go_parse_gtp(&s7, v, &m) == 0 && m == p);
        }

    CHECK(go_parse_gtp(&s7, "pass", &m) == 0 && m == MOVE_PASS);
    CHECK(go_parse_gtp(&s19, "I2", &m) == -1);    /* no I column */
    CHECK(go_parse_gtp(&s19, "A0", &m) == -1);
    CHECK(go_parse_gtp(&s19, "A20", &m) == -1);
    CHECK(go_parse_gtp(&s7, "H1", &m) == -1);     /* off the 7x7 board */
    CHECK(go_parse_gtp(&s7, "", &m) == -1);
}

static void test_zobrist_stable(void) {
    uint64_t a = zob_stone[BLACK][0];
    go_init();
    CHECK(zob_stone[BLACK][0] == a);
}

/* ---- fuzz ---------------------------------------------------------------- */

static void fuzz(int b, int games, int maxply, float komi) {
    for (int g = 0; g < games; g++) {
        State s;
        MoveList ml;
        go_state_init(&s, b, komi, maxply);
        for (;;) {
            if (state_termination(&s, NULL, NULL) == TERM_OVER) {
                float score, own[GO_MAX_POINTS];
                state_termination(&s, &score, own);
                CHECK(score == manual_score(&s));
                CHECK(score >= -(float)(b * b) - komi);
                CHECK(score <= (float)(b * b) - komi);
                for (int i = 0; i < b * b; i++)
                    CHECK(own[i] == 0.0f || own[i] == 0.5f || own[i] == 1.0f);
                break;
            }
            gen_legal(&s, &ml);
            CHECK(ml.count >= 1);        /* pass is always there */
            Move m = (rnd() % 16 == 0) ? MOVE_PASS : ml.moves[rnd() % ml.count];
            uint8_t before[GO_MAX_POINTS];
            memcpy(before, s.board, sizeof before);
            int mover = s.side;
            make_move(&s, m);
            CHECK(s.ply <= maxply);
            if (m == MOVE_PASS) {
                CHECK(memcmp(before, s.board, (size_t)(b * b)) == 0);
            } else {
                CHECK(before[m] == EMPTY);
                /* on suicide the played stone is cleared with its group */
                CHECK(s.board[m] == mover || s.board[m] == EMPTY);
            }
            for (int i = 0; i < b * b; i++) {   /* only removals besides */
                if (s.board[i] == before[i] || i == (int)m)
                    continue;
                CHECK(before[i] != EMPTY && s.board[i] == EMPTY);
            }
            if (m != MOVE_PASS)
                for (int i = 0; i < s.ply; i++)
                    CHECK(s.keys[i] != s.keys[s.ply]);   /* superko holds */
            CHECK(all_groups_alive(&s));
            CHECK(state_poskey(&s) == manual_key(&s));
        }
    }
}

int main(void) {
    go_init();
    test_zobrist_stable();
    test_empty_board();
    test_make_move_and_key();
    test_capture();
    test_suicide();
    test_ko();
    test_scoring();
    test_maxply();
    test_gtp();
    fuzz(7, 300, 150, 9.5f);
    fuzz(9, 60, 200, 7.5f);
    fuzz(19, 6, 250, 7.5f);
    printf("%s: %d checks, %d failures\n",
           fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
