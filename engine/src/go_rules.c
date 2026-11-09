/* go_rules.c — Tromp-Taylor rules (go.h). Pure except make_move. */
#include "go.h"

#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

uint64_t zob_stone[3][GO_MAX_POINTS];

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void go_init(void) {
    /* fixed seed: keys are stable across processes and platforms */
    uint64_t rng = 0x123456789ABCDEF0ULL;
    for (int c = BLACK; c <= WHITE; c++)
        for (int i = 0; i < GO_MAX_POINTS; i++)
            zob_stone[c][i] = splitmix64(&rng);
}

void go_state_init(State *s, int b, float komi, int maxply) {
    assert(b >= 2 && b <= GO_MAX_B);
    assert(maxply >= 1 && maxply <= MAX_PLY);
    memset(s, 0, sizeof *s);
    s->b = (uint8_t)b;
    s->side = BLACK;
    s->komi = komi;
    s->maxply = (uint16_t)maxply;
    s->keys[0] = GO_KEY_BASE;
}

/* Flood the group containing `root`; stones[] receives the group, *n its
 * size. Cell `pt` reads as a stone of `pt_color` (the candidate); points
 * set in `removed` (captures this move) read as empty. A neighbor is a
 * liberty when it reads empty. visited marks GROUP MEMBERS ONLY (never
 * liberty cells — several groups share an epoch and must each see them);
 * the caller bumps epoch per scan (never 0 — arrays start zeroed). */
static bool flood(const State *s, int root, int pt, int pt_color,
                  const uint8_t *removed, int *stones, int *n,
                  int *stack, uint16_t *visited, uint16_t epoch) {
    const int b = s->b;
    const int color = (root == pt) ? pt_color : s->board[root];
    int top = 0, n_st = 0;
    bool lib = false;
    visited[root] = epoch;
    stack[top++] = root;
    while (top > 0) {
        int p = stack[--top];
        stones[n_st++] = p;
        const int x = p % b, y = p / b;
#define GO_VISIT(q) do {                                          \
            if (visited[q] == epoch) break;                       \
            int cq = (q == pt) ? pt_color : s->board[q];          \
            if (cq == color) {                                    \
                visited[q] = epoch;                               \
                stack[top++] = q;                                 \
                break;                                            \
            }                                                     \
            if ((s->board[q] == EMPTY && q != pt)                 \
                || (removed && removed[q])) lib = true;           \
        } while (0)
        if (x > 0)     GO_VISIT(p - 1);
        if (x < b - 1) GO_VISIT(p + 1);
        if (y > 0)     GO_VISIT(p - b);
        if (y < b - 1) GO_VISIT(p + b);
#undef GO_VISIT
    }
    *n = n_st;
    return lib;
}

/* Placement delta of playing `pt` (must be EMPTY) for the side to move:
 * captured stones (opponent groups left libertyless, then the own group on
 * suicide) go to rm_list and *n_rm; returns the resulting placement key. Pure.
 * rm_mask must arrive zeroed and is left zeroed. visited/epoch carry over
 * between calls (epoch only grows). */
static uint64_t play_delta(const State *s, int pt, uint8_t *rm_mask,
                           int *rm_list, int *n_rm, int *stones, int *stack,
                           uint16_t *visited, uint16_t *epoch) {
    const int b = s->b;
    const int me = s->side, them = stone_opposite(me);
    uint64_t key = s->keys[s->ply] ^ zob_stone[me][pt];
    const int x = pt % b, y = pt / b;

    /* opponent groups adjacent to pt, evaluated simultaneously: the board
     * plus the candidate stone, other captures NOT yet empty (TT rule 5) */
    uint16_t ep = ++*epoch;
    *n_rm = 0;
#define GO_TRY(q) do {                                                        \
        if (s->board[q] != them || visited[q] == ep) break;                   \
        int n;                                                                \
        if (!flood(s, q, pt, me, NULL, stones, &n, stack, visited, ep)) {     \
            for (int i = 0; i < n; i++) {                                     \
                int c = stones[i];                                            \
                rm_mask[c] = 1; rm_list[(*n_rm)++] = c;                       \
                key ^= zob_stone[them][c];                                    \
            }                                                                 \
        }                                                                     \
    } while (0)
    if (x > 0)     GO_TRY(pt - 1);
    if (x < b - 1) GO_TRY(pt + 1);
    if (y > 0)     GO_TRY(pt - b);
    if (y < b - 1) GO_TRY(pt + b);
#undef GO_TRY

    /* own group including pt; captures now read as empty */
    ep = ++*epoch;
    int n;
    if (!flood(s, pt, pt, me, rm_mask, stones, &n, stack, visited, ep)) {
        for (int i = 0; i < n; i++) {
            int c = stones[i];
            rm_mask[c] = 1; rm_list[(*n_rm)++] = c;
            key ^= zob_stone[me][c];
        }
    }
    for (int i = 0; i < *n_rm; i++) rm_mask[rm_list[i]] = 0;
    return key;
}

static bool game_over(const State *s) {
    return s->n_pass >= 2 || s->ply >= s->maxply;
}

void gen_legal(const State *s, MoveList *ml) {
    ml->count = 0;
    if (game_over(s))
        return;
    const int b = s->b, n = b * b;
    uint8_t  rm_mask[GO_MAX_POINTS] = {0};
    int      rm_list[GO_MAX_POINTS], stones[GO_MAX_POINTS], stack[GO_MAX_POINTS];
    uint16_t visited[GO_MAX_POINTS] = {0};
    uint16_t epoch = 0;
    for (int pt = 0; pt < n; pt++) {
        if (s->board[pt] != EMPTY)
            continue;
        int n_rm;
        uint64_t k = play_delta(s, pt, rm_mask, rm_list, &n_rm,
                                stones, stack, visited, &epoch);
        for (int i = 0; i <= s->ply; i++)
            if (s->keys[i] == k)
                goto next;          /* positional superko */
        ml->moves[ml->count++] = (Move)pt;
    next:;
    }
    ml->moves[ml->count++] = MOVE_PASS;
}

void make_move(State *s, Move m) {
    assert(s->ply < MAX_PLY);
    uint64_t key;
    if (m == MOVE_PASS) {
        key = s->keys[s->ply];
        s->n_pass++;
    } else {
        assert(m < s->b * s->b && s->board[m] == EMPTY);
        uint8_t  rm_mask[GO_MAX_POINTS] = {0};
        int      rm_list[GO_MAX_POINTS], stones[GO_MAX_POINTS],
                 stack[GO_MAX_POINTS], n_rm;
        uint16_t visited[GO_MAX_POINTS] = {0}, epoch = 0;
        key = play_delta(s, m, rm_mask, rm_list, &n_rm,
                         stones, stack, visited, &epoch);
        s->board[m] = s->side;
        for (int i = 0; i < n_rm; i++)
            s->board[rm_list[i]] = EMPTY;
        s->n_pass = 0;
    }
    s->hist[s->ply % GO_HIST] = m;
    s->ply++;
    s->keys[s->ply] = key;
    s->side = stone_opposite(s->side);
}

uint64_t state_poskey(const State *s) { return s->keys[s->ply]; }

/* TT score and ownership of the position AS IT STANDS, mid-game included.
 * Same code path as the final score, so a decision taken from it is
 * consistent with the result the game will be given. */
void state_score(const State *s, float *score, float *ownership) {
    if (score || ownership) {
        float tmp[GO_MAX_POINTS];
        float *own = ownership ? ownership : tmp;
        const int b = s->b, n = b * b;
        uint8_t seen[GO_MAX_POINTS] = {0};
        int stack[GO_MAX_POINTS], region[GO_MAX_POINTS];
        int bs = 0, ws = 0;
        for (int p = 0; p < n; p++) {
            if (s->board[p] == BLACK) { bs++; own[p] = 1.0f; }
            else if (s->board[p] == WHITE) { ws++; own[p] = 0.0f; }
            else if (!seen[p]) {
                /* empty region: owned by a color iff it is the only one
                 * the region reaches (TT rule 9) */
                int top = 0, nr = 0;
                bool rb = false, rw = false;
                seen[p] = 1;
                stack[top++] = p;
                while (top > 0) {
                    int q = stack[--top];
                    region[nr++] = q;
                    const int x = q % b, y = q / b;
#define GO_TOUCH(r) do {                                   \
                        if (s->board[r] == EMPTY) {        \
                            if (!seen[r]) {                \
                                seen[r] = 1;               \
                                stack[top++] = r;          \
                            }                              \
                        } else if (s->board[r] == BLACK)   \
                            rb = true;                     \
                        else                               \
                            rw = true;                     \
                    } while (0)
                    if (x > 0)     GO_TOUCH(q - 1);
                    if (x < b - 1) GO_TOUCH(q + 1);
                    if (y > 0)     GO_TOUCH(q - b);
                    if (y < b - 1) GO_TOUCH(q + b);
#undef GO_TOUCH
                }
                if (rb && !rw) bs += nr;
                else if (rw && !rb) ws += nr;
                for (int i = 0; i < nr; i++)
                    own[region[i]] = (rb != rw) ? (rb ? 1.0f : 0.0f) : 0.5f;
            }
        }
        if (score)
            *score = (float)(bs - ws) - s->komi;
    }
}

int state_termination(const State *s, float *score, float *ownership) {
    if (!game_over(s))
        return TERM_PLAYING;
    state_score(s, score, ownership);
    return TERM_OVER;
}

static const char GO_COLS[] = "ABCDEFGHJKLMNOPQRST";   /* GTP: no I */

void move_to_gtp(Move m, int b, char out[6]) {
    if (m == MOVE_PASS) {
        strcpy(out, "pass");
        return;
    }
    out[0] = GO_COLS[m % b];
    int r = m / b + 1;
    if (r >= 10) {
        out[1] = (char)('0' + r / 10);
        out[2] = (char)('0' + r % 10);
        out[3] = '\0';
    } else {
        out[1] = (char)('0' + r);
        out[2] = '\0';
    }
}

int go_parse_gtp(const State *s, const char *v, Move *out) {
    if (!v || !*v)
        return -1;
    if (strcmp(v, "pass") == 0) {
        *out = MOVE_PASS;
        return 0;
    }
    const char *col = strchr(GO_COLS, toupper((unsigned char)v[0]));
    if (!col || v[1] < '0' || v[1] > '9')
        return -1;
    int x = (int)(col - GO_COLS), y = atoi(v + 1);
    if (x >= s->b || y < 1 || y > s->b)
        return -1;
    *out = (Move)((y - 1) * s->b + x);
    return 0;
}
