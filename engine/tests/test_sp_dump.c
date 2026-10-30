/* test_sp_dump.c — GOSHARD1 writer round-trip: an independent reader in
 * this test parses the bytes back (format as spec'd in go_sp_dump.h),
 * including the kill-mid-write truncated-tail behavior. */
#include "go_sp_dump.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails, checks;

#define CHECK(cond) do {                                          \
        checks++;                                                 \
        if (!(cond)) {                                            \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            fails++;                                              \
        }                                                         \
    } while (0)

/* f16 → f32 (the decode direction the trainer's numpy would do) */
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

typedef struct {
    uint32_t game_index, n_plies;
    uint8_t b, end_kind;
    uint16_t n_sims;
    float komi, score;
    float ownership[SP_PI_MAX];
    struct { uint8_t mover; uint16_t move, n_pi, sims; float outlook;
             float prob[SP_PI_MAX]; } plies[SP_PI_MAX];
} RGame;

/* Parse one game record; -1 on short read (the kill case), -2 on bad. */
static int read_game(FILE *f, RGame *g) {
    uint16_t own16[SP_PI_MAX], p16[SP_PI_MAX];
    if (read_exact(f, &g->game_index, 4) || read_exact(f, &g->n_plies, 4) ||
        read_exact(f, &g->b, 1) || read_exact(f, &g->end_kind, 1) ||
        read_exact(f, &g->n_sims, 2) || read_exact(f, &g->komi, 4) ||
        read_exact(f, &g->score, 4))
        return -1;
    int n = g->b * g->b;
    if (read_exact(f, own16, (size_t)n * 2))
        return -1;
    for (int i = 0; i < n; i++)
        g->ownership[i] = f16_to_f32(own16[i]);
    if (g->n_plies > SP_PI_MAX)
        return -2;
    for (uint32_t p = 0; p < g->n_plies; p++) {
        if (read_exact(f, &g->plies[p].mover, 1) ||
            read_exact(f, &g->plies[p].move, 2) ||
            read_exact(f, &g->plies[p].n_pi, 2) ||
            read_exact(f, &g->plies[p].sims, 2) ||
            read_exact(f, &g->plies[p].outlook, 4) ||
            read_exact(f, p16, (size_t)g->plies[p].n_pi * 2))
            return -1;
        for (int i = 0; i < g->plies[p].n_pi; i++)
            g->plies[p].prob[i] = f16_to_f32(p16[i]);
    }
    return 0;
}

static void fill_game(SpGameRec *g, SpPlyRec *plies, const float *ownership,
                      uint32_t index, uint8_t b, int n_plies, uint16_t sims,
                      float komi, float score, uint8_t end) {
    memset(plies, 0, (size_t)n_plies * sizeof *plies);
    g->game_index = index;
    g->n_plies = (uint32_t)n_plies;
    g->b = b;
    g->end_kind = end;
    g->n_sims = sims;
    g->komi = komi;
    g->score = score;
    g->ownership = ownership;
    g->plies = plies;
    for (int p = 0; p < n_plies; p++) {
        plies[p].mover = (uint8_t)(p % 2 == 0 ? 1 : 2);
        plies[p].move = (uint16_t)(p == n_plies - 1 ? 0xFFFF : p * 7 % (b * b));
        plies[p].n_pi = (uint16_t)(b * b - p + 1);   /* shrinking legals */
        plies[p].sims = (uint16_t)(p % 3 == 0 ? sims : 24);
        plies[p].outlook = (float)p / 100.0f;
        for (int i = 0; i < plies[p].n_pi; i++)
            plies[p].pi_prob[i] = 1.0f / plies[p].n_pi;
    }
}

int main(void) {
    const char *dir = "build/test_shard";
    uint64_t seed = 0xDEADBEEFCAFEULL;

    float ownership[SP_PI_MAX];
    for (int i = 0; i < 49; i++)
        ownership[i] = i % 3 == 0 ? 1.0f : (i % 3 == 1 ? 0.5f : 0.0f);
    SpPlyRec plies[80];
    SpGameRec g1, g2;
    fill_game(&g1, plies, ownership, 0, 7, 40, 300, 9.5f, -3.5f,
              SP_END_NATURAL);
    fill_game(&g2, plies, ownership, 1, 7, 12, 300, 9.5f, 12.0f,
              SP_END_ADJUDICATED);

    int fd = sp_dump_open_shard(dir, 0, seed);
    CHECK(fd >= 0);
    if (fd < 0) {
        printf("cannot open shard in %s\n", dir);
        return 1;
    }
    CHECK(sp_dump_game_head(fd, &g1) == 0);
    for (int p = 0; p < 40; p++)
        CHECK(sp_dump_ply(fd, &plies[p]) == 0);
    CHECK(sp_dump_game_head(fd, &g2) == 0);
    for (int p = 0; p < 12; p++)
        CHECK(sp_dump_ply(fd, &plies[p]) == 0);
    close(fd);
    CHECK(sp_dump_write_done(dir, 1, seed, 2) == 0);

    /* expected byte size: header + game(head + b*b*2) + plies(11 + 2*n_pi) */
    long want = 24;
    want += 20 + 49 * 2 + 40 * 11;
    for (int p = 0; p < 40; p++) want += 2 * (49 - p + 1);
    want += 20 + 49 * 2 + 12 * 11;
    for (int p = 0; p < 12; p++) want += 2 * (49 - p + 1);
    char path[512];
    sp_dump_shard_path(dir, 0, path, sizeof path);
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    fseek(f, 0, SEEK_END);
    CHECK(ftell(f) == want);
    rewind(f);

    char magic[8];
    struct { uint32_t version, worker; uint64_t s; } hdr;
    CHECK(read_exact(f, magic, 8) == 0 && memcmp(magic, "GOSHARD1", 8) == 0);
    CHECK(read_exact(f, &hdr, sizeof hdr) == 0);
    CHECK(hdr.version == SP_VERSION && hdr.worker == 0 && hdr.s == seed);

    RGame r;
    CHECK(read_game(f, &r) == 0);
    CHECK(r.game_index == 0 && r.n_plies == 40 && r.b == 7);
    CHECK(r.end_kind == SP_END_NATURAL && r.n_sims == 300);
    CHECK(r.komi == 9.5f && r.score == -3.5f);
    for (int i = 0; i < 49; i++)
        CHECK(r.ownership[i] == ownership[i]);   /* {0,.5,1} exact in f16 */
    for (int p = 0; p < 40; p++) {
        CHECK(r.plies[p].mover == plies[p].mover);
        CHECK(r.plies[p].move == plies[p].move);
        CHECK(r.plies[p].n_pi == plies[p].n_pi);
        CHECK(r.plies[p].sims == plies[p].sims);
        CHECK(r.plies[p].outlook == plies[p].outlook);
        float sum = 0;
        for (int i = 0; i < r.plies[p].n_pi; i++) {
            CHECK(r.plies[p].prob[i] > 0);
            sum += r.plies[p].prob[i];
        }
        CHECK(sum > 0.99f && sum < 1.01f);       /* uniform survives f16 */
    }
    CHECK(read_game(f, &r) == 0);
    CHECK(r.game_index == 1 && r.n_plies == 12 && r.end_kind ==
          SP_END_ADJUDICATED && r.score == 12.0f);
    CHECK(read_game(f, &r) == -1);               /* EOF: no third game */
    fclose(f);

    /* kill-mid-write: a truncated tail yields the complete games only.
     * Cut inside game 2's second ply block (mid π array). */
    FILE *src = fopen(path, "rb");
    char *buf = malloc((size_t)want);
    CHECK(fread(buf, 1, (size_t)want, src) == (size_t)want);
    fclose(src);
    long g2_start = 24 + 20 + 49 * 2 + 40 * 11;
    for (int p = 0; p < 40; p++)
        g2_start += 2 * (49 - p + 1);
    long trunc = g2_start + 20 + 49 * 2 + 11 + 2 * 30 + 3;
    char tpath[600];
    snprintf(tpath, sizeof tpath, "%s/trunc.spbin", dir);
    FILE *tf = fopen(tpath, "wb");
    fwrite(buf, 1, (size_t)trunc, tf);
    fclose(tf);
    free(buf);
    tf = fopen(tpath, "rb");
    fseek(tf, 24, SEEK_SET);
    CHECK(read_game(tf, &r) == 0);               /* game 1 complete */
    CHECK(read_game(tf, &r) == -1);              /* game 2 truncated */
    fclose(tf);
    unlink(tpath);

    printf("%s: %d checks, %d failures\n",
           fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
