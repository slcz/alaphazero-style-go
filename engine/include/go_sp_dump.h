/*
 * go_sp_dump.h — GOSHARD1 self-play shard writer (go_sp_dump.c).
 *
 * Efficiency rules (this file IS the network/training-time contract):
 * moves are u16 points (never strings — no GTP parsing at ingest), π is
 * dense f16 over LEGAL moves only in engine movegen order, ownership and
 * the final score live on the GAME (they are per-game labels, not per
 * ply), and no position data is stored — the trainer replays u16 moves
 * from the empty board through the C rules + encoder via cffi. There is
 * no start-position field: self-play games always start empty.
 *
 * THE FILE FORMAT (little-endian)
 * =====================================================================
 * Shard = 24-byte header, then finished games concatenated back to back,
 * one file per worker (selfplay writes worker 0 only):
 *
 *   shard header, 24 B:
 *     char[8]  magic      "GOSHARD1"
 *     u32      version    1
 *     u32      worker     writer index
 *     u64      unit_seed  the run's seed (ingest dedups (seed, game_index))
 *
 *   game record (self-delimiting: a reader consumes exactly n_plies ply
 *   blocks, so a run killed mid-write leaves only a truncated LAST record
 *   the reader detects as a short read and drops):
 *     u32      game_index
 *     u32      n_plies
 *     u8       b            board size (game started empty, komi below)
 *     u8       end_kind     SP_END_* (NATURAL = double pass, ADJUDICATED
 *                            = ply cap)
 *     u16      n_sims       the slow search budget (the training cap under
 *                            playout cap randomization; a ply trains iff
 *                            its sims == n_sims)
 *     f32      komi
 *     f32      score        final TT margin, black view (> 0 black wins;
 *                            never 0 in selfplay — fractional komi)
 *     b*b ×    f16 ownership  per point, black view {0, 0.5, 1}
 *     n_plies × ply block:
 *       u8      mover       BLACK / WHITE
 *       u16     move        board point or 0xFFFF (pass)
 *       u16     n_pi        == the legal-move count at this ply
 *       u16     sims        this ply's actual search budget
 *       f32     outlook     root value in the mover's view (fast-play
 *                            gating telemetry + loss weighting input)
 *       n_pi ×  f16 prob    π̄ over the legal moves, ENGINE MOVEGEN ORDER
 *
 * π order contract (load-bearing): gen_legal emits board points in index
 * order, then pass — every consumer replays through the same movegen and
 * pairs the k-th prob with the k-th legal move, asserting n_pi == the
 * legal count. f16 is enough for a [0,1] target: rel err ≤ ~5e-4 into a
 * bf16-autocast CE loss.
 */
#ifndef GO_SP_DUMP_H
#define GO_SP_DUMP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SP_PI_MAX  362   /* ≤ MAX_MOVES; u16 in file */
#define SP_VERSION 1

enum {
    SP_END_PLAYING = 0,  /* never written — reader-side state only */
    SP_END_NATURAL,      /* double pass */
    SP_END_RESIGN,       /* reserved: eval-only, never in selfplay shards */
    SP_END_ADJUDICATED   /* maxply reached */
};

/* One ply as the driver records it, before serialization. π̄ is dense
 * over the legal moves in engine movegen order. */
typedef struct SpPlyRec {
    uint8_t  mover;
    uint16_t move;
    uint16_t n_pi;
    uint16_t sims;
    float    outlook;
    float    pi_prob[SP_PI_MAX];
} SpPlyRec;

/* One finished game. */
typedef struct SpGameRec {
    uint32_t game_index;
    uint32_t n_plies;
    uint8_t  b;
    uint8_t  end_kind;
    uint16_t n_sims;
    float    komi;
    float    score;
    const float *ownership;      /* b*b entries, black view */
    const SpPlyRec *plies;
} SpGameRec;

/* "shard-<worker>.spbin" inside `dir`. */
void sp_dump_shard_path(const char *dir, int worker, char *buf, size_t cap);

/* Create `dir` if needed, open its shard with O_TRUNC (a shard belongs to
 * exactly one run) and write the header. Returns the fd or -1. */
int  sp_dump_open_shard(const char *dir, int worker, uint64_t unit_seed);

/* Streaming writes: game head (with ownership), then each ply in order.
 * 0 ok, -1 on any failed/short write — the caller fails the run. */
int  sp_dump_game_head(int fd, const SpGameRec *g);
int  sp_dump_ply(int fd, const SpPlyRec *r);

/* Write DONE.json into `dir` (the trainer's unit marker). */
int  sp_dump_write_done(const char *dir, int n_workers, uint64_t unit_seed,
                        uint32_t games_done);

#ifdef __cplusplus
}
#endif

#endif /* GO_SP_DUMP_H */
