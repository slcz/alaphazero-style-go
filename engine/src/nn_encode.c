/* nn_encode.c — input-feature encoder (go_nn.h). Pure read over State. */
#include "go_nn.h"

#include <string.h>

void nn_encode(const State *s, float out[NN_TOTAL]) {
    const int b = s->b, n = b * b;
    memset(out, 0, (size_t)n * NN_TOKEN_DIM * sizeof *out);

    for (int y = 0; y < b; y++)
        for (int x = 0; x < b; x++) {
            float *row = out + (size_t)(y * b + x) * NN_TOKEN_DIM;
            int c = s->board[y * b + x];
            if (c == s->side)
                row[GNN_OWN] = 1.0f;
            else if (c != EMPTY)
                row[GNN_OPP] = 1.0f;
            row[GNN_EDGE]     = (float)x / b;
            row[GNN_EDGE + 1] = (float)(b - 1 - x) / b;
            row[GNN_EDGE + 2] = (float)y / b;
            row[GNN_EDGE + 3] = (float)(b - 1 - y) / b;
        }

    float passflag[GO_HIST] = {0};
    for (int h = 0; h < GO_HIST; h++) {
        int ply = s->ply - 1 - h;      /* ply the h-th-back move was played */
        if (ply < 0)
            break;
        uint16_t m = s->hist[ply % GO_HIST];
        if (m == MOVE_PASS)
            passflag[h] = 1.0f;
        else
            out[(size_t)m * NN_TOKEN_DIM + GNN_LASTM + h] = 1.0f;
    }

    for (int p = 0; p < n; p++) {
        float *row = out + (size_t)p * NN_TOKEN_DIM;
        for (int h = 0; h < GO_HIST; h++)
            row[GNN_PASS + h] = passflag[h];
        row[GNN_KOMI] = s->komi / 15.0f;
        row[GNN_SIZE] = (float)b / 19.0f;
    }
}
