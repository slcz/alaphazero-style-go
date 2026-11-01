#!/bin/bash
# node_restore_hf.sh — fresh instance: pull the 9x9 artifacts from HF
# straight onto the node and place them under $W (it001/it002 restored,
# best.pt = it002). Run on the node: bash scripts/node_restore_hf.sh
set -e
cd /workspace/go
set -a; source .env; set +a
: "${HF_TOKEN:?HF_TOKEN missing}"
B=https://huggingface.co/chriszhang01/go-value-net/resolve/main/9x9
D=https://huggingface.co/datasets/chriszhang01/go-selfplay/resolve/main/9x9
W=/workspace/go-run-9x9
mkdir -p "$W/it001/selfplay" "$W/it002/selfplay"

dl() {
    curl -sL --fail -H "Authorization: Bearer $HF_TOKEN" "$1" -o "$2" \
        || { echo "FAILED $1"; exit 1; }
    printf '  %-10s %s\n' "$(du -h "$2" | cut -f1)" "$(basename "$2")"
}

echo "== downloads =="
dl "$B/it001-train.pt"      "$W/it001/train.pt"
dl "$B/it002-train.pt"      "$W/it002/train.pt"
dl "$B/it001-eval.json"     "$W/it001/eval.json"
dl "$B/it002-summary.json"  /tmp/it002-summary.json
curl -sL --fail -H "Authorization: Bearer $HF_TOKEN" \
    "$B/buffer.json" -o "$W/buffer.json" \
    && echo "  buffer.json restored" || echo "  no buffer.json on HF"
dl "$D/it001/shard-0.spbin" "$W/it001/selfplay/shard-0.spbin"
dl "$D/it002/shard-0.spbin" "$W/it002/selfplay/shard-0.spbin"

echo "== it002 gate result =="
python3 - <<'PY'
import json
from pathlib import Path
s = json.loads(Path('/tmp/it002-summary.json').read_text())
g = s['gate']
g.setdefault('ckpt_a', '/workspace/go-run-9x9/it002/train.pt')
g.setdefault('ckpt_b', '/workspace/go-run-9x9/it001/train.pt')
Path('/workspace/go-run-9x9/it002/eval.json').write_text(json.dumps(g, indent=2))
print('  score_a', g['score_a'], 'games', g['games'])
PY

echo "== markers =="
printf '{"n_workers":1,"unit_seed":2,"games_done":30000}\n' \
    > "$W/it001/selfplay/DONE.json"
printf '{"n_workers":1,"unit_seed":3,"games_done":30000}\n' \
    > "$W/it002/selfplay/DONE.json"
cp "$W/it002/train.pt" "$W/best.pt"

echo "== verify =="
sha256sum "$W/it001/train.pt" "$W/it002/train.pt" "$W/best.pt"
for d in it001 it002; do
    n=$(python3 -c "
import sys; sys.path.insert(0,'/workspace/go')
from go_ai.selfplay.shard import read_unit
print(len(read_unit('$W/$d/selfplay')))")
    echo "  $d shard games=$n"
done
