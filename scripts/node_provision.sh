#!/bin/bash
# node_provision.sh — bring a fresh node to the point where the loop can
# run: source, deps, engine build, checkpoint and loop.jsonl restore.
#
#   bash scripts/node_provision.sh --ckpt ckpts/it006.pt --iters 10
#
# --ckpt   local checkpoint to install as best.pt (the warm start)
# --iters  how many iterations to mark done in loop.jsonl (resume after)
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
CKPT=ckpts/it006.pt
ITERS=10
while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt) CKPT=$2; shift 2 ;;
        --iters) ITERS=$2; shift 2 ;;
        --skip-pip) SKIP_PIP=1; shift ;;
        *) echo "unknown arg $1"; exit 2 ;;
    esac
done
say() { printf '\n=== %s\n' "$*"; }

say "1/5 source"
bash scripts/ship_src.sh
scripts/salad.sh put /tmp/go-src.tgz /workspace/go-src.tgz >/dev/null 2>&1
scripts/salad.sh ssh 'mkdir -p /workspace/go && cd /workspace/go &&
    tar xzf /workspace/go-src.tgz && echo extracted'

if [ -z "$SKIP_PIP" ]; then
    say "2/5 deps (torch cu128, backgrounded on the node)"
    scripts/salad.sh ssh 'rm -f /tmp/pip.done /tmp/pip.result;
        cd /workspace/go && nohup bash scripts/node_pip.sh > /tmp/pip.log 2>&1 &
        echo started' >/dev/null 2>&1 || true
fi

say "3/5 engine build"
scripts/salad.sh ssh 'cd /workspace/go && make -C engine all nn 2>&1 |
    grep -cE "\berror\b" | sed "s/^/build errors: /"'

if [ -z "$SKIP_PIP" ]; then
    say "4/5 waiting for deps"
    for i in $(seq 1 120); do
        if scripts/salad.sh ssh 'test -f /tmp/pip.done && cat /tmp/pip.result' \
                2>/dev/null | grep -q '^torch'; then
            break
        fi
        sleep 15
    done
    scripts/salad.sh ssh 'cat /tmp/pip.result' 2>/dev/null | tail -2
fi

say "5/5 restore checkpoint + loop.jsonl"
scripts/salad.sh put "$CKPT" /workspace/go-run-9x9/best.pt >/dev/null 2>&1
scripts/salad.sh ssh "cd /workspace/go && python3 -c \"
import json, pathlib
p = pathlib.Path('/workspace/go-run-9x9/loop.jsonl')
p.parent.mkdir(parents=True, exist_ok=True)
with p.open('w') as f:
    for i in range(1, $ITERS + 1):
        f.write(json.dumps({'iteration': i, 'note': 'restored stub'}) + '\n')
print('loop resumes at it%03d' % ($ITERS + 1))
\""
echo
echo "ready: bash scripts/node_run9x9_tmux.sh"
