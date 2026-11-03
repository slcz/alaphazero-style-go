#!/bin/bash
python3 -m pip install --quiet py-spy 2>&1 | tail -1
P=$(pgrep -f selfplay_nn | head -1)
echo "pid $P"
echo "== fresh inductor compiles after 23:43? =="
find /tmp/torchinductor_root -maxdepth 1 -newermt "23:43" 2>/dev/null | head -5
ls /tmp/torchinductor_root 2>/dev/null | wc -l
echo "== py-spy dump (3 samples) =="
py-spy dump --pid $P 2>&1 | head -30
echo "== second sample =="
sleep 3
py-spy dump --pid $P 2>&1 | grep -E "Thread|push|pull|wait|forward|compiled|fp8|to_fp8|copy_|synchronize" | head -15
