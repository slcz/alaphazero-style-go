#!/bin/bash
L=/workspace/go-run-9x9/it001/eval.log
A=$(grep -ac '^GAME' $L)
T0=$(date +%s)
sleep 60
B=$(grep -ac '^GAME' $L)
T1=$(date +%s)
echo "games: $A -> $B in $((T1-T0))s = $(( (B-A) * 60 / (T1-T0) )) games/min"
tail -2 $L
