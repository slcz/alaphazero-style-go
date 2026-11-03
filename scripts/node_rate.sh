#!/bin/bash
L=/workspace/go-run-9x9/it001/generate.log
A=$(grep -c '^GAME' $L)
sleep 60
B=$(grep -c '^GAME' $L)
echo "games: $A -> $B  rate=$((B-A)) games/min"
grep -a 'WAVES' $L | tail -1
grep -a '^GAME' $L | tail -3
awk '{for(i=1;i<=NF;i++) if($i ~ /^plies=/){split($i,a,"=");s+=a[2];n++}}
     END{if(n) printf "mean_plies(finished)=%.0f over %d games\n", s/n, n}' \
    <(grep -a '^GAME' $L)
