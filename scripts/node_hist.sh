#!/bin/bash
tail -n 8 /root/.bash_history 2>/dev/null
echo ---
grep -c "node_cleanup" /root/.bash_history 2>/dev/null
echo ---
ps -eo pid,etime,comm | grep -vE "Z|ps$|grep$" | tail -12
