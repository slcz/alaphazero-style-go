#!/bin/bash
# node_neturl.sh — find the human-SL b18 net URL from katagotraining.org.
curl -s "https://katagotraining.org/networks/" | grep -oE 'href="[^"]*humanv0[^"]*\.bin\.gz"' | head -4
echo ---
curl -s "https://katagotraining.org/api/networks/newest_training/" | head -c 400
