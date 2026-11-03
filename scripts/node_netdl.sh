#!/bin/bash
# node_netdl.sh — download the human-SL b18 net (correct filename).
set -e
cd /workspace/katago
curl -sL -o human1.bin.gz \
    "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c406nbt-humanv0.bin.gz"
ls -la human1.bin.gz
./katago version | head -3
