#!/bin/bash
# node_katago2.sh — diagnose the failed download; try the other asset URL.
cd /workspace/katago
file katago.tar.gz; head -c 200 katago.tar.gz; echo
curl -sIL https://github.com/lightvector/KataGo/releases/download/v1.16.0/katago-v1.16.0-cuda12.4-linux-x64.tar.gz | grep -iE "HTTP|location" | head -6
