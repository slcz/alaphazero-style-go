#!/bin/bash
# node_katago3.sh — list real KataGo release assets (latest, linux).
curl -s https://api.github.com/repos/lightvector/KataGo/releases/latest \
    | grep -E '"tag_name"|browser_download_url' | grep -iE "tag_name|linux" | head -12
