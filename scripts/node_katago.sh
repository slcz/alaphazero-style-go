#!/bin/bash
# node_katago.sh — fetch KataGo v1.18.2 cuda12.8 binary + human-SL net.
set -e
mkdir -p /workspace/katago && cd /workspace/katago
rm -f katago.tar.gz
if [ ! -x katago ]; then
    curl -sL -o katago.zip \
        "https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip"
    python3 -c "import zipfile; zipfile.ZipFile('katago.zip').extractall('.')"
    find . -name katago -type f -exec mv {} katago \; -quit
fi
chmod +x katago
./katago version && echo KATAGO-OK
if [ ! -s human1.bin.gz ]; then
    curl -sL -o human1.bin.gz \
        "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c406nbt-humanv0-bin.bin.gz"
fi
ls -la /workspace/katago
