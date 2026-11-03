#!/bin/bash
set -e
ls /workspace/go-run-9x9/it002/gate-sgf | wc -l
tar czf /workspace/go-run-9x9/it002/gate-sgf.tgz -C /workspace/go-run-9x9/it002 gate-sgf
ls -la /workspace/go-run-9x9/it002/gate-sgf.tgz
