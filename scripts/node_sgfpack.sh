#!/bin/bash
# node_sgfpack.sh — tar the gate SGF dir for scp transfer.
tar czf /workspace/go-run-9x9/it001/gate-sgf.tgz -C /workspace/go-run-9x9/it001 gate-sgf
ls -la /workspace/go-run-9x9/it001/gate-sgf.tgz
