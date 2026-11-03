#!/bin/bash
grep -a "WAVES" /workspace/logs/probe.raw | tail -6
echo "--- summary-ish ---"
grep -a "GAME" /workspace/logs/probe.raw | tail -2
