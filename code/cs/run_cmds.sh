#!/bin/bash
# Runs one shell command per line of a file, JOBS at a time (default: number of CPU cores),
# each single-threaded. A line whose output prediction file already exists is skipped by train.py's caller.
# usage: JOBS=32 bash code/cs/run_cmds.sh code/cs/queue_seeds.txt
cd "$(dirname "$0")/../.."
mkdir -p results/logs
JOBS=${JOBS:-$(nproc)}
grep -v '^\s*#' "$1" | grep -v '^\s*$' | xargs -P "$JOBS" -I{} bash -c '{}'
