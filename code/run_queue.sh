#!/bin/bash
# Runs "model seed" lines from a queue file, two single-threaded jobs at a time.
# usage: bash run_queue.sh queue.txt
cd "$(dirname "$0")/.."
mkdir -p results/logs
cat "$1" | xargs -P 2 -L 1 bash -c 'if [ ! -f results/preds/${0}_s${1}.pt ]; then THREADS=1 python3 code/train.py --model $0 --seed $1 > results/logs/${0}_s${1}.log 2>&1; fi'
