#!/bin/bash
# Run the margin with and without gravity currents (4 ranks each, ~3 min each
# on a laptop). --summary writes a per-step record, including the gravity budget.
cd "$(dirname "$0")"
run() { mpirun -np 4 gospl -i input-$1.yml --summary summary_$1.jsonl > log_$1.txt 2>&1; echo "done $1"; }
run gravity & run nogravity & wait
