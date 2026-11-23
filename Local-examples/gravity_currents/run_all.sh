#!/bin/bash
# Run the margin without gravity currents, with them, and with them plus slope
# failure (4 ranks each, about 2 min each on a laptop). --summary writes a
# per-step record, including the gravity budget.
cd "$(dirname "$0")"
run() { mpirun -np 4 gospl -i input-$1.yml --summary summary_$1.jsonl > log_$1.txt 2>&1; echo "done $1"; }
run gravity & run nogravity & run failure & wait
