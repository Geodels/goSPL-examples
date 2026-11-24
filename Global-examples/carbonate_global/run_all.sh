#!/bin/bash
# Run the four experiments of the carbonate_global example (8 MPI ranks each;
# 20-40 min per run on a laptop, the geochemistry run being the longest).
# --summary writes a per-step record with the carbonate, pelagic and
# carbonate-supply budgets.
cd "$(dirname "$0")"
run() { mpirun -np 8 gospl -i input-$1.yml --summary summary_$1.jsonl > log_$1.txt 2>&1; echo "done $1"; }
run carb
run ccd_shallow
run waves
run geochem
