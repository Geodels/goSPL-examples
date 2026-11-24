#!/bin/bash
# Run every experiment of the carbonate_terraces example (4 ranks each, two
# runs at a time; 30 s to 2 min per run on a laptop). --summary writes a
# per-step record with the carbonate and coastal-erosion budgets.
cd "$(dirname "$0")"
run() { mpirun -np 4 gospl -i input-$1.yml --summary summary_$1.jsonl > log_$1.txt 2>&1; echo "done $1"; }
run res100 & run res200 & wait
run res050 & run stationary & wait
run subsidence & run sub_reefonly & wait
run sub_waves & run erosion10 & wait
