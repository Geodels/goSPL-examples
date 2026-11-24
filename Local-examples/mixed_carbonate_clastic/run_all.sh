#!/bin/bash
# Run every experiment of the mixed_carbonate_clastic example (6 ranks each,
# two runs at a time; 3-5 min per run on a laptop). --summary writes a per-step
# record with the carbonate budget.
cd "$(dirname "$0")"
run() { mpirun -np 6 gospl -i input-$1.yml --summary summary_$1.jsonl > log_$1.txt 2>&1; echo "done $1"; }
run clastic & run mixed & wait
run nocontrol & run waves & wait
run gravity & run wavecontrol & wait
run sedrate
