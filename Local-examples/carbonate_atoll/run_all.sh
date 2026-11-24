#!/bin/bash
# Run every experiment of the carbonate_atoll example (4 ranks each, three
# runs at a time; 3-5 min per run on a laptop). --summary writes a per-step
# record with the carbonate and coastal-erosion budgets.
cd "$(dirname "$0")"
run() { mpirun -np 4 gospl -i input-$1.yml --summary summary_$1.jsonl > log_$1.txt 2>&1; echo "done $1"; }
run nowaves & run ring & run east & wait
run season & run refraction & run erosion & wait
run sealevel
