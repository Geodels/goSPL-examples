#!/bin/bash
# Run the global model with and without gravity currents for 1 Myr (20 steps).
# Needs the inputs of ../erodep_1My (run its build_inputs.ipynb first).
# --summary writes a per-step record, including the gravity budget.
cd "$(dirname "$0")"
NP=${NP:-4}
run() { mpirun -np $NP gospl -i input-$1.yml --summary summary_$1.jsonl > log_$1.txt 2>&1; echo "done $1"; }
run gravity & run nogravity & wait
