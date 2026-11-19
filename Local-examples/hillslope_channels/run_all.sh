#!/bin/bash
# Run the six simulations (adjust -np to your machine; ~15 min on 16 cores,
# most of it the two 50 m runs).
cd "$(dirname "$0")"
run() { mpirun -np $2 gospl -i input-$1.yml > log_$1.txt 2>&1; echo "done $1"; }
run 200 2 & run 200_chan 2 & run 100 5 & run 100_chan 5 & wait
run 50 8 & run 50_chan 8 & wait
