#!/bin/bash

# Array of horizons to run
horizons=(1 3 6 12)

# Number of parallel instances to run
max_parallel=60

# Function to run AutoArima.py
run_autoarima() {
    horizon=$1
    python AutoArima.py --horizon $horizon
    echo "Completed horizon $horizon"
}

# Export the function so it's available to parallel
export -f run_autoarima

# Run the function in parallel for each horizon
parallel -j $max_parallel run_autoarima ::: "${horizons[@]}"

echo "All instances completed
