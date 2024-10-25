#!/bin/bash

# Array of horizons to test
horizons=(3 6 12 96 288 672)

# Array of algorithms to test
algorithms=("AutoARIMA")

# Base command
base_command="python AutoArima.py --mode multivariate"

# Loop through algorithms
for algorithm in "${algorithms[@]}"
do
    echo "Starting experiments for algorithm: $algorithm"
    echo "=========================================="

    # Loop through horizons for each algorithm
    for horizon in "${horizons[@]}"
    do
        echo "Running experiment with $algorithm, horizon $horizon"

        # Construct the full command
        full_command="$base_command --algorithm $algorithm --horizon $horizon"

        # Execute the command
        eval $full_command

        echo "Experiment with $algorithm, horizon $horizon completed"
        echo "----------------------------------------"
    done

    echo "All horizons completed for $algorithm"
    echo "==========================================\n"
done

echo "All experiments completed successfully"
