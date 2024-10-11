#!/bin/bash

# Array of horizons to test
horizons=(3 6 12)

# Base command
base_command="python darts_experiments.py --algorithm Block_GRU --params"

# Loop through horizons
for horizon in "${horizons[@]}"
do
    echo "Running experiment with horizon $horizon"

    # Construct the JSON parameters with dynamic output_chunk_length
    json_params=$(printf '{"input_chunk_length": 50, "output_chunk_length": %d}' "$horizon")

    # Construct the full command
    full_command="$base_command '$json_params' --horizon $horizon"

    # Execute the command
    eval $full_command

    echo "Experiment with horizon $horizon completed"
    echo "----------------------------------------"
done

echo "All experiments completed"