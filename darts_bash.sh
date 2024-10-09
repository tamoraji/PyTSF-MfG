#!/bin/bash

# Array of horizons to test
horizons=(3 6 12)

# Base command
base_command="python darts_experiments.py --algorithm LSTM --params"

# JSON parameters
json_params='{"input_chunk_length": 50, "training_length": 50}'

# Loop through horizons
for horizon in "${horizons[@]}"
do
    echo "Running experiment with horizon $horizon"

    # Construct the full command
    full_command="$base_command '$json_params' --horizon $horizon"

    # Execute the command
    eval $full_command

    echo "Experiment with horizon $horizon completed"
    echo "----------------------------------------"
done

echo "All experiments completed"