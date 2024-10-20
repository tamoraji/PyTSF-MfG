#!/bin/bash

# Array of horizons to test
horizons=(3 6 12)

# Base command
base_command="python neuralforecast_experiments.py --algorithm PatchTST --mode multivariate"

# Loop through horizons
for horizon in "${horizons[@]}"
do
    echo "Running experiment with horizon $horizon"
    
    # Construct the full command
    full_command="$base_command --horizon $horizon"
    
    # Execute the command
    eval $full_command
    
    echo "Experiment with horizon $horizon completed"
    echo "----------------------------------------"
done

echo "All experiments completed"
