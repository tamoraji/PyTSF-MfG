import argparse
import json
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast

from modules.utils import load_datasets_statforecast_uni
from modules.evaluator import Evaluator
from modules.results_saver import ResultsSaver
from modules.performance_utils import measure_time_and_memory
from modules.algorithm_factory import create_algorithm
from modules.config import ALGORITHM_POOL, DATASET_POOL

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set parameters
DATA_PATH = '/Users/moji/PyTSF-MfG/data'  # change this in your machine
OUTPUT_DIR = '/Users/moji/PyTSF-MfG/results'

# DATA_PATH = '/home/ma00048/Moji/TSF_data'  # change this in your machine
# OUTPUT_DIR = '/home/ma00048/Moji/TSF_results'


@measure_time_and_memory
def train_model(model, data):
    logger.info(f"Training model on data of shape: {data.shape}")
    return model.fit(data)


@measure_time_and_memory
def test_model(model, test_data):
    logger.info(f"Predicting for horizon: {len(test_data)}")
    return model.predict(futr_df=test_data)


def run_experiment(data, name, horizon, algorithm_name, algorithm_params):
    logger.info(f"\nStarting experiment for dataset: {name}")
    logger.info(f"Algorithm: {algorithm_name}, Horizon: {horizon}")
    logger.info(f"Initial data shape: {data.shape}")

    # Get dataset-specific configuration
    dataset_config = DATASET_POOL.get(name, {})
    frequency = dataset_config.get('frequency', 'h')

    # Split the data
    split_ratio = 0.9
    split_index = int(len(data) * split_ratio)
    train = data[:split_index]
    test = data[split_index:]

    # Ensure test set length is a multiple of horizon
    test_length = (len(test) // horizon) * horizon
    test = test[:test_length]

    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Create the model using the factory
    model = create_algorithm(algorithm_name, algorithm_params,horizon=horizon)
    logger.info(f"Model created: {type(model).__name__}")

    # Wrap the model in NeuralForecast
    nf = NeuralForecast(models=[model], freq=frequency)

    # Train the model
    _, train_time, train_memory = train_model(nf, train)
    logger.info(f"Model trained. Time: {train_time:.2f}s, Memory: {train_memory:.2f}MB")

    # Perform autoregressive prediction
    predictions = []
    history = train.copy()
    test_time = 0
    test_memory = 0

    logger.info("Starting autoregressive prediction")
    for i in range(0, len(test), horizon):
        test_chunk = test.iloc[i:i+horizon]
        # logger.info(f"Test chunk:\n{test_chunk}")
        logger.info(f"Predicting for dates: {test_chunk['ds'].iloc[0]} to {test_chunk['ds'].iloc[-1]}")

        forecasts, t_time, t_memory = test_model(nf, test_chunk)
        logger.info(f"Forecasts:\n{forecasts}")
        test_time += t_time
        test_memory += t_memory

        predictions.append(forecasts)

        history = pd.concat([history, test_chunk])

        # Retrain the model with updated history
        nf = NeuralForecast(models=[model], freq=frequency)
        _, retrain_time, retrain_memory = train_model(nf, history)
        train_time += retrain_time
        train_memory += retrain_memory

        logger.info(f"Prediction step {i // horizon + 1}: predicted {len(forecasts)} values")

    logger.info("Autoregressive prediction completed")

    # Combine predictions
    combined_predictions = pd.concat(predictions)

    # Calculate metrics

    actual = test['y'].values
    forecast = combined_predictions[f'{algorithm_name}'].values
    metrics = Evaluator.calculate_metrics(actual, forecast)

    # Add computational complexity metrics
    metrics['train_time'] = train_time
    metrics['avg_test_time'] = test_time / (len(test) / horizon)
    metrics['train_memory'] = train_memory
    metrics['avg_test_memory'] = test_memory / (len(test) / horizon)
    metrics['total_time'] = train_time + test_time
    metrics['total_memory'] = train_memory + test_memory

    logger.info("Experiment completed")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run time series forecasting experiment with specified algorithm and forecasting horizon')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name (e.g., TimesNet)')
    parser.add_argument('--horizon', type=int, default=12, help='Forecasting horizon')
    parser.add_argument('--params', type=str, default='{}', help='JSON string of algorithm parameters')
    parser.add_argument('--datasets', nargs='*', help='List of specific datasets to process. If not provided, all datasets will be processed.')
    args = parser.parse_args()

    # Parse the JSON string of parameters
    try:
        algorithm_params = json.loads(args.params)
        print(algorithm_params)
    except json.JSONDecodeError:
        print("Error: Invalid JSON string for parameters")
        exit(1)

    print(f"Starting forecasting experiment with {args.algorithm} algorithm and horizon {args.horizon}")
    print(f"Algorithm parameters: {algorithm_params}")

    datasets = load_datasets_statforecast_uni(DATA_PATH)
    print(f"Loaded {len(datasets)} datasets")

    # Filter datasets if specific ones are provided
    if args.datasets:
        datasets = {name: data for name, data in datasets.items() if name in args.datasets}
        print(f"Filtered to {len(datasets)} specified datasets")

    saver = ResultsSaver(OUTPUT_DIR)

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        print("Dataset preview:")
        print(data.head())

        try:
            metrics = run_experiment(data, name, args.horizon, args.algorithm, algorithm_params)

            # Save results
            saver.save_results({f'horizon_{args.horizon}': metrics}, args.algorithm, args.horizon, name)
            print(f"Results saved for {args.algorithm} on {name}")

            # Print summary of results
            print(f"Summary of results for {name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        except Exception as e:
            print(f"Error processing dataset {name} with algorithm {args.algorithm}: {str(e)}")
            continue

    print("\nAll experiments completed. Results saved.")