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
from modules.plot_utils import plot_forecast_vs_actual
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set parameters
DATA_PATH = '/Users/moji/PyTSF-MfG/data'  # change this in your machine
OUTPUT_DIR = '/Users/moji/PyTSF-MfG/results'
PLOT_DIR = '/Users/moji/PyTSF-MfG/plots'  # Directory for plots

# DATA_PATH = '/home/ma00048/Moji/TSF_data'  # change this in your machine
# OUTPUT_DIR = '/home/ma00048/Moji/TSF_results'
# PLOT_DIR = '/home/ma00048/Moji/plots'  # Directory for plots

# Create PLOT_DIR if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

@measure_time_and_memory
def train_model(model, data):
    logger.info(f"Training model on data of shape: {data.shape}")
    return model.fit(data)


@measure_time_and_memory
def test_model(model, for_test_data):
    return model.predict(for_test_data)


def run_experiment(data, name, horizon, algorithm_name, algorithm_params, mode):
    logger.info(f"\nStarting experiment for dataset: {name}")
    logger.info(f"Algorithm: {algorithm_name}, Horizon: {horizon}")
    logger.info(f"Initial data shape: {data.shape}")

    # Get dataset-specific configuration
    dataset_config = DATASET_POOL.get(name, {})
    frequency = dataset_config.get('frequency', 'h')
    hist_exog_list = dataset_config.get('hist_exog_list', []) if mode == 'multivariate' else None
    logger.info(f"History exogenous columns: {hist_exog_list}")

    # Split the data
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    train = data[:split_index]
    test = data[split_index:]

    # Ensure test set length is a multiple of horizon
    test_length = (len(test) // horizon) * horizon
    test = test[:test_length]
    print(test.head())

    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Create the model
    model = create_algorithm(algorithm_name, algorithm_params, mode, horizon=horizon, hist_exog_list=hist_exog_list)
    logger.info(f"Model created: {type(model).__name__}")
    # If the above doesn't work, try:
    logger.info(f"Model parameters: {model.__dict__}")
    # Wrap the model in NeuralForecast
    nf = NeuralForecast(models=[model], freq=frequency)

    # Initialize training metrics
    train_time = 0
    train_memory = 0

    # Initial training
    _, initial_train_time, initial_train_memory = train_model(nf, train)
    train_time += initial_train_time
    train_memory += initial_train_memory
    logger.info(f"Model initially trained. Time: {initial_train_time:.2f}s, Memory: {initial_train_memory:.2f}MB")

    # Perform autoregressive prediction
    predictions = []
    history = train.copy()
    logger.info(f"Train shape: {history.shape}")
    test_time = 0
    test_memory = 0

    logger.info("Starting autoregressive prediction")
    for i in range(0, len(test), horizon):

        # Make predictions
        test_chunk = test.iloc[i:i + horizon]
        logger.info(f"Test chunk shape: {test_chunk.shape}")
        logger.info(f"Predicting for dates: {test_chunk['ds'].iloc[0]} to {test_chunk['ds'].iloc[-1]}")

        logger.info(f"Predicting for horizon: {horizon}")
        forecasts, t_time, t_memory = test_model(nf, history)
        logger.info(f"Forecasts: {forecasts.shape}")
        test_time += t_time
        test_memory += t_memory

        predictions.append(forecasts)
        logger.info(f"Forecasts:\n{forecasts}")

        # Update history
        history = pd.concat([history, test_chunk])
        logger.info(f"history shape:\n{history.shape}")

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
    metrics['avg_test_time'] = test_time / (len(test) // horizon)
    metrics['train_memory'] = train_memory
    metrics['avg_test_memory'] = test_memory / (len(test) // horizon)
    metrics['total_time'] = train_time + test_time
    metrics['total_memory'] = train_memory + test_memory

    # Generate plots
    plot_forecast_vs_actual(actual, forecast, name, algorithm_name, horizon, PLOT_DIR)

    logger.info("Experiment completed")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run time series forecasting experiment with specified algorithm and forecasting horizon')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name (e.g., TimesNet)')
    parser.add_argument('--horizon', type=int, default=12, help='Forecasting horizon')
    parser.add_argument('--params', type=str, default='{}', help='JSON string of algorithm parameters')
    parser.add_argument('--datasets', nargs='*', help='List of specific datasets to process. If not provided, all datasets will be processed.')
    parser.add_argument('--mode', type=str, default='univariate', choices=['univariate', 'multivariate'], help='Forecasting mode')
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
            metrics = run_experiment(data, name, args.horizon, args.algorithm, algorithm_params, mode=args.mode)
            # Save results
            saver.save_results({f'horizon_{args.horizon}': metrics}, args.algorithm, args.horizon, name, args.mode)
            print(f"Results saved for {args.algorithm} on {name} in {args.mode} mode")

            # Print summary of results
            print(f"Summary of results for {name} in {args.mode} mode:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        except Exception as e:
            print(f"Error processing dataset {name} with algorithm {args.algorithm}: {str(e)}")
            continue

    print("\nAll experiments completed. Results saved.")