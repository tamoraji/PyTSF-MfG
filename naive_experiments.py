import argparse
import json
import pandas as pd
import numpy as np
from modules.utils import load_datasets_statforecast_uni
from modules.evaluator import Evaluator
from modules.results_saver import ResultsSaver
from modules.performance_utils import measure_time_and_memory
from modules.config import DATASET_POOL
from modules.plot_utils import plot_forecast_vs_actual, plot_forecast_vs_actual_with_history

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
def naive_forecast(history, horizon, name, frequency):

    last_value = history['y'].iloc[-1]

    try:
        last_date = history['ds'].iloc[-1]
        forecast = pd.DataFrame({
            'ds': pd.date_range(start=last_date, periods=horizon + 1, freq=frequency)[1:],
            'naive_forecast': [last_value] * horizon
        })
        logger.info(f"Forecast date range: {forecast['ds'].min()} to {forecast['ds'].max()}")
        logger.info(f"Forecast frequency: {frequency}")
        logger.info(f"Number of forecast points: {len(forecast)}")

        if len(forecast) != horizon:
            logger.warning(f"Number of forecast points ({len(forecast)}) does not match horizon ({horizon})")


    except ValueError as e:
        logger.error(f"Error creating forecast for {name}: {str(e)}")
        logger.error(f"Last historical date: {last_date}")
        raise

    return forecast

def run_experiment(data, name, horizon):
    logger.info(f"\nStarting experiment for dataset: {name}")
    logger.info(f"Algorithm: Naive Forecast, Horizon: {horizon}")
    logger.info(f"Initial data shape: {data.shape}")

    dataset_config = DATASET_POOL.get(name, {})
    logger.info(f"dataset config is: {dataset_config}")
    frequency = dataset_config.get('frequency', 'h')
    logger.info(f"The inferred freq is: {frequency}")


    # Split the data
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    train = data[:split_index]
    test = data[split_index:]

    # Ensure test set length is a multiple of horizon
    test_length = (len(test) // horizon) * horizon
    test = test[:test_length]

    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Perform autoregressive prediction
    predictions = []
    history = train.copy()
    test_time = 0
    test_memory = 0

    logger.info("Starting autoregressive prediction")
    for i in range(0, len(test), horizon):
        # Make predictions
        test_chunk = test.iloc[i:i + horizon]
        logger.info(f"Predicting for dates: {test_chunk['ds'].iloc[0]} to {test_chunk['ds'].iloc[-1]}")

        forecasts, t_time, t_memory = naive_forecast(history, horizon, name, frequency)
        test_time += t_time
        test_memory += t_memory

        predictions.append(forecasts)

        # Update history
        history = pd.concat([history, test_chunk])

        logger.info(f"Prediction step {i // horizon + 1}: predicted {len(forecasts)} values")

    logger.info("Autoregressive prediction completed")

    # Combine predictions
    combined_predictions = pd.concat(predictions)

    # Calculate metrics
    actual = test['y'].values
    forecast = combined_predictions['naive_forecast'].values
    metrics = Evaluator.calculate_metrics(actual, forecast)

    # Add computational complexity metrics
    metrics['avg_test_time'] = test_time / (len(test) // horizon)
    metrics['avg_test_memory'] = test_memory / (len(test) // horizon)
    metrics['total_time'] = test_time
    metrics['total_memory'] = test_memory

    # Generate plots
    plot_forecast_vs_actual(actual, forecast, name, 'NaiveForecast', horizon, PLOT_DIR)
    # plot_forecast_vs_actual_with_history(train['y'], actual, forecast, name, 'NaiveForecast', horizon, PLOT_DIR)

    logger.info("Experiment completed")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run naive forecasting experiment with specified forecasting horizon')
    parser.add_argument('--horizon', type=int, default=12, help='Forecasting horizon')
    parser.add_argument('--datasets', nargs='*', help='List of specific datasets to process. If not provided, all datasets will be processed.')
    args = parser.parse_args()

    print(f"Starting naive forecasting experiment with horizon {args.horizon}")
    print(f"Plots will be saved in: {PLOT_DIR}")

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
            metrics = run_experiment(data, name, args.horizon)
            # Save results
            saver.save_results({f'horizon_{args.horizon}': metrics}, 'NaiveForecast', args.horizon, name)
            print(f"Results saved for NaiveForecast on {name}")

            # Print summary of results
            print(f"Summary of results for {name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        except Exception as e:
            print(f"Error processing dataset {name} with NaiveForecast: {str(e)}")
            continue

    print("\nAll experiments completed. Results saved.")
    print(f"Plots are saved in: {PLOT_DIR}")