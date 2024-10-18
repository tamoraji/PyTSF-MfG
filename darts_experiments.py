import argparse
import json
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from modules.utils import load_datasets_statforecast_uni
from modules.evaluator import Evaluator
from modules.results_saver import ResultsSaver
from modules.performance_utils import measure_time_and_memory
from modules.algorithm_factory import create_algorithm
from modules.config import DATASET_POOL
from modules.plot_utils import plot_forecast_vs_actual
import os

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
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
def train_model(model, data , past_cov = None):
    print(f"Training model on data of shape: {len(data)}")
    return model.fit(data, past_covariates=past_cov)


@measure_time_and_memory
def test_model(model, n, history, past_cov=None):
    print(f"Predicting for horizon: {n}")
    pred = model.predict(n=n, series=history, past_covariates=past_cov)
    return pred


def split_data(series, split_ratio=0.8):
    train, test = series.split_before(split_ratio)
    print(f"Data split - Train shape: {len(train)}, Test shape: {len(test)}")
    print(f"train timestamps: {train.time_index}, Test timestamps: {test.time_index}")
    return train, test


def run_experiment(data, name, horizon, algorithm_name, algorithm_params, mode):
    print(f"\nStarting experiment for dataset: {name}")
    print(f"Algorithm: {algorithm_name}, Horizon: {horizon}")
    print(f"Initial data shape: {len(data)}")

    # Get dataset-specific configuration
    dataset_config = DATASET_POOL.get(name, {})
    frequency = dataset_config.get('frequency', 'h')
    hist_exog_list = dataset_config.get('hist_exog_list', []) if mode == 'multivariate' else None

    try:
        if mode == 'univariate':
            series = TimeSeries.from_dataframe(data, 'ds', 'y', freq=frequency)
        else:
            series = TimeSeries.from_dataframe(data, 'ds', 'y', freq=frequency)
            past_cov = TimeSeries.from_dataframe(data, 'ds', hist_exog_list, freq=frequency)
        series = series.astype(np.float32)
        past_cov = past_cov.astype(np.float32)
    except Exception as e:
        print(f"Error creating TimeSeries: {str(e)}")
        print(f"Dataset head:\n{data.head()}")
        raise

    # Split the data
    train, test = split_data(series)
    past_cov_train, past_cov_test = split_data(past_cov)

    # Scale the data
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    print(f"Scaled data - Train length: {len(train_scaled)}, Test length: {len(test_scaled)}")

    # Create the model using the factory
    model = create_algorithm(algorithm_name, algorithm_params, mode)
    print(f"Model created: {type(model).__name__}")

    # Train the model
    # past_cov = train[hist_exog_list] if hist_exog_list else None
    logger.info(f'past cov time is: {past_cov_train.time_index}')
    _, train_time, train_memory = train_model(model, train_scaled, past_cov_train)
    print(f"Model trained. Time: {train_time:.2f}s, Memory: {train_memory:.2f}MB")

    # Perform autoregressive prediction
    predictions = []
    history = train_scaled
    past_cov_history = past_cov_train

    test_time = 0
    test_memory = 0

    print("Starting autoregressive prediction")
    prediction_length = len(test_scaled)
    for i in range(0, prediction_length, horizon):
        n = min(horizon, prediction_length - i)

        print(f"Predicting for timestamps: {test_scaled.time_index[i:i+n]}")
        print(f"Past cov timestamps: {past_cov_history.time_index}")
        print(f"step: {i+1}")


        pred, t_time, t_memory = test_model(model, n, history, past_cov_history)
        test_time += t_time
        test_memory += t_memory

        # Debug print: Show the timestamps of the predictions
        print(f"Prediction timestamps: {pred.time_index}")
        print(f"Prediction values: {pred.values()}")

        predictions.append(pred)

        # Update history with the true values
        history = history.append(test_scaled[i:i + n])
        past_cov_history = past_cov_history.append(past_cov_test[i:i + n])


        print(f"Prediction step {i // horizon + 1}: predicted {len(pred)} values")

    print("Autoregressive prediction completed")

    # Combine predictions
    combined_predictions = predictions[0]
    for pred in predictions[1:]:
        combined_predictions = combined_predictions.concatenate(pred)

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(combined_predictions)
    print(f"Inverse transformed predictions shape: {len(predictions)}")

    # Calculate metrics
    actual = test['y'].pd_dataframe()
    print(f"Actual: {actual}")
    forecast = predictions.pd_dataframe()
    print(f"Forecast: {forecast}")
    metrics = Evaluator.calculate_metrics(actual.values, forecast.values)

    # Add computational complexity metrics
    metrics['train_time'] = train_time
    metrics['avg_test_time'] = test_time / (len(test) / horizon)
    metrics['train_memory'] = train_memory
    metrics['avg_test_memory'] = test_memory / (len(test) / horizon)
    metrics['total_time'] = train_time + test_time
    metrics['total_memory'] = train_memory + test_memory

    plot_forecast_vs_actual(actual, forecast, name, algorithm_name, horizon, PLOT_DIR)

    print("Experiment completed")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run time series forecasting experiment with specified algorithm and forecasting horizon')
    parser.add_argument('--algorithm', type=str, default="TCN", help='Algorithm name (e.g., TCN)')
    parser.add_argument('--horizon', type=int, default=12, help='Forecasting horizon')
    parser.add_argument('--params', type=str, default='{}', help='JSON string of algorithm parameters')
    parser.add_argument('--datasets', nargs='*', help='List of specific datasets to process. If not provided, all datasets will be processed.')
    parser.add_argument('--mode', type=str, default='univariate', choices=['univariate', 'multivariate'], help='Forecasting mode')
    args = parser.parse_args()

    # Parse the JSON string of parameters
    try:
        algorithm_params = json.loads(args.params)
    except json.JSONDecodeError:
        print("Error: Invalid JSON string for parameters")
        exit(1)

    print(f"Starting forecasting experiment with algorithm: {args.algorithm} and horizon {args.horizon}")
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
        print("Dataset looks like:")
        print(data.head())

        try:
            metrics = run_experiment(data, name, args.horizon, args.algorithm, algorithm_params, mode=args.mode)
            # Save results
            saver.save_results({f'horizon_{args.horizon}': metrics}, args.algorithm, args.horizon, name, args.mode)
            print(f"Results saved for {args.algorithm} on {name}")

            # Print summary of results
            print(f"Summary of results for {name} with {args.algorithm} in {args.mode} mode:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        except Exception as e:
            print(f"Error processing dataset {name} with algorithm {args.algorithm}: {str(e)}")
            continue

    print("\nAll experiments completed. Results saved.")