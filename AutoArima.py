import argparse
import pandas as pd
import numpy as np
from statsforecast.models import AutoARIMA
from sklearn.preprocessing import StandardScaler
from modules.utils import load_datasets_statforecast_uni
from modules.evaluator import Evaluator
from modules.results_saver import ResultsSaver
from modules.performance_utils import measure_time_and_memory
from modules.algorithm_factory import create_algorithm

# Set parameters
# DATA_PATH = '/Users/moji/PyTSF-MfG/data'  # change this in your machine
# OUTPUT_DIR = '/Users/moji/PyTSF-MfG/results'

DATA_PATH = '/home/ma00048/Moji/TSF_data'
OUTPUT_DIR = '/home/ma00048/Moji/TSF_results'

@measure_time_and_memory
def train_model(model, data):
    print(f"Training model on data of shape: {data.shape}")
    print(f"training data type is:{type(data)}")
    if isinstance(data, pd.Series):
        data = data.values
        print(f"new training data type is:{type(data)}")
    return model.fit(data)

@measure_time_and_memory
def test_model(model, horizon):
    print(f"Predicting for horizon: {horizon}")
    return model.predict(h=horizon)

def split_data(data, split_ratio=0.9):
    split_index = int(len(data) * split_ratio)
    train = data[:split_index]
    test = data[split_index:]
    print(f"Data split - Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Train data type is:{type(train)}")
    return train, test

def run_experiment(data, name, horizon, algorithm_name):
    print(f"\nStarting experiment for dataset: {name}")
    print(f"Algorithm: {algorithm_name}, Horizon: {horizon}")
    print(f"Initial data shape: {data.shape}")

    # Split the data
    train, test = split_data(data)

    # # Scale the data
    # scaler = StandardScaler()
    # train_scaled = pd.DataFrame(scaler.fit_transform(train[['y']]), columns=['y'], index=train.index)
    # test_scaled = pd.DataFrame(scaler.transform(test[['y']]), columns=['y'], index=test.index)
    # print(f"Scaled data - Train shape: {train_scaled.shape}, Test shape: {test_scaled.shape}")

    # Create the model using the factory
    model = create_algorithm(algorithm_name)
    print(f"Model created: {type(model).__name__}")

    # Train the model
    _, train_time, train_memory = train_model(model, train['y'])
    print(f"Model trained. Time: {train_time:.2f}s, Memory: {train_memory:.2f}MB")

    # Perform autoregressive prediction
    predictions = []
    history = train['y'].values

    test_time = 0
    test_memory = 0

    print("Starting autoregressive prediction")
    for i in range(0, len(test), horizon):
        n = min(horizon, len(test) - i)
        pred, t_time, t_memory = test_model(model, n)
        print(f"Prediction shape: {pred['mean'].shape}")
        test_time += t_time
        test_memory += t_memory

        # Check if the prediction is a pandas Series or a numpy array
        if isinstance(pred, pd.Series):
            predictions.extend(pred.values)
        elif isinstance(pred, dict) and 'mean' in pred:
            predictions.extend(pred['mean'])
        else:
            predictions.extend(pred)

        print(f"Prediction step {i // horizon + 1}: predicted {len(pred['mean'])} values")
        print(f'new prediction shape: {len(predictions)}')

        # Update history with the true values
        history = np.append(history, test['y'].iloc[i:i+n].values)
        print(f'history shape is:{history.shape}')

        # Refit the model with updated history
        print(f"Refitting model with history of length: {len(history)}")
        model.fit(history)

    print("Autoregressive prediction completed")
    print(f"Total predictions: {len(predictions)}, Expected: {len(test)}")

    # # Inverse transform the predictions
    # predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    # print(f"Inverse transformed predictions shape: {predictions.shape}")

    # Calculate metrics
    actual = test['y'].values
    print(f"Actual values shape: {actual.shape}")
    metrics = Evaluator.calculate_metrics(actual, predictions)

    # Add computational complexity metrics
    metrics['train_time'] = train_time
    metrics['avg_test_time'] = test_time / (len(test) / horizon)
    metrics['train_memory'] = train_memory
    metrics['avg_test_memory'] = test_memory / (len(test) / horizon)
    metrics['total_time'] = train_time + test_time
    metrics['total_memory'] = train_memory + test_memory

    print("Experiment completed")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run time series forecasting experiment with specified algorithm and forecasting horizon')
    parser.add_argument('--algorithm', type=str, default="AutoARIMA", help='Algorithm name (e.g., AutoARIMA, TCN)')
    parser.add_argument('--horizon', type=int, default=3, help='Forecasting horizon')
    args = parser.parse_args()

    print(f"Starting forecasting experiment with {args.algorithm} algorithm and horizon {args.horizon}")

    datasets = load_datasets_statforecast_uni(DATA_PATH)
    print(f"Loaded {len(datasets)} datasets")

    saver = ResultsSaver(OUTPUT_DIR)

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        print("dataset look like:")
        print(data.head())

        metrics = run_experiment(data, name, args.horizon, args.algorithm)

        # Save results
        saver.save_results({f'horizon_{args.horizon}': metrics}, args.algorithm, args.horizon, name)
        print(f"Results saved for {args.algorithm} on {name}")

        # Print summary of results
        print(f"Summary of results for {name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("\nAll experiments completed. Results saved.")
