from src.config import DATASET_POOL
import pandas as pd
import os
def load_datasets(data_path):
    datasets = {}
    for name, config in DATASET_POOL.items():
        file_path = os.path.join(data_path, config['file'])
        print(file_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=[config['date_column']])
            df = df.rename(columns={
                config['date_column']: 'ds',
                config['target_column']: 'y'
            })
            # If there's no unique_id column, create one using the dataset name
            if 'unique_id' not in df.columns:
                df['unique_id'] = 1
            # Select only the required columns
            df = df[['unique_id', 'ds', 'y']]
            datasets[name] = df
            print(df.shape)
        else:
            print(f"Warning: File {config['file']} not found for dataset {name}")
    return datasets


import pandas as pd


def blocked_rolling_origin_update(
        data: pd.DataFrame,
        target_column: str,
        lookback_window: int,
        forecast_horizon: int,
        rolling_step: int
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Args:
    data (pd.DataFrame): The time series data.
    target_column (str): Name of the target column.
    lookback_window (int): Number of past time steps to use for prediction.
    forecast_horizon (int): Number of future time steps to predict.
    rolling_step (int): Number of time steps to move forward for each split.

    Returns:
    List of tuples, each containing (X_train, X_test, y_train, y_test)
    """
    results = []
    total_samples = len(data)

    for start in range(0, total_samples - lookback_window - forecast_horizon + 1, rolling_step):
        # Define the indices for this iteration
        train_end = start + lookback_window
        test_end = train_end + forecast_horizon

        if test_end > total_samples:
            break

        # Create train and test sets
        train_data = data.iloc[start:train_end]
        test_data = data.iloc[train_end:test_end]

        # # Prepare X (features) and y (target) for both train and test
        # X_train = train_data.drop(columns=[target_column])
        # y_train = train_data[target_column]
        # X_test = test_data.drop(columns=[target_column])
        # y_test = test_data[target_column]

        results.append((train_data, test_data))

    return results