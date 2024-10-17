from modules.config import DATASET_POOL
import os
import pandas as pd
import numpy as np
from dateutil.parser import parse
import re


def infer_date_format(date_string):
    """Infer the date format from a string."""
    # Check for ISO format (YYYY-MM-DD)
    if re.match(r'\d{4}-\d{2}-\d{2}', date_string):
        return "%Y-%m-%d %H:%M:%S"

    # Check for DD-MM-YYYY format
    elif re.match(r'\d{2}-\d{2}-\d{4}', date_string):
        return "%d-%m-%Y %H:%M"

    # Check for DD-Mon-YY format
    elif re.match(r'\d{2}-[A-Za-z]{3}-\d{2}', date_string):
        return "%d-%b-%y"

    # If no match, return None
    return None


def parse_date_column(df, date_column):
    """Parse the date column using multiple formats."""
    # Try parsing with inferred format
    first_date = df[date_column].dropna().iloc[0]
    inferred_format = infer_date_format(first_date)

    if inferred_format:
        try:
            return pd.to_datetime(df[date_column], format=inferred_format)
        except ValueError:
            pass  # If it fails, we'll try the next method

    # Try parsing with a set of common formats
    common_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",
        "%d-%b-%y",
        "%d-%b-%Y"
    ]

    for fmt in common_formats:
        try:
            return pd.to_datetime(df[date_column], format=fmt)
        except ValueError:
            continue

    # If all else fails, use pandas' flexible parser with a warning
    print(f"Warning: Could not infer consistent date format for column '{date_column}'. "
          "Falling back to flexible parsing, which may be slow and inconsistent.")
    return pd.to_datetime(df[date_column], format='mixed')


def load_datasets_statforecast_uni(data_path):
    datasets = {}
    for name, config in DATASET_POOL.items():
        file_path = os.path.join(data_path, config['file'])
        print(f"\nLoading dataset: {name}")
        print(f"File path: {file_path}")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Date handling
            if config['date_column'] in df.columns:
                df['ds'] = parse_date_column(df, config['date_column'])
                print(f"Date parsing complete for {name}")
            else:
                # If date column doesn't exist, create it
                start_date = pd.to_datetime(config.get('start_date', '1970-01-01'))
                df['ds'] = pd.date_range(start=start_date, periods=len(df), freq=config['frequency'])

            # Set the index (but keep 'ds' as a column for StatForecast compatibility)
            df.set_index('ds', inplace=True)
            df.reset_index(inplace=True)

            # Handle multiple target columns if specified
            target_columns = config.get('target_columns', [config['target_column']])

            for target in target_columns:
                df_target = df.copy()
                df_target = df_target.rename(columns={target: 'y'})

                # If there's no unique_id column, create one using the dataset name and target
                if 'unique_id' not in df_target.columns:
                    df_target['unique_id'] = f"{target}"

                # Select only required columns
                columns_to_keep = ['unique_id', 'ds', 'y'] + config.get('hist_exog_columns', [])

                # Ensure all specified columns exist in the dataframe
                existing_columns = [col for col in columns_to_keep if col in df_target.columns]
                df_target = df_target[existing_columns]

                dataset_name = f"{name}_{target}" if len(target_columns) > 1 else name
                datasets[dataset_name] = df_target

                print(f"Dataset shape for {dataset_name}: {df_target.shape}")
                # print(df_target.info())
                # print(df_target.head())

                # Calculate and print the frequency of the time series
                time_diff = df_target['ds'].diff().mode().iloc[0]
                print(f"Most common time difference: {time_diff}")

        else:
            print(f"Warning: File {config['file']} not found for dataset {name}")
    print(f'the number of datasets: {len(datasets)}')
    return datasets




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