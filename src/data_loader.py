import pandas as pd

def load_dataset_statforecast(dataset_name: str, config: dict[str, any]) -> pd.DataFrame:
    """
    Load a dataset from file and format it for StatsForecast.
    Returns a DataFrame with columns: unique_id, ds, y
    """
    # Load the CSV file
    df = pd.read_csv(f"../data/{config['file']}")
    # Convert date column to datetime
    df[config['date_column']] = pd.to_datetime(df[config['date_column']])
    # Rename columns to match StatsForecast requirements
    df = df.rename(columns={
        config['date_column']: 'ds',
        config['target_column']: 'y'
    })
    # If there's no unique_id column, create one using the dataset name
    if 'unique_id' not in df.columns:
        df['unique_id'] = dataset_name
    # Select only the required columns
    df = df[['unique_id', 'ds', 'y']]
    # Sort by date
    df = df.sort_values('ds')
    # Resample to desired frequency if specified
    if 'frequency' in config:
        df = df.set_index('ds')
        df = df.asfreq(config['frequency'])
        df = df.reset_index()
    print(f"Loaded dataset shape: {df.shape}")
    print(df.head())
    return df
#
# def load_dataset(dataset_name: str, config: dict[str, any]) -> pd.DataFrame:
#     """Load a dataset from file."""
#     df = pd.read_csv(f"../data/{config['file']}")
#     df[config['date_column']] = pd.to_datetime(df[config['date_column']])
#     df.set_index(config['date_column'], inplace=True)
#     df = df.asfreq(config['frequency'])
#     print(df.shape)
#     return df[config['target_column']]

def split_data(data: pd.Series, test_size: float = 0.2) -> tuple:
    """Split the data into training and testing sets."""
    split_point = int(len(data) * (1 - test_size))
    return data[:split_point], data[split_point:]