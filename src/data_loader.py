import pandas as pd
from typing import Dict, Any

def load_dataset(dataset_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Load a dataset from file."""
    df = pd.read_csv(f"data/{config['file']}")
    df[config['date_column']] = pd.to_datetime(df[config['date_column']])
    df.set_index(config['date_column'], inplace=True)
    df = df.asfreq(config['frequency'])
    return df[config['target_column']]

def split_data(data: pd.Series, test_size: float = 0.2) -> tuple:
    """Split the data into training and testing sets."""
    split_point = int(len(data) * (1 - test_size))
    return data[:split_point], data[split_point:]