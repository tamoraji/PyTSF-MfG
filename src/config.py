ALGORITHM = {
    'name': 'AutoArima',
    'class': 'statsforecast.models.AutoARIMA',
    'params': {},
    'data_format': 'StatsForecast'
}


DATASET_POOL = {
    'air_passengers': {
        'file': 'AirPassengers.csv',
        'date_column': 'Month',
        'target_column': 'Passengers',
        'frequency': 'MS'
    },
    'stock_prices': {
        'file': 'ABM.csv',
        'date_column': 'Date',
        'target_column': 'Close',
        'frequency': 'D'
    }
}

METRICS = [
    "MSE",
    "RSME",
    "MAE",
    "WAPE",
    "MAPE",
    "SMAPE",
    "RAE",
    "RSE",
    "MASE",
    "R2",
]
OUTPUT_DIR = 'results'
