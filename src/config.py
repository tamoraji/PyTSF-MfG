ALGORITHM_POOL = {
    'arima': {
        'class': 'statsmodels.tsa.arima.model.ARIMA',
        'params': {'order': (1,1,1)}
    },
    'prophet': {
        'class': 'prophet.Prophet',
        'params': {}
    },
    'exponential_smoothing': {
        'class': 'statsmodels.tsa.holtwinters.ExponentialSmoothing',
        'params': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12}
    }
}

DATASET_POOL = {
    'air_passengers': {
        'file': 'AirPassengers.csv',
        'date_column': 'Date',
        'target_column': 'Passengers',
        'frequency': 'MS'
    }
}

SCENARIOS = [
    {
        'name': 'all_algorithms_comparison',
        'algorithms': ['arima', 'prophet', 'exponential_smoothing'],
        'datasets': ['air_passengers']
    }
]

METRICS = ['mse', 'mae', 'rmse']
OUTPUT_DIR = 'results'