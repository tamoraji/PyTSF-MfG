import torch
ALGORITHM_POOL = {
    "AutoArima": {
        'name': 'AutoArima',
        'class': 'statsforecast.models.AutoARIMA',
        'params': {},
        'data_format': 'StatsForecast'
    },
    "TCN":{
        'name': 'TCN',
        'class': 'darts.models.TCNModel',
        'params': {
            'input_chunk_length': 50,  # Example value, adjust as needed
            'output_chunk_length': 3,
            'output_chunk_shift': 0,
            'num_layers': 3,
            'num_filters': 64,
            'kernel_size': 3,
            'n_epochs': 20,
            'force_reset': "True",
            'pl_trainer_kwargs': {"accelerator": "cpu"}
        },
        'data_format': 'Darts'
    }

}


DATASET_POOL = {
    'air_passengers': {
        'file': 'AirPassengers.csv',
        'date_column': 'Month',
        'target_column': 'Passengers',
        'frequency': 'MS'
    },
    'ETTh1': {
        'file': 'ETTh1.csv',
        'date_column': 'date',
        'target_column': 'OT',
        'frequency': 'h'
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
OUTPUT_DIR = '../src/results'
