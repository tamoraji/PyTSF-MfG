import torch
ALGORITHM_POOL = {
    "AutoARIMA": {
        'name': 'AutoARIMA',
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
    },
    'ETTm2': {
        'file': 'ETTm2.csv',
        'date_column': 'date',
        'target_column': 'OT',
        'frequency': '15min'
    },
    'ai4i2020': {
        'file': 'ai4i2020.csv',
        'date_column': None,
        'target_column': 'Process temperature [K]',
        'frequency': 'min'
    },
    'Steel_industry_Usage_kWh': {
        'file': 'Steel_industry.csv',
        'date_column': 'Date_Time',
        "date_format": "%d-%m-%Y %H:%M",
        'target_column': 'Usage_kWh',
        'frequency': '15min'
    },
    'BrentOilPrices': {
        'file': 'BrentOilPrices.csv',
        'date_column': 'Date',
        'date_format': "%d-%b-%y",
        'target_column': 'Price',
        'frequency': 'B'
    },
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
