import torch
ALGORITHM_POOL = {
    "AutoARIMA": {
        'name': 'AutoARIMA',
        'class': 'statsforecast.models.AutoARIMA',
        'default_params': {},
        'data_format': 'StatsForecast'
    },
    "TCN": {
        'name': 'TCN',
        'class': 'darts.models.TCNModel',
        'default_params': {
            'num_layers': 1,
            'num_filters': 64,
            "dilation_base": 1,
            'kernel_size': 6,
            'n_epochs': 20,
            'force_reset': "True",
            'pl_trainer_kwargs': {"accelerator": "cpu"},
            # 'pl_trainer_kwargs': {"accelerator": "gpu", "devices": -1, "auto_select_gpus": True} #To use GPU
        },
        'data_format': 'Darts'
    },
    "Block_GRU": {
        'name': 'Block_GRU',
        'class': 'darts.models.BlockRNNModel',
        'default_params': {
            'model': 'GRU',
            'hidden_dim': 32,
            'n_rnn_layers': 2,
            'n_epochs': 20,
            'force_reset': "True",
            'pl_trainer_kwargs': {"accelerator": "cpu"},
            # 'pl_trainer_kwargs': {"accelerator": "gpu", "devices": -1, "auto_select_gpus": True} #To use GPU
        },
        'data_format': 'Darts'
    },
    "LSTM": {
        'name': 'LSTM',
        'class': 'darts.models.RNNModel',
        'default_params': {
            'model': 'LSTM',
            'hidden_dim': 32,
            'n_rnn_layers': 2,
            'n_epochs': 20,
            'force_reset': "True",
            'pl_trainer_kwargs': {"accelerator": "cpu"},
            # 'pl_trainer_kwargs': {"accelerator": "gpu", "devices": -1, "auto_select_gpus": True} #To use GPU
        },
        'data_format': 'Darts'
    },
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
        'target_column': 'Process_temperature',
        'frequency': 'min'
    },
    'Steel_industry_Usage_kWh': {
        'file': 'Steel_industry.csv',
        'date_column': 'ds',
        "date_format": "%y-%m-%d %H:%M",
        'target_column': 'y',
        'frequency': '15min'
    },
    'BrentOilPrices': {
        'file': 'BrentOilPrices.csv',
        'date_column': 'Date',
        'date_format': "%d-%b-%y",
        'target_column': 'Price',
        'frequency': 'D'
    },
    'ECL': {
        'file': 'ECL.csv',
        'date_column': 'date',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'MT_320',
        'frequency': 'h'
    },
    'Monroe Water Treatment Plant': {
        'file': 'MWTP_Elec_Daily.csv',
        'date_column': 'date',
        'date_format': "%m-%d-%y",
        'target_column': 'total_kwh',
        'frequency': 'D'
    },
    'Appliances Energy': {
        'file': 'energydata_complete.csv',
        'date_column': 'date',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Press_mm_hg',
        'frequency': '10min'
    },
    'Seoul Bike Demand': {
        'file': 'SeoulBikeData_processed.csv',
        'date_column': 'Date',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Rented_Bike_Count',
        'frequency': 'h'
    },
    'Gas sensor dynamic gas mixtures': {
        'file': 'gas_sensors_dynamic_mixtures.csv',
        'date_column': 'Date',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'sensor_16',
        'frequency': 's'
    },
    'Gas sensor temperature modulation': {
        'file': 'gas_sensors_temperature.csv',
        'date_column': 'Datetime',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Temperature',
        'frequency': '30s'
    },
    'ISO-NY': {
        'file': 'ISO-NY_Central.csv',
        'date_column': 'Time_Stamp',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Load',
        'frequency': '15min'
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
