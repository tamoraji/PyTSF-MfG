from neuralforecast.losses.pytorch import MSE

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
            'num_layers': 2,
            'num_filters': 64,
            "dilation_base": 2,
            'kernel_size': 6,
            'n_epochs': 100,
            'force_reset': "True",
            # 'pl_trainer_kwargs': {"accelerator": "cpu"},
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
            'n_epochs': 100,
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
            'n_epochs': 100,
            'force_reset': "True",
            'training_length': 1000, #TODO: take this outside to the arguments
            'pl_trainer_kwargs': {"accelerator": "cpu"},
            # 'pl_trainer_kwargs': {"accelerator": "gpu", "devices": -1, "auto_select_gpus": True} #To use GPU
        },
        'data_format': 'Darts'
    },
    "XGBoost": {
        'name': 'XGBoost',
        'class': 'darts.models.XGBModel',
        'default_params': {
            'multi_models': False,
        },
        'data_format': 'Darts'
    },
    "TimesNet": {
        'name': 'TimesNet',
        'class': 'neuralforecast.models.TimesNet',
        'default_params': {
            'input_size': 50,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    "Informer": {
        'name': 'Informer',
        'class': 'neuralforecast.models.Informer',
        'default_params': {
            'input_size': 50,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    "MLP": {
        'name': 'MLP',
        'class': 'neuralforecast.models.MLP',
        'default_params': {
            'input_size': 50,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    "FEDformer": {
        'name': 'FEDformer',
        'class': 'neuralforecast.models.FEDformer',
        'default_params': {
            'input_size': 1000,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    "TimeLLM": {
        'name': 'TimeLLM',
        'class': 'neuralforecast.models.TimeLLM',
        'default_params': {
            'input_size': 50,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    "NBEATS": {
        'name': 'NBEATS',
        'class': 'neuralforecast.models.NBEATS',
        'default_params': {
            'input_size': 50,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    "NHITS": {
        'name': 'NHITS',
        'class': 'neuralforecast.models.NHITS',
        'default_params': {
            'input_size': 1000,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    "TiDE": {
        'name': 'TiDE',
        'class': 'neuralforecast.models.TiDE',
        'default_params': {
            'input_size': 1000,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    "BiTCN": {
        'name': 'BiTCN',
        'class': 'neuralforecast.models.BiTCN',
        'default_params': {
            'input_size': 1000,
            'loss': MSE(),
            'max_steps': 100,
            'batch_size': 32,
            'scaler_type': 'minmax',
        },
        'data_format': 'NeuralForecast'
    },
    'SegRNN': {
        'name': 'SegRNN',
        'default_params': {
            'seq_len': 100,
            'enc_in': 1,
            'd_model': 512,
            'dropout': 0.1,
            'seg_len': 5,
            'task_name': 'long_term_forecast'
        },
        'data_format': 'TSLib',
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
        'frequency': 'h',
        'hist_exog_list': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    },
    'ETTm2': {
        'file': 'ETTm2.csv',
        'date_column': 'date',
        'target_column': 'OT',
        'frequency': '15min',
        'hist_exog_list': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    },
    'ai4i2020': {
        'file': 'ai4i2020.csv',
        'date_column': None,
        'target_column': 'Process_temperature',
        'frequency': 'min',
        'hist_exog_list': ['Air temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
    },
    'Steel_industry_Usage_kWh': {
        'file': 'Steel_industry.csv',
        'date_column': 'Date_Time',
        "date_format": "%y-%m-%d %H:%M%S",
        'target_column': 'Usage_kWh',
        'hist_exog_list': ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor'],
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
        'hist_exog_list': ['MT_313', 'MT_314', 'MT_315', 'MT_316', 'MT_317', 'MT_318', 'MT_319'],
        'frequency': 'h'
    },
    'Monroe Water Treatment Plant': {
        'file': 'MWTP_Elec_Daily.csv',
        'date_column': 'date',
        'date_format': "%m-%d-%y",
        'target_column': 'total_kwh',
        'hist_exog_list': ['kwh1', 'kw1', 'billed_kwh', 'mg_finish'],
        'frequency': 'D'
    },
    'Appliances Energy': {
        'file': 'energydata_complete.csv',
        'date_column': 'date',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Press_mm_hg',
        'hist_exog_columns': ["T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint"],
        'frequency': '10min'
    },
    'Seoul Bike Demand': {
        'file': 'SeoulBikeData_processed.csv',
        'date_column': 'Date',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Rented_Bike_Count',
        'hist_exog_list': ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)'],
        'frequency': 'h'
    },
    'Gas sensor dynamic gas mixtures': {
        'file': 'gas_sensors_dynamic_mixtures.csv',
        'date_column': 'Date',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'sensor_16',
        'hist_exog_list': ['sensor 10', 'sensor 11', 'sensor 12', 'sensor 13', 'sensor 14', 'sensor 15'],
        'frequency': 's'
    },
    'Gas sensor temperature modulation': {
        'file': 'gas_sensors_temperature.csv',
        'date_column': 'Datetime',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Temperature',
        'hist_exog_list': ['Humidity (%r.h.)', 'Flow rate (mL/min)', 'Heater voltage (V)', 'R1 (MOhm)'],
        'frequency': '30s'
    },
    'ISO-NY': {
        'file': 'ISO-NY_Central.csv',
        'date_column': 'Time_Stamp',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Load',
        'frequency': '15min'
    },
    'Electricity': {
        'file': 'electricity_data.csv',
        'date_column': 'Timestamp',
        'date_format': "%Y-%m-%d %H:%M:%S",
        'target_column': 'Global_active_power',
        'hist_exog_list': ['Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
        'frequency': 'h'
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
