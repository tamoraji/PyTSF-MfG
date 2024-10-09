# PyTSF-MfG
A python Time-Series Forecasting (TSF) repository for Manufacturing datasets

Examples to run the algorithms:

python darts_experiments.py --algorithm Block_GRU --horizon 3 --param '{"input_chunk_length": 50, "output_chunk_length": 3}'
python darts_experiments.py --algorithm LSTM --horizon 3 --params '{"input_chunk_length": 50, "training_length": 50}’
python neuralforecast_experiments.py --algorithm Informer --horizon 6


