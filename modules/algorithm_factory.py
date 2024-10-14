import logging
from enum import Enum

from modules.config import ALGORITHM_POOL
from statsforecast.models import AutoARIMA
from darts.models import TCNModel, RNNModel, BlockRNNModel, XGBModel
from neuralforecast.models import TimesNet, Informer, MLP, FEDformer, TimeLLM, NHITS, NBEATS, TiDE
from neuralforecast.losses.pytorch import MSE, MAE
from TSLib.models.SegRNN import Model as SegRNN  # Import the existing SegRNN model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LossFunction(Enum):
    MSE = MSE()
    MAE = MAE()

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

ALGORITHM_CLASSES = {
    'AutoARIMA': AutoARIMA,
    'TCN': TCNModel,
    'Block_GRU': BlockRNNModel,
    'LSTM': RNNModel,
    'XGBoost': XGBModel,
    'TimesNet': TimesNet,
    'Informer': Informer,
    'MLP': MLP,
    'FEDformer': FEDformer,
    'TimeLLM': TimeLLM,
    'NHITS': NHITS,
    'NBEATS': NBEATS,
    'TiDE': TiDE,
    'SegRNN': SegRNN  # Add SegRNN to the ALGORITHM_CLASSES dictionary
}

def create_algorithm(algorithm_name: str, runtime_params: dict[str, any], horizon: int = None) -> any:
    """
    Create and return an instance of the specified forecasting algorithm.

    Args:
        algorithm_name (str): Name of the algorithm to create.
        runtime_params (Dict[str, Any]): Parameters to override default configuration.
        horizon (int, optional): Forecasting horizon. Defaults to None.

    Returns:
        Any: An instance of the specified forecasting algorithm.

    Raises:
        ValueError: If the algorithm name is unknown or if there's an issue with parameters.
    """
    if algorithm_name not in ALGORITHM_POOL:
        logger.error(f"Unknown algorithm: {algorithm_name}")
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    config = ALGORITHM_POOL[algorithm_name]
    logger.debug(f"Base configuration for {algorithm_name}: {config}")

    algorithm_class = ALGORITHM_CLASSES.get(config['name'])
    if not algorithm_class:
        logger.error(f"Unknown algorithm class: {config['class']}")
        raise ValueError(f"Unknown algorithm class: {config['class']}")

    params = {**config['default_params'], **runtime_params}
    print(f'the running params are: {params}')

    if config['name'] == 'SegRNN':
        if horizon is not None:
            if 'h' in params:
                logger.warning(f"Overriding default horizon {params['h']} with provided horizon {horizon}")
            params['pred_len'] = horizon  # Use pred_len for SegRNN
        elif 'pred_len' not in params:
            logger.error(f"Horizon not provided for algorithm {algorithm_name}")
            raise ValueError(f"Horizon not provided for algorithm {algorithm_name}")

    if config['data_format'] == 'NeuralForecast':
        params['h'] = horizon

    if 'loss' in params and isinstance(params['loss'], str):
        try:
            params['loss'] = LossFunction[params['loss'].upper()].value
        except KeyError:
            logger.error(f"Unknown loss function: {params['loss']}")
            raise ValueError(f"Unknown loss function: {params['loss']}")

    logger.debug(f"Final parameters for {algorithm_name}: {params}")

    try:
        if algorithm_name == 'SegRNN':
            # Create a Config object with all parameters
            config_obj = Config(**params)
            # Initialize SegRNN with the Config object
            return algorithm_class(config_obj)
        else:
            return algorithm_class(**params)
    except TypeError as e:
        logger.error(f"Invalid parameters for {algorithm_name}: {str(e)}")
        raise ValueError(f"Invalid parameters for {algorithm_name}: {str(e)}")