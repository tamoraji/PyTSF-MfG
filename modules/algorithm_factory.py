import logging
from enum import Enum

from modules.config import ALGORITHM_POOL
from statsforecast.models import AutoARIMA
from darts.models import TCNModel, RNNModel, BlockRNNModel
from neuralforecast.models import TimesNet, Informer, MLP, FEDformer
from neuralforecast.losses.pytorch import MSE, MAE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LossFunction(Enum):
    MSE = MSE()
    MAE = MAE()

ALGORITHM_CLASSES = {
    'AutoARIMA': AutoARIMA,
    'TCN': TCNModel,
    'Block_GRU': BlockRNNModel,
    'LSTM': RNNModel,
    'TimesNet': TimesNet,
    'Informer': Informer,
    'MLP': MLP,
    'FEDformer': FEDformer
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

    if horizon is not None:
        if 'h' in params:
            logger.warning(f"Overriding default horizon {params['h']} with provided horizon {horizon}")
        params['h'] = horizon
    elif 'h' not in params:
        logger.error(f"Horizon not provided for algorithm {algorithm_name}")
        raise ValueError(f"Horizon not provided for algorithm {algorithm_name}")

    if 'loss' in params and isinstance(params['loss'], str):
        try:
            params['loss'] = LossFunction[params['loss'].upper()].value
        except KeyError:
            logger.error(f"Unknown loss function: {params['loss']}")
            raise ValueError(f"Unknown loss function: {params['loss']}")

    logger.debug(f"Final parameters for {algorithm_name}: {params}")

    try:
        return algorithm_class(**params)
    except TypeError as e:
        logger.error(f"Invalid parameters for {algorithm_name}: {str(e)}")
        raise ValueError(f"Invalid parameters for {algorithm_name}: {str(e)}")