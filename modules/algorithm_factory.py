import logging
from enum import Enum

from modules.config import ALGORITHM_POOL
from statsforecast.models import AutoARIMA
from darts.models import TCNModel, RNNModel, BlockRNNModel, XGBModel
from darts.models.forecasting.dlinear import DLinearModel
from neuralforecast.models import TimesNet, Informer, MLP, FEDformer, TimeLLM, NHITS, NBEATS, TiDE, BiTCN, PatchTST, TSMixerx, iTransformer, TFT
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
    'DLinear': DLinearModel,
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
    'BiTCN': BiTCN,
    'PatchTST': PatchTST,
    'TSMixerx': TSMixerx,
    'iTransformer': iTransformer,
    'TFT': TFT,
    'SegRNN': SegRNN  # Add SegRNN to the ALGORITHM_CLASSES dictionary
}

def create_algorithm(algorithm_name: str, runtime_params: dict[str, any], mode, horizon: int = None, hist_exog_list: list = None) -> any:
    """
    Create and return an instance of the specified forecasting algorithm.

    Args:
        algorithm_name (str): Name of the algorithm to create.
        runtime_params (Dict[str, Any]): Parameters to override default configuration.
        horizon (int, optional): Forecasting horizon. Defaults to None.
        mode (str, optional): 'univariate' or 'multivariate'. Defaults to 'univariate'.
        hist_exog_columns (list, optional): List of historic exogenous columns. Required for multivariate mode.

    Returns:
        Any: An instance of the specified forecasting algorithm.

    Raises:
        ValueError: If the algorithm name is unknown or if there's an issue with parameters.
    """
    if algorithm_name not in ALGORITHM_POOL:
        logger.error(f"Unknown algorithm: {algorithm_name}")
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    config = ALGORITHM_POOL[algorithm_name]
    logger.info(f"Base configuration for {algorithm_name}: {config}")

    algorithm_class = ALGORITHM_CLASSES.get(config['name'])
    if not algorithm_class:
        logger.error(f"Unknown algorithm class: {config['class']}")
        raise ValueError(f"Unknown algorithm class: {config['class']}")

    params = {**config['default_params'], **runtime_params}
    logger.info(f'the running params are: {params}')

    if config['data_format'] == 'NeuralForecast':
        params['h'] = horizon

    if config['data_format'] == 'NeuralForecast':
        if mode == 'multivariate':
            if not hist_exog_list:
                logger.error("Historic exogenous columns must be provided for multivariate mode")
                raise ValueError("Historic exogenous columns must be provided for multivariate mode")
            else:
                params['hist_exog_list'] = hist_exog_list
        # Add similar conditions for other data formats if needed

    logger.info(f'the updated params are: {params}')

    if 'loss' in params and isinstance(params['loss'], str):
        try:
            params['loss'] = LossFunction[params['loss'].upper()].value
        except KeyError:
            logger.error(f"Unknown loss function: {params['loss']}")
            raise ValueError(f"Unknown loss function: {params['loss']}")

    logger.info(f"Final parameters for {algorithm_name}: {params}")

    try:
        model = algorithm_class(**params)
        logger.info(f"printing algorithm params {algorithm_class(**params)} for {algorithm_name} algorithm")
        return model
    except TypeError as e:
        logger.error(f"Invalid parameters for {algorithm_name}: {str(e)}")
        raise ValueError(f"Invalid parameters for {algorithm_name}: {str(e)}")