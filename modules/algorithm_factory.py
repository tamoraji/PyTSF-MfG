from modules.config import ALGORITHM_POOL
from statsforecast.models import AutoARIMA
from darts.models import TCNModel


def create_algorithm(algorithm_name: str, runtime_params: dict) -> any:
    if algorithm_name not in ALGORITHM_POOL:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    config = ALGORITHM_POOL[algorithm_name]
    print(f"Base configuration for {algorithm_name}:", config)

    name = config["name"]
    algorithm_class = config['class']
    default_params = config['default_params']

    # Merge default params with runtime params
    params = default_params.copy()
    params.update(runtime_params)

    print(f"Final parameters for {algorithm_name}:", params)

    if name == 'AutoARIMA':
        return AutoARIMA(**params)
    elif name == 'TCN':
        return TCNModel(**params)
    else:
        raise ValueError(f"Unknown algorithm class: {algorithm_class}")