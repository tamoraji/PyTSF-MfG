from modules.config import ALGORITHM_POOL
from statsforecast.models import AutoARIMA
from darts.models import TCNModel
def create_algorithm(algorithm_name: str) -> any:
    if algorithm_name not in ALGORITHM_POOL:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    config = ALGORITHM_POOL[algorithm_name]
    print(config)
    name = config["name"]
    algorithm_class = config['class']
    params = config['params']

    if name == 'AutoARIMA':
        return AutoARIMA(**params)
    elif name == 'TCN':
        return TCNModel(**params)
    else:
        raise ValueError(f"Unknown algorithm class: {algorithm_class}")