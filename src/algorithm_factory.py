from algorithm_wrapper import BaseAlgorithmWrapper, AutoArimaWrapper, ProphetWrapper, ExponentialSmoothingWrapper
def create_algorithm(config: dict[str, any]) -> BaseAlgorithmWrapper:
    if config['name'] == 'AutoArima':
        return AutoArimaWrapper(**config['params'])
    elif config['name'] == 'prophet':
        return ProphetWrapper(**config['params'])
    elif config['name'] == 'exponential_smoothing':
        return ExponentialSmoothingWrapper(**config['params'])
    else:
        raise ValueError(f"Unknown algorithm: {config['name']}")