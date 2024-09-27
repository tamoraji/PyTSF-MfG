from algorithm_wrapper import BaseAlgorithmWrapper, ARIMAWrapper, ProphetWrapper, ExponentialSmoothingWrapper

class AlgorithmPool:
    def __init__(self, algorithm_config: dict[str, dict[str, any]]):
        self.algorithms = {
            'arima': ARIMAWrapper,
            'prophet': ProphetWrapper,
            'exponential_smoothing': ExponentialSmoothingWrapper
        }
        self.config = algorithm_config

    def get_algorithm(self, name: str) -> BaseAlgorithmWrapper:
        if name not in self.config:
            raise ValueError(f"Algorithm {name} not found in pool")

        algo_class = self.algorithms[name]
        return algo_class(**self.config[name]['params'])
