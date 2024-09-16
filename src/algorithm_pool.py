from typing import Any
from algorithm_wrapper import BaseAlgorithmWrapper, ARIMAWrapper, ProphetWrapper, ExponentialSmoothingWrapper

class AlgorithmPool:
    def __init__(self, algorithm_config: dict[str, dict[str, Any]]):
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

    def get_algorithms_subset(self, names: list) -> dict[str, BaseAlgorithmWrapper]:
        return {name: self.get_algorithm(name) for name in names}