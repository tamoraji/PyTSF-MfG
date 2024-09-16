from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


class Evaluator:
    @staticmethod
    def calculate_metrics(actual: np.array, predicted: np.array) -> dict[str, float]:
        return {
            'mse': mean_squared_error(actual, predicted),
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted))
        }

    @staticmethod
    def aggregate_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
        aggregated = {}
        for metric in results[next(iter(results))].keys():
            aggregated[metric] = np.mean([r[metric] for r in results.values()])
        return aggregated