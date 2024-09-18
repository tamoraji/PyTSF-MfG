from modules.metrics import mse, rmse, mae, wape, mape, smape, rae, rse, mase, r2_score
import numpy as np


class Evaluator:
    @staticmethod
    def calculate_metrics(actual: np.array, predicted: np.array) -> dict[str, float]:
        return {
            "MSE": mse(actual, predicted),
            "RSME": rmse(actual, predicted),
            "MAE": mae(actual, predicted),
            "WAPE": wape(actual, predicted),
            "MAPE": mape(actual, predicted),
            "SMAPE": smape(actual, predicted),
            "RAE": rae(actual, predicted),
            "RSE": rse(actual, predicted),
            "MASE": mase(actual, predicted),
            "R2": r2_score(actual, predicted),
            # 'mse': mean_squared_error(actual, predicted),
            # 'mae': mean_absolute_error(actual, predicted),
            # 'rmse': np.sqrt(mean_squared_error(actual, predicted))
        }

    @staticmethod
    def aggregate_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
        aggregated = {}
        for metric in results[next(iter(results))].keys():
            aggregated[metric] = np.mean([r[metric] for r in results.values()])
        return aggregated