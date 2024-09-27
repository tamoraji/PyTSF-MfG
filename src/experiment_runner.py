from dataset_pool import DatasetPool
import pandas as pd
from algorithm_wrapper import BaseAlgorithmWrapper

class ExperimentRunner:
    def __init__(self, algorithm: BaseAlgorithmWrapper, dataset_pool: DatasetPool, algorithm_config: dict[str, any]):
        self.algorithm = algorithm
        self.dataset_pool = dataset_pool
        self.algorithm_config = algorithm_config

    def run_experiments(self):
        results = {}
        for dataset_name in self.dataset_pool.datasets.keys():
            train_data, test_data = self.dataset_pool.get_dataset(dataset_name)
            print(train_data.shape)
            print(test_data.shape)

            # Adjust data format based on algorithm requirements
            if self.algorithm_config['data_format'] == 'StatsForecast':
                self.algorithm.fit(train_data)
                predictions = self.algorithm.predict(len(test_data))
                print(predictions.shape)
                results[dataset_name] = (test_data['y'].values, predictions)
            else:
                print("Algorithm not implemented yet")

        return results