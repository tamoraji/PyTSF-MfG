from abc import ABC, abstractmethod
from algorithm_pool import AlgorithmPool
from dataset_pool import DatasetPool


class BaseScenario(ABC):
    def __init__(self, name: str, algorithms: list, datasets: list):
        self.name = name
        self.algorithms = algorithms
        self.datasets = datasets

    @abstractmethod
    def run(self, algorithm_pool: AlgorithmPool, dataset_pool: DatasetPool):
        pass


class SimpleScenario(BaseScenario):
    def run(self, algorithm_pool: AlgorithmPool, dataset_pool: DatasetPool):
        results = {}
        for dataset_name in self.datasets:
            train_data, test_data = dataset_pool.get_dataset(dataset_name)
            for algo_name in self.algorithms:
                algorithm = algorithm_pool.get_algorithm(algo_name)
                algorithm.fit(train_data)
                predictions = algorithm.predict(len(test_data))
                results[(algo_name, dataset_name)] = (test_data.values, predictions)
        return results