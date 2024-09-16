from data_loader import load_dataset, split_data
from typing import Dict, Any

class DatasetPool:
    def __init__(self, dataset_config: Dict[str, Dict[str, Any]]):
        self.datasets = dataset_config

    def get_dataset(self, dataset_name: str) -> tuple:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in pool")

        data = load_dataset(dataset_name, self.datasets[dataset_name])
        return split_data(data)

    def get_datasets_subset(self, dataset_names: list) -> Dict[str, tuple]:
        return {name: self.get_dataset(name) for name in dataset_names}