from data_loader import load_dataset_statforecast, split_data

class DatasetPool:
    def __init__(self, dataset_config: dict[str, dict[str, any]]):
        self.datasets = dataset_config

    def get_dataset(self, dataset_name: str) -> tuple:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in pool")

        data = load_dataset_statforecast(dataset_name, self.datasets[dataset_name])
        return split_data(data)