class ExperimentRunner:
    def __init__(self, scenarios: list, algorithm_pool: AlgorithmPool, dataset_pool: DatasetPool):
        self.scenarios = scenarios
        self.algorithm_pool = algorithm_pool
        self.dataset_pool = dataset_pool

    def run_all_experiments(self):
        results = {}
        for scenario in self.scenarios:
            results[scenario.name] = scenario.run(self.algorithm_pool, self.dataset_pool)
        return results