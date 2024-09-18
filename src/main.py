from config import ALGORITHM, DATASET_POOL, OUTPUT_DIR
from algorithm_factory import create_algorithm
from dataset_pool import DatasetPool
from experiment_runner import ExperimentRunner
from evaluator import Evaluator
from modules.results_saver import ResultsSaver


def main():
    # Initialize pools
    algorithm = create_algorithm(ALGORITHM)
    data_pool = DatasetPool(DATASET_POOL)

    # Initialize experiment runner
    runner = ExperimentRunner(algorithm, data_pool, ALGORITHM)

    # Run experiments
    results = runner.run_experiments()

    # Evaluate results
    evaluator = Evaluator()
    evaluated_results = {
        dataset: evaluator.calculate_metrics(actual, predicted)
        for dataset, (actual, predicted) in results.items()
    }

    # Save results
    saver = ResultsSaver(OUTPUT_DIR)
    for dataset, metrics in evaluated_results.items():
        saver.save_results(metrics, ALGORITHM['name'], dataset)

    print(f"Experiments completed for {ALGORITHM['name']} and results saved.")


if __name__ == "__main__":
    main()