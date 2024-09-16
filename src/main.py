from config import ALGORITHM_POOL, DATASET_POOL, SCENARIOS, OUTPUT_DIR
from algorithm_pool import AlgorithmPool
from dataset_pool import DatasetPool
from scenarios import SimpleScenario
from experiment_runner import ExperimentRunner
from evaluator import Evaluator
from results_saver import ResultsSaver


def main():
    # Initialize pools
    algo_pool = AlgorithmPool(ALGORITHM_POOL)
    data_pool = DatasetPool(DATASET_POOL)

    # Create scenarios
    scenarios = [SimpleScenario(**scenario) for scenario in SCENARIOS]

    # Initialize experiment runner
    runner = ExperimentRunner(scenarios, algo_pool, data_pool)

    # Run experiments
    results = runner.run_all_experiments()

    # Evaluate results
    evaluator = Evaluator()
    evaluated_results = {
        scenario_name: {
            (algo, dataset): evaluator.calculate_metrics(actual, predicted)
            for (algo, dataset), (actual, predicted) in scenario_results.items()
        }
        for scenario_name, scenario_results in results.items()
    }

    # Save results
    saver = ResultsSaver(OUTPUT_DIR)
    for scenario_name, scenario_results in evaluated_results.items():
        for (algo, dataset), metrics in scenario_results.items():
            saver.save_results(metrics, scenario_name, algo, dataset)

    print("Experiments completed and results saved.")


if __name__ == "__main__":
    main()