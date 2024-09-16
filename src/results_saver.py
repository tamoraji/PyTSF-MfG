import json
import os


class ResultsSaver:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, results: Dict[str, Any], scenario: str, algorithm: str, dataset: str):
        filename = f"{scenario}_{algorithm}_{dataset}.json"
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)

    def load_results(self, scenario: str, algorithm: str, dataset: str) -> Dict[str, Any]:
        filename = f"{scenario}_{algorithm}_{dataset}.json"
        with open(os.path.join(self.output_dir, filename), 'r') as f:
            return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
