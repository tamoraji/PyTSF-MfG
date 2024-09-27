import json
import os
import numpy as np

class ResultsSaver:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, results: dict[str, any], algorithm: str, horizon: int, dataset: str):
        filename = f"{algorithm}_{horizon}_{dataset}.json"
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
