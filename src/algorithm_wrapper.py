
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from statsforecast.models import AutoARIMA
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class BaseAlgorithmWrapper(ABC):
    @abstractmethod
    def fit(self, data: pd.Series):
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.array:
        pass

class AutoArimaWrapper(BaseAlgorithmWrapper):
    def __init__(self):
        self.model = None

    def fit(self, data: pd.Series):
        self.model = AutoARIMA().fit(data['y'])
        return self

    def predict(self, steps: int) -> np.array:
        prediction = self.model.predict(steps)
        return prediction['mean']

class ProphetWrapper(BaseAlgorithmWrapper):
    def __init__(self):
        self.model = Prophet()

    def fit(self, data: pd.Series):
        df = pd.DataFrame({'ds': data.index, 'y': data.values})
        self.model.fit(df)
        return self

    def predict(self, steps: int) -> np.array:
        future = self.model.make_future_dataframe(periods=steps, freq='MS')
        forecast = self.model.predict(future)
        return forecast.tail(steps)['yhat'].values

class ExponentialSmoothingWrapper(BaseAlgorithmWrapper):
    def __init__(self, trend='add', seasonal='add', seasonal_periods=12):
        self.params = {'trend': trend, 'seasonal': seasonal, 'seasonal_periods': seasonal_periods}
        self.model = None

    def fit(self, data: pd.Series):
        self.model = ExponentialSmoothing(data, **self.params).fit()
        return self

    def predict(self, steps: int) -> np.array:
        return self.model.forecast(steps)