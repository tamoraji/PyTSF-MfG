import numpy as np

EPSILON = 1e-10

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted + EPSILON


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) / (
            _error(actual[seasonality:], _naive_forecasting(actual, seasonality))
            + EPSILON
        )

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


def wape(actual: np.ndarray, predicted: np.ndarray):
    """ Weighted Absolute Percentage Error """
    return mae(actual, predicted) / (np.mean(actual) + EPSILON)


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error

    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return 100 * np.mean(
        2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
    )


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error

    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / (
        mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    )


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Error """
    __mae = mae(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1)
    )


def std_ape(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_percentage_error(actual, predicted) - __mape))
        / (len(actual) - 1)
    )


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Error """
    # return np.mean( np.abs(_error(actual, predicted)) / (actual + EPSILON))
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (
        np.sum(np.abs(actual - np.mean(actual))) + EPSILON
    )


def rse(actual: np.ndarray, predicted: np.ndarray):
    """
    Calculate the Root Relative Squared Error (RSE).

    RSE = sqrt(∑(y_true - y_pred)^2 / (∑(y_true - mean(y_true))^2 + EPSILON))

    :param actual: Actual values (numpy array or list)
    :param predicted: Predicted values (numpy array or list)
    :return: Root Relative Squared Error (float)
    """
    numerator = np.sum(np.square(actual - predicted))
    denominator = np.sum(np.square(actual - np.mean(actual))) + EPSILON
    return np.sqrt(numerator / denominator)


def r2_score(actual: np.ndarray, predicted: np.ndarray):
    """
    Calculate the R² score (coefficient of determination).

    :param actual: Actual values (numpy array or list)
    :param predicted: Predicted values (numpy array or list)
    :return: R² score (float)
    """
    ss_res = np.sum(np.square(actual - predicted))  # Sum of squared residuals
    ss_tot = np.sum(np.square(actual - np.mean(actual)))  # Total sum of squares
    return 1 - (ss_res / (ss_tot + EPSILON))  # Adding EPSILON for stability


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))