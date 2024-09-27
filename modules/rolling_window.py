import pandas as pd
import numpy as np

def create_rolling_window_dataset_with_time(data, window_size, horizon, step=1, target_column=0):
    """
    Create a rolling window dataset for time series forecasting.

    Parameters:
    -----------
    data (pd.DataFrame): The input data as a pandas DataFrame. Must have a DateTimeIndex.
    window_size (int): The size of the rolling window (number of past observations to use).
    horizon (int): The number of future observations to forecast.
    step (int, optional): The step size between consecutive windows. Defaults to 1.
    target_column (int, optional): The index of the target column in the DataFrame. Defaults to 5.

    Returns:
    --------
    tuple: A tuple containing:
        - X_stride (np.ndarray): The input features for each window with shape (n_windows, window_size, n_features).
        - y_stride (np.ndarray): The target values for each window with shape (n_windows, horizon).
        - time_stride (list): A list of DateTimeIndex objects corresponding to the time indices for each window.

    Raises:
    -------
    ValueError: If the input data is not a pandas DataFrame.
    ValueError: If the DataFrame index is not a DateTimeIndex.
    ValueError: If there are not enough data points for the given window size and horizon.

    Example:
    ---------
    >>> import pandas as pd
    >>> import numpy as np
    >>> date_rng = pd.date_range(start='2023-01-01', periods=200, freq='D')
    >>> df = pd.DataFrame(np.random.randn(200, 6), index=date_rng, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6'])
    >>> window_size = 10
    >>> horizon = 5
    >>> step = 1
    >>> target_column = 5
    >>> X_stride, y_stride, time_stride = create_rolling_window_dataset_with_time(df, window_size, horizon, step, target_column)
    >>> print(X_stride.shape)  # Output: (186, 10, 6)
    >>> print(y_stride.shape)  # Output: (186, 5)
    >>> print(len(time_stride))  # Output: 186
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DateTimeIndex")
    
    n_samples, n_features = data.shape

    if n_samples < window_size + horizon:
        raise ValueError("Not enough data points for the given window size and horizon.")

    n_windows = (n_samples - window_size - horizon) // step + 1

    # Convert the DataFrame to a NumPy array
    data_values = data.values

    X_stride = np.lib.stride_tricks.as_strided(
        data_values,
        shape=(n_windows, window_size, n_features),
        strides=(data_values.strides[0] * step, data_values.strides[0], data_values.strides[1])
    )

    y_stride = np.lib.stride_tricks.as_strided(
        data_values[:, target_column],
        shape=(n_windows, horizon),
        strides=(data_values.strides[0] * step, data_values.strides[0])
    )

    time_stride = [data.index[i: i + window_size] for i in range(0, n_samples - window_size - horizon + 1, step)]

    return X_stride, y_stride, time_stride