import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt


def find_seasonal_period(time_series, max_lag=None):
    """
    Estimate the seasonal period of a time series using autocorrelation.

    Parameters:
    time_series (array-like): The input time series data
    max_lag (int): Maximum lag to consider for autocorrelation

    Returns:
    int: Estimated seasonal period
    """
    if max_lag is None:
        max_lag = len(time_series) // 2

    # Calculate autocorrelation
    autocorr = acf(time_series, nlags=max_lag)

    # Find peaks in autocorrelation
    peaks = np.where((autocorr[1:-1] > autocorr[:-2]) &
                     (autocorr[1:-1] > autocorr[2:]))[0] + 1

    # Return the lag with the highest autocorrelation among peaks
    if len(peaks) > 0:
        return peaks[np.argmax(autocorr[peaks])]
    else:
        return 1  # No clear seasonality found


# # Example usage
# np.random.seed(42)
# t = np.arange(365 * 2)  # Two years of daily data
# seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Yearly seasonality
# trend = 0.05 * t
# noise = np.random.normal(0, 1, len(t))
# time_series = seasonal + trend + noise
#
# period = find_seasonal_period(time_series)
# print(f"Estimated seasonal period: {period}")
#
# # Plot the time series and its autocorrelation
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
#
# ax1.plot(t, time_series)
# ax1.set_title("Time Series")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Value")
#
# ax2.plot(acf(time_series, nlags=len(time_series) // 2))
# ax2.set_title("Autocorrelation Function")
# ax2.set_xlabel("Lag")
# ax2.set_ylabel("Autocorrelation")
#
# plt.tight_layout()
# plt.show()