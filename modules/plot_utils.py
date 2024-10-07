# plot_utils.py

import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_forecast_vs_actual(actual, forecast, dataset_name, algorithm_name, horizon, plot_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(forecast, label='Forecast', color='red', linestyle='--')
    plt.title(f'{algorithm_name} Forecast vs Actual for {dataset_name} (Horizon: {horizon})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    filename = f'{dataset_name}_{algorithm_name}_h{horizon}_forecast.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()


def plot_forecast_vs_actual_with_history(history, actual, forecast, dataset_name, algorithm_name, horizon, plot_dir):
    full_actual = pd.concat([history, pd.Series(actual)]).reset_index(drop=True)
    full_forecast = pd.concat(
        [history, pd.Series([None] * len(history), dtype=float), pd.Series(forecast)]).reset_index(drop=True)

    plt.figure(figsize=(15, 7))
    plt.plot(full_actual, label='Actual', color='blue')
    plt.plot(full_forecast, label='Forecast', color='red', linestyle='--')
    plt.axvline(x=len(history), color='green', linestyle=':', label='Forecast Start')
    plt.title(f'{algorithm_name} Forecast vs Actual for {dataset_name} (Horizon: {horizon})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    filename = f'{dataset_name}_{algorithm_name}_h{horizon}_forecast_with_history.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()