import os
import sys
from data_preparation import prepare_all_data
from symbolic_regression_gp import run_symbolic_regression
from signal_generation import generate_signals_multi_threshold
from backtesting import backtest_strategy, calculate_performance_metrics
from utils import save_checkpoint
import numpy as np
from datetime import datetime

def main(file_path):
    thresholds = [0.01, 0.015, 0.02]

    # Data preparation
    price_series, data = prepare_all_data(file_path, thresholds)

    models = []
    best_individuals = []
    for i, (dc_events, os_events, X, y, scaler_X, scaler_y) in enumerate(data):
        # Symbolic regression
        best_individual, toolbox = run_symbolic_regression(X, y, f'srgp_checkpoint_{i}.pkl')
        best_individuals.append(best_individual)

        func = toolbox.compile(expr=best_individual)
        models.append(func)

        # Save the best model
        save_checkpoint((best_individual, toolbox), f'best_model_{i}.pkl')

        print(f"Best individual for threshold {thresholds[i]}: {best_individual}")

    # Generate signals
    dc_events_list = [d[0] for d in data]
    scaler_X_list = [d[4] for d in data]
    scaler_y_list = [d[5] for d in data]

    # Initialize weights (equal weights for all thresholds)
    weights = np.ones(len(thresholds)) / len(thresholds)

    signals = generate_signals_multi_threshold(dc_events_list, scaler_X_list, scaler_y_list, models, thresholds, price_series, weights)

    # Backtesting
    initial_capital = 10000
    start_date = datetime(2023, 1, 1)  # Replace with the actual start date of your data
    final_capital, trades, return_percentage, trade_log = backtest_strategy(price_series, signals, initial_capital=initial_capital, start_date=start_date)

    # Performance metrics
    calculate_performance_metrics(trades, initial_capital=initial_capital)

    print(f"Final capital: {final_capital}")
    print(f"Return percentage: {return_percentage}%")
    print(f"Trades have been saved to 'trades.csv'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_csv_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)
