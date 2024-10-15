import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta

def backtest_strategy(price_series, signals, initial_capital=10000, transaction_cost=0.0005, start_date=None):
    capital = initial_capital
    position = 0
    entry_price = None
    trades = []
    trade_log = []

    if start_date is None:
        start_date = datetime(2023, 1, 1)  # Use an appropriate default start date

    for i, (index, signal, _) in enumerate(signals):
        current_price = price_series[index]
        current_time = start_date + timedelta(minutes=15*index)  # Assuming 15-minute intervals

        if signal != 0 and signal != position:
            # Close existing position if any
            if position != 0:
                exit_price = current_price * (1 - transaction_cost * abs(position))
                profit = (exit_price - entry_price) * position
                capital += profit
                trades.append((entry_index, index, entry_price, exit_price, profit))
                trade_log.append({
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': entry_price,
                    'direction': 'Long' if position == 1 else 'Short',
                    'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_price': exit_price,
                    'profit': profit
                })

            # Open new position
            position = signal
            entry_price = current_price * (1 + transaction_cost * abs(position))
            entry_index = index
            entry_time = current_time

    # Close any remaining position at the end
    if position != 0:
        exit_price = price_series[-1] * (1 - transaction_cost * abs(position))
        profit = (exit_price - entry_price) * position
        capital += profit
        trades.append((entry_index, len(price_series) - 1, entry_price, exit_price, profit))
        trade_log.append({
            'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price': entry_price,
            'direction': 'Long' if position == 1 else 'Short',
            'exit_time': (start_date + timedelta(minutes=15*(len(price_series)-1))).strftime('%Y-%m-%d %H:%M:%S'),
            'exit_price': exit_price,
            'profit': profit
        })

    # Save trades to CSV
    with open('trades.csv', 'w', newline='') as csvfile:
        fieldnames = ['entry_time', 'entry_price', 'direction', 'exit_time', 'exit_price', 'profit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trade_log:
            writer.writerow(trade)

    return capital, trades, (capital - initial_capital) / initial_capital * 100, trade_log

def calculate_performance_metrics(trades, initial_capital=10000):
    if not trades:
        print("No trades were executed.")
        return

    profits = [trade[4] for trade in trades]
    total_profit = sum(profits)
    total_return = total_profit / initial_capital * 100
    win_rate = sum(1 for p in profits if p > 0) / len(profits)

    # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    returns = np.array(profits) / initial_capital
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

    # Calculate Max Drawdown
    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = peak - cumulative_returns
    max_drawdown = drawdown.max()

    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
