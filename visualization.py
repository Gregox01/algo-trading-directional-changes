import numpy as np
import matplotlib.pyplot as plt

def plot_results(X_train, y_train, X_test, y_test, best_individual, toolbox):
    func = toolbox.compile(expr=best_individual)
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    y_pred_plot = [func(x[0]) for x in X_plot]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
    plt.scatter(X_test, y_test, alpha=0.5, label='Testing Data')
    plt.plot(X_plot, y_pred_plot, 'r-', label='Model Prediction')
    plt.xlabel('Normalized DC Length')
    plt.ylabel('Normalized OS Length')
    plt.title('Symbolic Regression Results')
    plt.legend()
    plt.show()

def plot_equity_curve(trades, initial_capital=10000):
    cumulative_profits = np.cumsum([trade[4] for trade in trades])
    equity = initial_capital + cumulative_profits

    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Equity Curve')
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Capital')
    plt.legend()
    plt.show()