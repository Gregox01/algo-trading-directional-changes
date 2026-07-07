import numpy as np
import pytest

from metrics import (annualized_sharpe, bar_returns, bootstrap_sharpe_ci,
                     compute_metrics, daily_returns, max_drawdown)
from strategy import Trade


def make_trade(pnl):
    return Trade(direction=1, entry_bar=0, exit_bar=1,
                 entry_price=100.0, exit_price=101.0, pnl=pnl, ret=pnl / 10_000.0)


def test_monotonic_equity_zero_drawdown():
    equity = np.linspace(10_000, 20_000, 100)
    assert max_drawdown(equity) == 0.0


def test_known_drawdown():
    equity = np.array([100.0, 120.0, 90.0, 130.0])
    assert max_drawdown(equity) == pytest.approx(0.25)


def test_sharpe_matches_manual_computation():
    returns = np.array([0.01, 0.02, -0.01, 0.0, 0.005])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(365 * 96)
    assert annualized_sharpe(returns) == pytest.approx(expected)


def test_sharpe_zero_for_constant_returns():
    assert annualized_sharpe(np.full(100, 0.01)) == 0.0


def test_profit_factor_edge_cases():
    all_wins = compute_metrics(np.array([100.0, 110.0]), trades=[make_trade(50.0)])
    assert all_wins['profit_factor'] == float('inf')
    no_trades = compute_metrics(np.array([100.0, 100.0]), trades=[])
    assert no_trades['n_trades'] == 0
    assert no_trades['profit_factor'] == 0.0


def test_compute_metrics_cumulative_return():
    equity = np.array([10_000.0, 11_000.0, 12_100.0])
    m = compute_metrics(equity)
    assert m['cumulative_return'] == pytest.approx(0.21)


def test_daily_aggregation():
    returns = np.full(96 * 3 + 10, 0.001)
    daily = daily_returns(returns)
    assert len(daily) == 3
    assert daily[0] == pytest.approx(0.096)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0005, 0.01, 5000)
    point = annualized_sharpe(returns)
    lo, hi = bootstrap_sharpe_ci(returns, n_boot=200, seed=1)
    assert lo < point < hi


def test_exposure():
    equity = np.array([100.0, 100.0, 100.0, 100.0])
    positions = np.array([0, 1, 1, 0], dtype=np.int8)
    m = compute_metrics(equity, positions=positions)
    assert m['exposure'] == pytest.approx(0.5)
