"""Benchmark strategies evaluated with the same simulator, costs and fill rules
as the MTDC strategy (doc §7A), plus a random-timing null model.
"""

import numpy as np

from directional_changes import identify_dc_events
from symbolic_regression import prepare_data_for_regression, fit_os_predictor
from signals import build_signal_series
from strategy import simulate_trading
from metrics import compute_metrics, annualized_sharpe, bar_returns


def buy_and_hold(open_prices, close_prices, transaction_cost=0.001):
    """Enter long at the first available open, hold to the end."""
    actions = np.zeros(len(close_prices), dtype=np.int8)
    actions[0] = 1
    return simulate_trading(actions, open_prices, close_prices,
                            transaction_cost=transaction_cost, allow_short=False)


def rsi(closes, period=14):
    """Wilder-smoothed RSI. The first `period` values are set to a neutral 50."""
    closes = np.asarray(closes, dtype=np.float64)
    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)

    out = np.full(len(closes), 50.0)
    if len(closes) <= period:
        return out
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def rsi_strategy(open_prices, close_prices, transaction_cost=0.001,
                 period=14, oversold=30.0, overbought=70.0):
    """Classic long-only RSI mean reversion: buy oversold, exit overbought."""
    values = rsi(close_prices, period)
    actions = np.zeros(len(close_prices), dtype=np.int8)
    actions[values < oversold] = 1
    actions[values > overbought] = -1
    return simulate_trading(actions, open_prices, close_prices,
                            transaction_cost=transaction_cost, allow_short=False)


def single_threshold_dc_strategy(close_prices_full, open_prices_full, threshold,
                                 train_slice, eval_slice, transaction_cost=0.001,
                                 allow_short=True, seed=42, gp_kwargs=None):
    """
    The MTDC signal logic restricted to a single threshold: fit the GP predictor
    and signal statistics on the training window, trade sign(signal) on the
    evaluation window. Returns (SimResult, metrics) for the evaluation window.
    """
    gp_kwargs = gp_kwargs or {}
    events = identify_dc_events(close_prices_full, threshold)
    train_end = train_slice.stop
    train_events = [e for e in events if e[3] < train_end]
    if len(train_events) < 10:
        return None, None

    dc_len, os_len = prepare_data_for_regression(train_events)
    predict_fn, _, _ = fit_os_predictor(dc_len, os_len, seed=seed, **gp_kwargs)
    os_log = np.log1p(os_len)
    avg_os = float(os_log.mean())
    sig_level = float(os_log.std())

    sig = build_signal_series(events, predict_fn, len(close_prices_full), avg_os, sig_level)
    actions = np.sign(sig[eval_slice]).astype(np.int8)
    sim = simulate_trading(actions, open_prices_full[eval_slice], close_prices_full[eval_slice],
                           transaction_cost=transaction_cost, allow_short=allow_short)
    return sim, compute_metrics(sim.equity, sim.trades, sim.positions)


def random_timing_null(actions, open_prices, close_prices, transaction_cost=0.001,
                       stop_loss_pct=None, take_profit_pct=None, allow_short=True,
                       n_shuffles=200, seed=0):
    """
    Empirical null distribution: circularly rotate the strategy's action series by
    random offsets (preserving its timing structure while breaking any alignment
    with prices) and re-simulate. Returns (null_returns, null_sharpes).
    """
    rng = np.random.default_rng(seed)
    n = len(actions)
    null_returns = np.empty(n_shuffles)
    null_sharpes = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shift = int(rng.integers(1, n))
        rotated = np.roll(actions, shift)
        sim = simulate_trading(rotated, open_prices, close_prices,
                               transaction_cost=transaction_cost,
                               stop_loss_pct=stop_loss_pct,
                               take_profit_pct=take_profit_pct,
                               allow_short=allow_short)
        null_returns[i] = sim.equity[-1] / sim.equity[0] - 1.0
        null_sharpes[i] = annualized_sharpe(bar_returns(sim.equity))
    return null_returns, null_sharpes


def empirical_p_value(observed, null_values):
    """One-sided p-value: fraction of null outcomes at least as good as observed
    (with the +1 correction)."""
    null_values = np.asarray(null_values)
    return float((1 + np.sum(null_values >= observed)) / (1 + len(null_values)))
