"""Performance metrics and statistical significance helpers.

Annualization assumes 24/7 crypto trading: 96 fifteen-minute bars per day,
365 days per year. The design document's Nemenyi test is replaced with a block
bootstrap confidence interval plus a Wilcoxon signed-rank test on daily returns —
equivalent evidence for a two-strategy comparison with one fewer dependency.
"""

import numpy as np
from scipy import stats

BARS_PER_DAY = 96
BARS_PER_YEAR = 365 * BARS_PER_DAY


def bar_returns(equity):
    equity = np.asarray(equity, dtype=np.float64)
    return np.diff(equity) / equity[:-1]


def annualized_sharpe(returns, bars_per_year=BARS_PER_YEAR):
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return 0.0
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(bars_per_year))


def annualized_sortino(returns, bars_per_year=BARS_PER_YEAR):
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0.0:
        return 0.0
    return float(returns.mean() / downside_std * np.sqrt(bars_per_year))


def max_drawdown(equity):
    equity = np.asarray(equity, dtype=np.float64)
    running_max = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / running_max
    return float(drawdowns.max())


def compute_metrics(equity, trades=None, positions=None, bars_per_year=BARS_PER_YEAR):
    """Full metric set for one equity curve (plus optional trade list/positions)."""
    equity = np.asarray(equity, dtype=np.float64)
    returns = bar_returns(equity)
    n_bars = len(equity)
    years = n_bars / bars_per_year

    cumulative = float(equity[-1] / equity[0] - 1.0)
    with np.errstate(over='ignore'):
        cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    out = {
        'cumulative_return': cumulative,
        'cagr': cagr,
        'sharpe': annualized_sharpe(returns, bars_per_year),
        'sortino': annualized_sortino(returns, bars_per_year),
        'max_drawdown': max_drawdown(equity),
        'n_bars': int(n_bars),
    }

    if trades is not None:
        pnls = np.array([t.pnl for t in trades], dtype=np.float64)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(-losses.sum()) if len(losses) else 0.0
        out.update({
            'n_trades': int(len(pnls)),
            'win_rate': float(len(wins) / len(pnls)) if len(pnls) else 0.0,
            'profit_factor': (gross_profit / gross_loss) if gross_loss > 0
                             else (float('inf') if gross_profit > 0 else 0.0),
            'avg_trade_return': float(np.mean([t.ret for t in trades])) if len(pnls) else 0.0,
        })

    if positions is not None:
        positions = np.asarray(positions)
        out['exposure'] = float(np.mean(positions != 0))

    return out


def block_bootstrap_indices(n, block, rng):
    """Moving-block bootstrap: indices for one resampled series of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return idx


def bootstrap_sharpe_ci(returns, n_boot=2000, block=BARS_PER_DAY, seed=0,
                        bars_per_year=BARS_PER_YEAR, ci=0.95):
    """Moving-block bootstrap CI for the annualized Sharpe ratio."""
    returns = np.asarray(returns, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = block_bootstrap_indices(len(returns), block, rng)
        sharpes[b] = annualized_sharpe(returns[idx], bars_per_year)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(sharpes, alpha)), float(np.quantile(sharpes, 1.0 - alpha))


def bootstrap_return_diff_ci(returns_a, returns_b, n_boot=2000, block=BARS_PER_DAY,
                             seed=0, ci=0.95):
    """
    Moving-block bootstrap CI for the mean bar-return difference (a - b). The two
    series are resampled with the same time blocks (paired), preserving their
    cross-correlation.
    """
    a = np.asarray(returns_a, dtype=np.float64)
    b = np.asarray(returns_b, dtype=np.float64)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_bootstrap_indices(n, block, rng)
        means[i] = diff[idx].mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def daily_returns(returns, bars_per_day=BARS_PER_DAY):
    returns = np.asarray(returns, dtype=np.float64)
    n_days = len(returns) // bars_per_day
    if n_days == 0:
        return returns.copy()
    trimmed = returns[:n_days * bars_per_day]
    return trimmed.reshape(n_days, bars_per_day).sum(axis=1)


def wilcoxon_daily(returns_a, returns_b, bars_per_day=BARS_PER_DAY):
    """Wilcoxon signed-rank test on the daily-aggregated return differences."""
    a = daily_returns(returns_a, bars_per_day)
    b = daily_returns(returns_b, bars_per_day)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    if np.allclose(diff, 0.0):
        return float('nan')
    return float(stats.wilcoxon(diff).pvalue)
