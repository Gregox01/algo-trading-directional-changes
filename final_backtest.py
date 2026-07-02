"""Final one-shot backtest for the asset class that survived the kill-gate grid.

Pre-registered rules (fixed before touching any test region):
- Universe: the PURSUE class from PROFITABILITY_SEARCH.md (crypto).
- Per asset, select ONE cell (timeframe x threshold) using TRAIN data only:
  the cell must have passed the gate (OOF AUC > max(0.52, noise band) ) AND had
  a positive top-tercile net tradeable return in train; among those, pick the
  highest logistic OOF AUC. Assets with no qualifying cell are skipped.
- Model: logistic regression (scaled, C=1) fit on ALL train events of the cell.
  Trade filter: predicted probability >= the train top-tercile cutoff.
- Execution: enter at the open of the bar AFTER confirmation, in the event's
  direction (long/short); exit at the open of the bar after the NEXT
  same-threshold confirmation. Costs: 0.12% round trip (perp taker + slippage).
- Evaluation: test region only (bars >= 60% of each series), touched exactly
  once. Pooled and per-asset stats, plus a label-rotation null.

This script appends its results to PROFITABILITY_SEARCH.md.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_event_dataset, label_exceeds_cost, FEATURE_COLS
from killgate_study import load_prices, oof_auc
from multi_asset_study import robust_sigma, VOL_WINDOWS, VOL_MULTIPLES, resample_ohlc

COST_RT = 0.0012
TRAIN_FRAC = 0.6
SEED = 42
ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']


def load_asset(sym, tf):
    if tf == '4h':
        df = load_prices(f'marketdata/{sym}_1h.csv.gz')
        return resample_ohlc(df, '4h')
    return load_prices(f'marketdata/{sym}_{tf}.csv.gz')


def cell_events(sym, tf):
    df = load_asset(sym, tf)
    closes = df['Close'].values.astype(np.float64)
    opens = df['Open'].values.astype(np.float64)
    train_end = int(len(closes) * TRAIN_FRAC)
    sigma = robust_sigma(closes[:train_end], VOL_WINDOWS[tf])
    thresholds = [round(m * sigma, 6) for m in VOL_MULTIPLES]
    return df, closes, opens, train_end, thresholds


def select_cell(search_results, sym):
    """Pick the asset's cell per the pre-registered rule from train-only stats."""
    def passes(r):
        floor = max(0.52, (r.get('shuffle_auc_95band') or [0, 1])[1])
        return any((r.get(f'auc_{m}') or 0) > floor for m in ('logistic', 'gboost'))

    cands = [r for r in search_results
             if r['label'].startswith(sym) and r['asset_class'] == 'crypto'
             and passes(r)
             and max(r.get('top_tercile_net_logistic') or -1,
                     r.get('top_tercile_net_gboost') or -1) > 0]
    if not cands:
        return None
    return max(cands, key=lambda r: r.get('auc_logistic') or 0)


def run_asset(sym, cell):
    tf = cell['timeframe']
    theta = cell['theta']
    df, closes, opens, train_end, thresholds = cell_events(sym, tf)
    ti = int(np.argmin([abs(t - theta) for t in thresholds]))

    ev = build_event_dataset(closes, thresholds, ti, dates=df.index,
                             vol_window=VOL_WINDOWS[tf])
    train = ev[ev['conf_idx'] < train_end]
    # Keep ev's positional index on test rows: each trade's exit is defined by
    # the NEXT event in ev order, so positions must refer to ev, not to a
    # reset-index copy (a reset here silently mapped exits onto train events
    # and produced zero trades).
    test = ev[ev['conf_idx'] >= train_end]
    if len(test) < 10:
        return None

    y_train = label_exceeds_cost(train, COST_RT).values
    X_train = train[FEATURE_COLS].values.astype(np.float64)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    model.fit(X_train, y_train)
    cutoff = float(np.quantile(model.predict_proba(X_train)[:, 1], 2 / 3))

    X_test = test[FEATURE_COLS].values.astype(np.float64)
    p = model.predict_proba(X_test)[:, 1]
    take = p >= cutoff

    # Realistic fills: enter at open of bar conf+1, exit at open of bar next_conf+1.
    n_bars = len(closes)
    trades = []
    test_idx = test.index.to_numpy()
    conf = test['conf_idx'].to_numpy()
    direction = test['direction'].to_numpy()
    for i in range(len(test)):
        if not take[i]:
            continue
        entry_bar = int(conf[i]) + 1
        # next same-threshold confirmation: the following event row in ev order
        pos_in_ev = test_idx[i]
        if pos_in_ev + 1 >= len(ev):
            continue
        exit_conf = int(ev.iloc[pos_in_ev + 1]['conf_idx'])
        exit_bar = exit_conf + 1
        if entry_bar >= n_bars or exit_bar >= n_bars or exit_bar <= entry_bar:
            continue
        gross = direction[i] * (opens[exit_bar] - opens[entry_bar]) / opens[entry_bar]
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'direction': int(direction[i]),
            'gross': float(gross), 'net': float(gross - COST_RT),
            'entry_time': str(df.index[entry_bar]),
        })

    rets = np.array([t['net'] for t in trades])
    # Compounded equity assuming sequential full-equity trades
    equity_final = float(np.prod(1 + rets)) if len(rets) else 1.0
    test_bh = closes[-1] / closes[train_end] - 1.0

    # Rotation null: same number of trades, random event subset (label rotation)
    rng = np.random.default_rng(SEED)
    null_means = []
    all_test_rets = []
    for i in range(len(test)):
        pos_in_ev = test_idx[i]
        if pos_in_ev + 1 >= len(ev):
            continue
        entry_bar = int(conf[i]) + 1
        exit_bar = int(ev.iloc[pos_in_ev + 1]['conf_idx']) + 1
        if entry_bar >= n_bars or exit_bar >= n_bars or exit_bar <= entry_bar:
            continue
        all_test_rets.append(direction[i] * (opens[exit_bar] - opens[entry_bar]) / opens[entry_bar] - COST_RT)
    all_test_rets = np.array(all_test_rets)
    k = len(rets)
    if k and len(all_test_rets) > k:
        for _ in range(1000):
            null_means.append(rng.choice(all_test_rets, size=k, replace=False).mean())
        null_means = np.array(null_means)
        p_sel = float((1 + np.sum(null_means >= rets.mean())) / (1 + len(null_means)))
    else:
        p_sel = np.nan

    return {
        'symbol': sym, 'timeframe': tf, 'theta': theta,
        'train_auc_logit': cell.get('auc_logistic'),
        'n_test_events': int(len(test)), 'n_trades': int(k),
        'mean_net_per_trade': float(rets.mean()) if k else 0.0,
        'median_net_per_trade': float(np.median(rets)) if k else 0.0,
        'win_rate': float((rets > 0).mean()) if k else 0.0,
        'total_compounded': equity_final - 1.0,
        'all_events_mean_net': float(all_test_rets.mean()) if len(all_test_rets) else 0.0,
        'buy_hold_test': float(test_bh),
        'p_selection_vs_all_events': p_sel,
        'test_period': f"{run_period(df, train_end)}",
        'trades': trades,
    }


def run_period(df, train_end):
    return f"{df.index[train_end].date()} .. {df.index[-1].date()}"


def main():
    with open('results/profitability_search.json') as f:
        search = json.load(f)

    per_asset = []
    for sym in ASSETS:
        cell = select_cell(search, sym)
        if cell is None:
            print(f"{sym}: no qualifying cell - skipped")
            continue
        print(f"{sym}: selected {cell['timeframe']} theta={cell['theta']:.3%} "
              f"(train logit AUC {cell.get('auc_logistic'):.3f})")
        r = run_asset(sym, cell)
        if r:
            per_asset.append(r)
            print(f"  -> {r['n_trades']} trades, mean net {r['mean_net_per_trade']:+.3%}/trade, "
                  f"win rate {r['win_rate']:.0%}, compounded {r['total_compounded']:+.1%}, "
                  f"B&H {r['buy_hold_test']:+.1%}, p(selection) {r['p_selection_vs_all_events']:.3f}")

    # Pooled per-trade stats
    pooled = np.concatenate([[t['net'] for t in r['trades']] for r in per_asset]) \
        if per_asset else np.array([])
    if len(pooled):
        boot = []
        rng = np.random.default_rng(SEED)
        for _ in range(5000):
            boot.append(rng.choice(pooled, size=len(pooled), replace=True).mean())
        lo, hi = np.quantile(boot, [0.025, 0.975])
    else:
        lo = hi = np.nan

    with open('results/final_backtest.json', 'w') as f:
        json.dump({'per_asset': [{k: v for k, v in r.items() if k != 'trades'}
                                 for r in per_asset],
                   'pooled_mean_net': float(pooled.mean()) if len(pooled) else None,
                   'pooled_n': int(len(pooled)),
                   'pooled_ci95': [float(lo), float(hi)]}, f, indent=2)

    lines = ["", "## Final one-shot test (crypto, untouched regions)", ""]
    lines.append("Selection rules fixed on train only (see final_backtest.py docstring); "
                 "each asset's test region (last 40% of its history) evaluated exactly once.")
    lines.append("")
    lines.append("| Asset | Cell | Test period | Events | Trades | Mean net/trade | Win rate | Compounded | All-events mean net | B&H | p(selection) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in per_asset:
        lines.append(
            f"| {r['symbol']} | {r['timeframe']} θ={r['theta']:.2%} | {r['test_period']} | "
            f"{r['n_test_events']} | {r['n_trades']} | {r['mean_net_per_trade']:+.3%} | "
            f"{r['win_rate']:.0%} | {r['total_compounded']:+.1%} | "
            f"{r['all_events_mean_net']:+.3%} | {r['buy_hold_test']:+.1%} | "
            f"{r['p_selection_vs_all_events']:.3f} |")
    lines.append("")
    if len(pooled):
        lines.append(f"**Pooled: {len(pooled)} trades, mean net {pooled.mean():+.3%}/trade, "
                     f"95% bootstrap CI [{lo:+.3%}, {hi:+.3%}].**")
    lines.append("")
    lines.append("`p(selection)` asks whether the model's trade filter beats trading ALL "
                 "test events of the same cell (1000 random same-size subsets); "
                 "the all-events column is the unfiltered DC-following baseline.")
    lines.append("")
    with open('PROFITABILITY_SEARCH.md', 'a') as f:
        f.write("\n".join(lines))
    print("\nAppended final test to PROFITABILITY_SEARCH.md")


if __name__ == "__main__":
    main()
