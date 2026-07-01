"""End-to-end MTDC backtest pipeline.

Runs the full system described in predictor_algo_documentation.md — DC event
detection at multiple thresholds, GP prediction of overshoot lengths, weighted
multi-threshold voting with GA-optimized weights and risk parameters — under a
strict chronological train/validation/test protocol, then evaluates the test
window against benchmarks and writes RESULTS.md plus charts to results/.

Usage:
    python run_backtest.py            # full run (~3-6 minutes)
    python run_backtest.py --quick    # tiny GP/GA budgets, plumbing check only
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from directional_changes import load_and_prepare_data, identify_dc_events
from symbolic_regression import prepare_data_for_regression, fit_os_predictor
from signals import build_signal_series, build_confidence_matrices, compute_event_features, signal_strengths
from strategy import weighted_voting, simulate_trading
from metrics import (compute_metrics, bar_returns, annualized_sharpe,
                     bootstrap_sharpe_ci, bootstrap_return_diff_ci, wilcoxon_daily)
from ga_optimizer import optimize_weights, select_on_validation, evaluate_genome, decode_genome
from benchmarks import buy_and_hold, rsi_strategy, random_timing_null, empirical_p_value

DEFAULT_THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03]
MIN_TRAIN_EVENTS = 50


def parse_args():
    p = argparse.ArgumentParser(description="MTDC strategy backtest")
    p.add_argument("--data", default="ETHUSDT_15m.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cost", type=float, default=0.001,
                   help="headline per-side transaction cost (0.001 = 0.1%%)")
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.add_argument("--gp-pop", type=int, default=200)
    p.add_argument("--gp-gens", type=int, default=30)
    p.add_argument("--ga-pop", type=int, default=40)
    p.add_argument("--ga-gens", type=int, default=30)
    p.add_argument("--outdir", default="results")
    p.add_argument("--quick", action="store_true", help="tiny GP/GA budgets for a plumbing check")
    return p.parse_args()


def fit_threshold_models(closes, thresholds, fit_end, seed, gp_pop, gp_gens):
    """
    For each threshold: detect DC events on the full series (detection is causal),
    fit the GP OS-length predictor and the signal statistics on events confirmed
    before fit_end, and build the bar-space signal series over all bars.
    Thresholds with too few training events are dropped.
    """
    models = []
    n_bars = len(closes)
    for theta in thresholds:
        events = identify_dc_events(closes, theta)
        train_events = [e for e in events if e[3] < fit_end]
        if len(train_events) < MIN_TRAIN_EVENTS:
            print(f"  threshold {theta:.3%}: only {len(train_events)} train events - dropped")
            continue
        dc_len, os_len = prepare_data_for_regression(train_events)
        predict_fn, expr, info = fit_os_predictor(
            dc_len, os_len, seed=seed, population_size=gp_pop, num_generations=gp_gens)
        avg_os = float(os_len.mean())
        sig_level = float(os_len.std())
        sig = build_signal_series(events, predict_fn, n_bars, avg_os, sig_level)
        models.append({
            'threshold': theta,
            'events': events,
            'n_train_events': len(train_events),
            'predict_fn': predict_fn,
            'expr': expr,
            'gp_info': info,
            'avg_os': avg_os,
            'sig_level': sig_level,
            'signal': sig,
            'train_dc_len': dc_len,
            'train_os_len': os_len,
        })
        print(f"  threshold {theta:.3%}: {len(train_events)} train events, "
              f"GP beats mean baseline: {info['beats_mean_baseline']}")
    if not models:
        raise RuntimeError("No threshold produced enough training events")
    return models


def build_naive_lookahead_signals(models, n_bars):
    """
    The cautionary variant: identical strengths and horizons, but signal windows
    start at the *extremum* bar — information that does not exist yet in real
    time. Only used for the RESULTS.md appendix.
    """
    naive = []
    for m in models:
        feats = compute_event_features(m['events'])
        predicted = np.atleast_1d(np.asarray(m['predict_fn'](feats['dc_len']), dtype=float))
        strengths = signal_strengths(predicted, m['avg_os'], m['sig_level'])
        sig = np.zeros(n_bars, dtype=np.int8)
        ext_idx = feats['ext_idx']
        direction = feats['direction']
        for k in range(len(ext_idx)):
            s = int(strengths[k]) * int(direction[k])
            if s == 0:
                continue
            start = int(ext_idx[k])
            end = min(start + int(max(1, round(predicted[k]))), n_bars)
            if k + 1 < len(ext_idx):
                end = min(end, int(ext_idx[k + 1]))
            if end > start:
                sig[start:end] = s
        naive.append(sig)
    return build_confidence_matrices(naive)


def run_fold(closes, opens, fold_start, fold_end, cfg, label=""):
    """
    One complete experiment: GP + GA fitted on [0, ga_train_end), champion picked
    on [ga_train_end, fold_start), tested on [fold_start, fold_end). All model
    fitting sees nothing at or after fold_start.
    """
    n_bars = len(closes)
    ga_train_end = int(fold_start * 0.75)
    train_sl = slice(0, ga_train_end)
    val_sl = slice(ga_train_end, fold_start)
    test_sl = slice(fold_start, fold_end)

    print(f"[fold {label}] train [0, {ga_train_end}), val [{ga_train_end}, {fold_start}), "
          f"test [{fold_start}, {fold_end})")

    models = fit_threshold_models(closes, cfg['thresholds'], ga_train_end,
                                  cfg['seed'], cfg['gp_pop'], cfg['gp_gens'])
    B, S, H = build_confidence_matrices([m['signal'] for m in models])

    hof, _ = optimize_weights(
        B[:, train_sl], S[:, train_sl], H[:, train_sl],
        opens[train_sl], closes[train_sl],
        transaction_cost=cfg['cost'], allow_short=True, seed=cfg['seed'],
        population_size=cfg['ga_pop'], num_generations=cfg['ga_gens'])

    champion, val_fitness = select_on_validation(
        hof, B[:, val_sl], S[:, val_sl], H[:, val_sl],
        opens[val_sl], closes[val_sl], transaction_cost=cfg['cost'])

    train_fitness, _, _ = evaluate_genome(
        champion, B[:, train_sl], S[:, train_sl], H[:, train_sl],
        opens[train_sl], closes[train_sl], cfg['cost'])

    # Test-window evaluation (the only look at [fold_start, fold_end))
    test_fitness, sim_ls, m_ls = evaluate_genome(
        champion, B[:, test_sl], S[:, test_sl], H[:, test_sl],
        opens[test_sl], closes[test_sl], cfg['cost'])

    weights, stop_loss, take_profit = decode_genome(champion, B.shape[0])
    actions_test = weighted_voting(B[:, test_sl], S[:, test_sl], H[:, test_sl], weights)
    sim_lo = simulate_trading(actions_test, opens[test_sl], closes[test_sl],
                              transaction_cost=cfg['cost'], stop_loss_pct=stop_loss,
                              take_profit_pct=take_profit, allow_short=False)
    m_lo = compute_metrics(sim_lo.equity, sim_lo.trades, sim_lo.positions)

    return {
        'label': label,
        'slices': (train_sl, val_sl, test_sl),
        'models': models,
        'B': B, 'S': S, 'H': H,
        'champion': champion,
        'weights': weights, 'stop_loss': stop_loss, 'take_profit': take_profit,
        'train_fitness': train_fitness, 'val_fitness': val_fitness,
        'test_fitness': test_fitness,
        'actions_test': actions_test,
        'sim_ls': sim_ls, 'metrics_ls': m_ls,
        'sim_lo': sim_lo, 'metrics_lo': m_lo,
    }


def evaluate_benchmarks(fold, closes, opens, cost):
    """Benchmarks on the fold's test window, single-threshold picked on validation."""
    _, val_sl, test_sl = fold['slices']
    out = {}

    sim_bh = buy_and_hold(opens[test_sl], closes[test_sl], transaction_cost=cost)
    out['Buy & Hold'] = (sim_bh, compute_metrics(sim_bh.equity, sim_bh.trades, sim_bh.positions))

    sim_rsi = rsi_strategy(opens[test_sl], closes[test_sl], transaction_cost=cost)
    out['RSI(14) 30/70'] = (sim_rsi, compute_metrics(sim_rsi.equity, sim_rsi.trades, sim_rsi.positions))

    # Best single threshold, chosen on the validation window
    best = None
    for m in fold['models']:
        actions_val = np.sign(m['signal'][val_sl]).astype(np.int8)
        sim_val = simulate_trading(actions_val, opens[val_sl], closes[val_sl],
                                   transaction_cost=cost)
        mv = compute_metrics(sim_val.equity, sim_val.trades)
        fitness = mv['sharpe'] - 2.0 * mv['max_drawdown']
        if best is None or fitness > best[1]:
            best = (m, fitness)
    m_best = best[0]
    actions_test = np.sign(m_best['signal'][test_sl]).astype(np.int8)
    sim_st = simulate_trading(actions_test, opens[test_sl], closes[test_sl],
                              transaction_cost=cost)
    name = f"Single-threshold DC ({m_best['threshold']:.1%})"
    out[name] = (sim_st, compute_metrics(sim_st.equity, sim_st.trades, sim_st.positions))
    return out


def fmt_pct(x):
    return f"{x * 100:+.2f}%"


def metrics_row(name, m):
    pf = m.get('profit_factor', float('nan'))
    pf_str = "inf" if pf == float('inf') else f"{pf:.2f}"
    return (f"| {name} | {fmt_pct(m['cumulative_return'])} | {m['sharpe']:.2f} | "
            f"{m['sortino']:.2f} | {m['max_drawdown'] * 100:.1f}% | "
            f"{m.get('n_trades', 0)} | {m.get('win_rate', 0) * 100:.0f}% | {pf_str} |")


TABLE_HEADER = ("| Strategy | Return | Sharpe | Sortino | Max DD | Trades | Win rate | Profit factor |\n"
                "|---|---|---|---|---|---|---|---|")


def make_charts(fold, bench, closes, opens, dates, cost_rows, wf_rows, outdir):
    _, _, test_sl = fold['slices']
    test_dates = dates[test_sl]

    # Equity curves
    plt.figure(figsize=(13, 6))
    curves = [("MTDC (long/short)", fold['sim_ls'].equity),
              ("MTDC (long-only)", fold['sim_lo'].equity)]
    curves += [(name, sim.equity) for name, (sim, _) in bench.items()]
    for name, eq in curves:
        plt.plot(test_dates, eq / eq[0], label=name, lw=1.2)
    plt.legend()
    plt.title("Test window equity (normalized), 0.1% cost per side")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, "equity_test.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Drawdowns
    plt.figure(figsize=(13, 4))
    for name, eq in [("MTDC (long/short)", fold['sim_ls'].equity),
                     ("Buy & Hold", bench['Buy & Hold'][0].equity)]:
        dd = eq / np.maximum.accumulate(eq) - 1.0
        plt.plot(test_dates, dd * 100, label=name, lw=1.0)
    plt.legend()
    plt.title("Test window drawdown")
    plt.ylabel("Drawdown (%)")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, "drawdown_test.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # GP fits
    models = fold['models']
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5), squeeze=False)
    for ax, m in zip(axes[0], models):
        ax.scatter(m['train_dc_len'], m['train_os_len'], s=6, alpha=0.25, label="train events")
        x_plot = np.linspace(1, m['train_dc_len'].max(), 200)
        ax.plot(x_plot, m['predict_fn'](x_plot), 'r-', lw=1.5, label="GP")
        ax.axhline(m['avg_os'], color='gray', ls='--', lw=0.8, label="train avg OS")
        ax.set_title(f"θ = {m['threshold']:.1%}")
        ax.set_xscale('log')
        ax.set_yscale('symlog')
        ax.set_xlabel("DC length (bars)")
    axes[0][0].set_ylabel("OS length (bars)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("GP overshoot-length predictors (fit on train only)")
    fig.savefig(os.path.join(outdir, "gp_fits.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Champion weights
    plt.figure(figsize=(7, 4))
    labels = [f"{m['threshold']:.1%}" for m in models]
    plt.bar(labels, fold['weights'])
    plt.title(f"Champion threshold weights (SL {fold['stop_loss']:.1%}, TP {fold['take_profit']:.1%})")
    plt.ylabel("Weight")
    plt.savefig(os.path.join(outdir, "weights.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Cost sensitivity
    plt.figure(figsize=(7, 4))
    costs = [r[0] for r in cost_rows]
    plt.plot(costs, [r[1] * 100 for r in cost_rows], 'o-', label="MTDC long/short")
    plt.plot(costs, [r[2] * 100 for r in cost_rows], 's-', label="MTDC long-only")
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel("Per-side transaction cost")
    plt.ylabel("Test return (%)")
    plt.title("Cost sensitivity (test window)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, "cost_sensitivity.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Walk-forward
    plt.figure(figsize=(8, 4))
    idx = np.arange(len(wf_rows))
    width = 0.35
    plt.bar(idx - width / 2, [r['mtdc_return'] * 100 for r in wf_rows], width, label="MTDC L/S")
    plt.bar(idx + width / 2, [r['bh_return'] * 100 for r in wf_rows], width, label="Buy & Hold")
    plt.axhline(0, color='k', lw=0.8)
    plt.xticks(idx, [r['period'] for r in wf_rows], fontsize=8)
    plt.ylabel("Fold return (%)")
    plt.title("Walk-forward folds (each refit on data before the fold)")
    plt.legend()
    plt.savefig(os.path.join(outdir, "walk_forward.png"), dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    if args.quick:
        args.gp_pop, args.gp_gens = 50, 5
        args.ga_pop, args.ga_gens = 12, 5

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    cfg = {
        'thresholds': args.thresholds, 'seed': args.seed, 'cost': args.cost,
        'gp_pop': args.gp_pop, 'gp_gens': args.gp_gens,
        'ga_pop': args.ga_pop, 'ga_gens': args.ga_gens,
    }

    print("Loading data...")
    df = load_and_prepare_data(args.data)
    closes = df['Close'].values.astype(np.float64)
    opens = df['Open'].values.astype(np.float64)
    dates = df.index
    n = len(closes)
    print(f"{n} bars, {dates[0]} to {dates[-1]}")

    # ---- Headline experiment: train 60% / val 20% / test 20% --------------------
    test_start = int(n * 0.8)
    fold = run_fold(closes, opens, test_start, n, cfg, label="headline")
    _, val_sl, test_sl = fold['slices']

    bench = evaluate_benchmarks(fold, closes, opens, args.cost)

    # Significance on the test window
    ret_ls = bar_returns(fold['sim_ls'].equity)
    ret_bh = bar_returns(bench['Buy & Hold'][0].equity)
    sharpe_ci = bootstrap_sharpe_ci(ret_ls, seed=args.seed)
    diff_ci = bootstrap_return_diff_ci(ret_ls, ret_bh, seed=args.seed)
    wilcoxon_p = wilcoxon_daily(ret_ls, ret_bh)
    null_returns, _ = random_timing_null(
        fold['actions_test'], opens[test_sl], closes[test_sl],
        transaction_cost=args.cost, stop_loss_pct=fold['stop_loss'],
        take_profit_pct=fold['take_profit'], n_shuffles=200, seed=args.seed)
    null_p = empirical_p_value(fold['metrics_ls']['cumulative_return'], null_returns)

    # Cost sensitivity (same champion, same signals, varying cost)
    cost_rows = []
    for c in [0.0, 0.00025, 0.0005, 0.001]:
        _, sim_c, m_c = evaluate_genome(
            fold['champion'], fold['B'][:, test_sl], fold['S'][:, test_sl],
            fold['H'][:, test_sl], opens[test_sl], closes[test_sl], c)
        actions = fold['actions_test']
        sim_c_lo = simulate_trading(actions, opens[test_sl], closes[test_sl],
                                    transaction_cost=c, stop_loss_pct=fold['stop_loss'],
                                    take_profit_pct=fold['take_profit'], allow_short=False)
        cost_rows.append((c, m_c['cumulative_return'],
                          sim_c_lo.equity[-1] / sim_c_lo.equity[0] - 1.0))

    # ---- Walk-forward robustness: three refit folds over the last 40% -----------
    wf_rows = []
    edges = [int(n * f) for f in (0.6, 0.7333, 0.8667, 1.0)]
    for i in range(3):
        a, b = edges[i], edges[i + 1]
        wf = run_fold(closes, opens, a, b, cfg, label=f"wf{i + 1}")
        wf_test = slice(a, b)
        bh = buy_and_hold(opens[wf_test], closes[wf_test], transaction_cost=args.cost)
        wf_rows.append({
            'period': f"{dates[a].date()} - {dates[b - 1].date()}",
            'mtdc_return': wf['metrics_ls']['cumulative_return'],
            'mtdc_sharpe': wf['metrics_ls']['sharpe'],
            'mtdc_dd': wf['metrics_ls']['max_drawdown'],
            'n_trades': wf['metrics_ls']['n_trades'],
            'bh_return': bh.equity[-1] / bh.equity[0] - 1.0,
        })

    # ---- Naive look-ahead appendix ----------------------------------------------
    Bn, Sn, Hn = build_naive_lookahead_signals(fold['models'], n)
    _, sim_naive, m_naive = evaluate_genome(
        fold['champion'], Bn[:, test_sl], Sn[:, test_sl], Hn[:, test_sl],
        opens[test_sl], closes[test_sl], args.cost)

    # ---- Verdict ------------------------------------------------------------------
    m_ls = fold['metrics_ls']
    wf_positive = sum(1 for r in wf_rows if r['mtdc_return'] > 0)
    crit_a = m_ls['cumulative_return'] > 0
    crit_b = m_ls['sharpe'] > 0 and sharpe_ci[0] > 0
    crit_c = wf_positive >= 2
    if crit_a and crit_b and crit_c:
        verdict = "PROFITABLE (by the pre-registered criteria)"
    elif not crit_a:
        verdict = "NOT PROFITABLE"
    else:
        verdict = "INCONCLUSIVE / NOT ROBUSTLY PROFITABLE"

    make_charts(fold, bench, closes, opens, dates, cost_rows, wf_rows, args.outdir)
    write_results(args, fold, bench, dates, cost_rows, wf_rows, m_naive,
                  sharpe_ci, diff_ci, wilcoxon_p, null_p, null_returns,
                  verdict, (crit_a, crit_b, crit_c))

    # Machine-readable summary for reproducibility checks
    summary = {
        'seed': args.seed, 'cost': args.cost,
        'test_return_ls': m_ls['cumulative_return'],
        'test_sharpe_ls': m_ls['sharpe'],
        'test_return_lo': fold['metrics_lo']['cumulative_return'],
        'champion': list(fold['champion']),
        'wf_returns': [r['mtdc_return'] for r in wf_rows],
        'verdict': verdict,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== TEST WINDOW ({} to {}) ====".format(dates[test_sl][0], dates[test_sl][-1]))
    print(f"MTDC long/short: return {fmt_pct(m_ls['cumulative_return'])}, "
          f"Sharpe {m_ls['sharpe']:.2f}, maxDD {m_ls['max_drawdown']:.1%}, "
          f"{m_ls['n_trades']} trades")
    print(f"Buy & Hold:      return {fmt_pct(bench['Buy & Hold'][1]['cumulative_return'])}")
    print(f"\nVERDICT: {verdict}")
    print(f"Report written to RESULTS.md, charts to {args.outdir}/")


def write_results(args, fold, bench, dates, cost_rows, wf_rows, m_naive,
                  sharpe_ci, diff_ci, wilcoxon_p, null_p, null_returns,
                  verdict, criteria):
    train_sl, val_sl, test_sl = fold['slices']
    m_ls, m_lo = fold['metrics_ls'], fold['metrics_lo']
    crit_a, crit_b, crit_c = criteria

    lines = []
    add = lines.append
    add("# MTDC Strategy Backtest Results")
    add("")
    add(f"Generated by `python run_backtest.py --seed {args.seed}` "
        f"(deterministic given the seed).")
    add("")
    add("## Setup")
    add("")
    add(f"- Data: ETH/USDT 15-minute bars, {dates[0]} to {dates[-1]} ({len(dates)} bars)")
    add(f"- Split: train [{dates[train_sl][0].date()} - {dates[train_sl][-1].date()}], "
        f"validation [{dates[val_sl][0].date()} - {dates[val_sl][-1].date()}], "
        f"**test [{dates[test_sl][0].date()} - {dates[test_sl][-1].date()}]** "
        f"(test touched exactly once)")
    add(f"- Headline transaction cost: {args.cost:.3%} per side (Binance spot taker); "
        f"execution: decide on bar close, fill at next bar open")
    thresholds_str = ", ".join(f"{m['threshold']:.1%}" for m in fold['models'])
    add(f"- Thresholds: {thresholds_str}; "
        f"GP pop {args.gp_pop} x {args.gp_gens} gens; GA pop {args.ga_pop} x {args.ga_gens} gens; seed {args.seed}")
    add("")
    add("### Methodological repairs vs the design document")
    add("")
    add("1. **Look-ahead bias removed**: DC events act at their *confirmation* bar "
        "(when the θ move is observable), not at the extremum bar the document's "
        "pseudocode uses. See the appendix for how much fake profit the naive "
        "version manufactures.")
    add("2. **Target leakage removed**: the original regression predicted OS length "
        "from a 'DC length' that *contained* the OS period itself. The honest "
        "feature is confirmation-minus-extremum; the honest target is "
        "next-extremum-minus-confirmation.")
    add("3. **No scaler leakage**: all statistics (GP fit, average OS, signal "
        "levels) come from training data only.")
    add("4. The document's Nemenyi test is replaced by a moving-block bootstrap CI "
        "and a Wilcoxon signed-rank test on daily returns (two-strategy comparison).")
    add("")
    add("## GP overshoot predictors (fit on train only)")
    add("")
    add("| Threshold | Train events | Expression (log space) | Train MSE | Mean-baseline MSE | Beats mean? |")
    add("|---|---|---|---|---|---|")
    for m in fold['models']:
        gi = m['gp_info']
        add(f"| {m['threshold']:.1%} | {m['n_train_events']} | `{m['expr']}` | "
            f"{gi['train_mse_log']:.3f} | {gi['baseline_mse_log']:.3f} | "
            f"{'yes' if gi['beats_mean_baseline'] else 'no'} |")
    add("")
    add("![GP fits](results/gp_fits.png)")
    add("")
    add("## GA champion")
    add("")
    weights_str = ", ".join(f"{m['threshold']:.1%}: {w:.2f}"
                            for m, w in zip(fold['models'], fold['weights']))
    add(f"- Weights: {weights_str}")
    add(f"- Stop-loss {fold['stop_loss']:.1%}, take-profit {fold['take_profit']:.1%}")
    add(f"- Fitness (Sharpe - 2 x maxDD): train {fold['train_fitness']:.2f}, "
        f"validation {fold['val_fitness']:.2f}, test {fold['test_fitness']:.2f}")
    add("")
    add("![Weights](results/weights.png)")
    add("")
    add("## Test window results (headline, 0.1% cost per side)")
    add("")
    add(TABLE_HEADER)
    add(metrics_row("**MTDC (long/short)**", m_ls))
    add(metrics_row("**MTDC (long-only)**", m_lo))
    for name, (_, mb) in bench.items():
        add(metrics_row(name, mb))
    add("")
    add("![Equity](results/equity_test.png)")
    add("")
    add("![Drawdown](results/drawdown_test.png)")
    add("")
    add("## Statistical significance (test window)")
    add("")
    add(f"- Annualized Sharpe (MTDC L/S): {m_ls['sharpe']:.2f}, "
        f"95% block-bootstrap CI [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")
    add(f"- Mean bar-return difference vs Buy & Hold: 95% CI "
        f"[{diff_ci[0]:.2e}, {diff_ci[1]:.2e}] per 15-min bar")
    add(f"- Wilcoxon signed-rank (daily returns vs Buy & Hold): p = {wilcoxon_p:.3f}")
    add(f"- Random-timing null (200 rotations of the action series): "
        f"strategy return beat {100 * (1 - null_p):.0f}% of null runs "
        f"(one-sided p = {null_p:.3f}; null median {np.median(null_returns):.1%})")
    add("")
    add("## Cost sensitivity (test window)")
    add("")
    add("| Cost per side | MTDC L/S return | MTDC long-only return |")
    add("|---|---|---|")
    for c, r_ls, r_lo in cost_rows:
        add(f"| {c:.3%} | {fmt_pct(r_ls)} | {fmt_pct(r_lo)} |")
    add("")
    add("![Cost sensitivity](results/cost_sensitivity.png)")
    add("")
    add("## Walk-forward robustness (each fold fully refit on prior data only)")
    add("")
    add("| Fold period | MTDC L/S return | Sharpe | Max DD | Trades | Buy & Hold return |")
    add("|---|---|---|---|---|---|")
    for r in wf_rows:
        add(f"| {r['period']} | {fmt_pct(r['mtdc_return'])} | {r['mtdc_sharpe']:.2f} | "
            f"{r['mtdc_dd'] * 100:.1f}% | {r['n_trades']} | {fmt_pct(r['bh_return'])} |")
    add("")
    add("![Walk-forward](results/walk_forward.png)")
    add("")
    add("## Verdict")
    add("")
    add("Pre-registered criteria — the system is called profitable only if, on the "
        "untouched test window at 0.1% per-side costs:")
    add("")
    add(f"- (a) net return > 0: **{'PASS' if crit_a else 'FAIL'}** "
        f"({fmt_pct(m_ls['cumulative_return'])})")
    add(f"- (b) Sharpe > 0 with 95% bootstrap CI excluding 0: "
        f"**{'PASS' if crit_b else 'FAIL'}** "
        f"(Sharpe {m_ls['sharpe']:.2f}, CI [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}])")
    add(f"- (c) positive in at least 2 of 3 walk-forward folds: "
        f"**{'PASS' if crit_c else 'FAIL'}** "
        f"({sum(1 for r in wf_rows if r['mtdc_return'] > 0)}/3 positive)")
    add("")
    add(f"### {verdict}")
    add("")
    add("Beating buy-and-hold is reported separately above: failing to beat a "
        "bull-market benchmark does not by itself make a strategy unprofitable, "
        "and beating it in one window does not make it a money machine. This is a "
        "single asset over a single ~21-month regime; treat any conclusion as "
        "conditional on that.")
    add("")
    add("## Appendix: what the look-ahead bug would have reported")
    add("")
    add("Building the signals at the extremum bar (as the design document's "
        "pseudocode does) instead of the confirmation bar, with everything else "
        "identical:")
    add("")
    add(f"- Naive look-ahead test return: **{fmt_pct(m_naive['cumulative_return'])}** "
        f"(Sharpe {m_naive['sharpe']:.2f})")
    add(f"- Honest test return: {fmt_pct(m_ls['cumulative_return'])} "
        f"(Sharpe {m_ls['sharpe']:.2f})")
    add("")
    add("The difference is manufactured by trading on information that does not "
        "exist yet at decision time.")
    add("")

    with open("RESULTS.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
