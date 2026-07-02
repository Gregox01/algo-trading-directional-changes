"""Multi-asset / multi-timeframe DC edge search with pre-registered rules.

Runs the kill-gate classification study (killgate_study.py) over every dataset
in marketdata/ plus the original ETH 15m CSV, then writes
PROFITABILITY_SEARCH.md with per-cell results and a per-asset-class verdict.

Pre-registered rules (fixed before looking at any new data):
- Thresholds are VOL-ADAPTIVE, not tuned: theta = {8, 12, 16, 24} x sigma_bar,
  where sigma_bar is the std of bar log returns over the training region.
  (On ETH 15m this reproduces approximately the original 1%-3% grid.)
- Round-trip costs by asset class: crypto perp 0.12% (2x0.05% taker + slippage),
  FX 0.05% (the DC literature's 0.025% per action), stocks/ETFs 0.10%.
- Only the first 60% of each series (train region) is used - test regions stay
  untouched for any strategy that survives the gate.
- Gate per cell: out-of-fold AUC must exceed both 0.52 and the upper edge of
  its own label-shuffle noise band. With ~100 cells, a few isolated passes are
  expected by chance (~2-3 at the 2.5% band edge); the pre-registered criterion
  for pursuing an asset class is CONSISTENCY: >= 3 passing cells in that class
  AND a positive top-tercile net tradeable return in the passing cells.

Usage: python multi_asset_study.py [--datadir marketdata] [--out PROFITABILITY_SEARCH.md]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from killgate_study import load_prices, run_study

VOL_MULTIPLES = [8.0, 12.0, 16.0, 24.0]

CLASS_PARAMS = {
    # asset class: (round-trip cost, vol_window by timeframe)
    'crypto': {'cost_rt': 0.0012},
    'fx': {'cost_rt': 0.0005},
    'stock': {'cost_rt': 0.0010},
}
VOL_WINDOWS = {'10m': 144, '15m': 96, '1h': 48, '4h': 30, '1d': 20}
CRYPTO_SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT')
FX_SYMBOLS = ('EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD')


def classify(name):
    sym = name.split('_')[0]
    if sym in CRYPTO_SYMBOLS:
        return 'crypto'
    if sym in FX_SYMBOLS:
        return 'fx'
    return 'stock'


def resample_ohlc(df, rule):
    g = df.resample(rule)
    out = pd.DataFrame({
        'Open': g['Open'].first(), 'High': g['High'].max(),
        'Low': g['Low'].min(), 'Close': g['Close'].last(),
        'Volume': g['Volume'].sum(),
    }).dropna(subset=['Open', 'Close'])
    return out


def collect_datasets(datadir):
    """Yield (label, timeframe, asset_class, dataframe)."""
    files = sorted(glob.glob(os.path.join(datadir, '*.csv.gz')))
    for path in files:
        name = os.path.basename(path).replace('.csv.gz', '')
        if name == 'MANIFEST':
            continue
        sym, tf = name.rsplit('_', 1)
        df = load_prices(path)
        yield name, tf, classify(name), df
        # Derive 4h from 1h for crypto
        if tf == '1h' and classify(name) == 'crypto':
            yield f"{sym}_4h", '4h', 'crypto', resample_ohlc(df, '4h')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datadir', default='marketdata')
    p.add_argument('--out', default='PROFITABILITY_SEARCH.md')
    p.add_argument('--json-out', default='results/profitability_search.json')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-frac', type=float, default=0.6)
    args = p.parse_args()

    all_results = []
    for label, tf, aclass, df in collect_datasets(args.datadir):
        if len(df) < 5000 and tf != '1d':
            print(f"{label}: only {len(df)} bars - skipped")
            continue
        if tf == '1d' and len(df) < 1500:
            print(f"{label}: only {len(df)} bars - skipped")
            continue
        closes = df['Close'].values.astype(np.float64)
        train_end = int(len(closes) * args.train_frac)
        sigma = float(np.std(np.diff(np.log(closes[:train_end]))))
        thresholds = [round(m * sigma, 6) for m in VOL_MULTIPLES]
        cost_rt = CLASS_PARAMS[aclass]['cost_rt']
        vol_window = VOL_WINDOWS.get(tf, 96)
        print(f"\n=== {label} ({aclass}, {len(df)} bars, sigma={sigma:.5f}, "
              f"thresholds={[f'{t:.3%}' for t in thresholds]}, cost_rt={cost_rt:.3%}) ===")
        try:
            res = run_study(df, thresholds, cost_rt, train_frac=args.train_frac,
                            seed=args.seed, label=label, vol_window=vol_window)
        except Exception as e:
            print(f"{label}: FAILED {e}")
            continue
        for r in res:
            r['asset_class'] = aclass
            r['timeframe'] = tf
            r['sigma_bar'] = sigma
        all_results.extend(res)

    os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
    with open(args.json_out, 'w') as f:
        json.dump(all_results, f, indent=2)

    write_report(all_results, args)
    print(f"\nReport written to {args.out}")


def cell_passes(r):
    floor = max(0.52, (r.get('shuffle_auc_95band') or [0, 1])[1])
    return any((r.get(f'auc_{m}') or 0) > floor for m in ('logistic', 'gboost'))


def best_auc(r):
    return max(r.get('auc_logistic') or 0.0, r.get('auc_gboost') or 0.0)


def write_report(results, args):
    lines = []
    add = lines.append
    add("# Multi-Asset / Multi-Timeframe DC Edge Search")
    add("")
    add(f"Generated by `python multi_asset_study.py --seed {args.seed}`. "
        "Pre-registered rules are documented in the script docstring: vol-adaptive "
        "thresholds (8/12/16/24 x bar sigma), class-level costs (crypto perp 0.12% RT, "
        "FX 0.05% RT, stocks 0.10% RT), first 60% of each series only, and a "
        "consistency requirement (>= 3 passing cells per asset class, with positive "
        "top-tercile tradeable returns) before any cell earns a backtest.")
    add("")
    add("The question each cell answers: can a classifier (logistic / gradient "
        "boosting over causal event features) predict which DC events' overshoots "
        "will cover round-trip costs, better than chance (out-of-fold AUC beating "
        "both 0.52 and the cell's own label-shuffle noise band)?")
    add("")
    add("| Dataset | Class | θ | Events | Base rate | AUC (logit) | AUC (gboost) | Noise band | Top-tercile net | PASS |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    n_pass = 0
    by_class = {}
    for r in results:
        band = r.get('shuffle_auc_95band') or [np.nan, np.nan]
        ok = cell_passes(r)
        n_pass += ok
        by_class.setdefault(r['asset_class'], []).append(r)
        top_net = r.get('top_tercile_net_gboost')
        top_net_s = f"{top_net * 100:+.2f}%" if top_net is not None else "-"
        add(f"| {r['label']} | {r['asset_class']} | {r['theta']:.3%} | {r['n_events']} | "
            f"{r['base_rate']:.2f} | {(r.get('auc_logistic') or 0):.3f} | "
            f"{(r.get('auc_gboost') or 0):.3f} | ({band[0]:.2f}, {band[1]:.2f}) | "
            f"{top_net_s} | {'**YES**' if ok else 'no'} |")
    add("")
    add(f"**{n_pass} of {len(results)} cells pass.** Under the null, ~2.5% of cells "
        f"(~{max(1, round(0.025 * len(results)))}) are expected to pass by chance.")
    add("")
    add("## Per-class verdicts (pre-registered: >= 3 passing cells AND positive "
        "top-tercile net in those cells)")
    add("")
    for aclass, rows in sorted(by_class.items()):
        passing = [r for r in rows if cell_passes(r)]
        pos_net = [r for r in passing
                   if (r.get('top_tercile_net_gboost') or r.get('top_tercile_net_logistic') or -1) > 0]
        verdict = "PURSUE (consistent signal)" if len(passing) >= 3 and len(pos_net) >= 3 \
            else "NO EDGE FOUND"
        aucs = [best_auc(r) for r in rows]
        add(f"- **{aclass}** ({len(rows)} cells): {len(passing)} pass, "
            f"{len(pos_net)} with positive net top-tercile; median best-AUC "
            f"{np.median(aucs):.3f} -> **{verdict}**")
    add("")
    add("A cell that 'passes' in isolation inside a 100-cell grid is expected noise; "
        "only class-level consistency counts. Any PURSUE verdict leads to a proper "
        "walk-forward backtest on that class's untouched test regions - profitability "
        "is not claimed from AUC alone.")
    add("")
    with open(args.out, 'w') as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
