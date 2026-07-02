"""Kill-gate study: can any model predict which DC events are worth trading?

Pre-registered gate (ASSESSMENT.md §4): if out-of-fold AUC <= 0.52 on the
training region for the cost-aware label, across thresholds and model classes,
the DC-overshoot hypothesis is falsified at that asset/frequency and no
downstream machinery (voting, GA, filters) can rescue it.

Protocol: events from the first `train_frac` of bars only (test region never
touched); 5-fold TimeSeriesSplit over events in confirmation order; models are
logistic regression (scaled) and gradient boosting; a label-shuffle control
gives the noise band. Also reports the fully tradeable outcome (enter at
confirmation, exit at next confirmation) conditioned on predicted probability.

Usage:
    python killgate_study.py                       # ETH 15m, perp costs
    python killgate_study.py --data X.csv.gz --bars-per-day 24 --cost-rt 0.0012
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_event_dataset, label_exceeds_cost, FEATURE_COLS


def load_prices(path):
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').drop_duplicates('Datetime').set_index('Datetime')
    df = df[df['Close'] > 0]
    return df


def oof_auc(model_factory, X, y, n_splits=5):
    """Out-of-fold AUC over chronological folds; returns (auc, per_fold, oof_proba)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    proba = np.full(len(y), np.nan)
    fold_aucs = []
    for tr, te in tscv.split(X):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = model_factory()
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        proba[te] = p
        fold_aucs.append(roc_auc_score(y[te], p))
    mask = ~np.isnan(proba)
    overall = roc_auc_score(y[mask], proba[mask]) if mask.sum() and len(np.unique(y[mask])) > 1 else np.nan
    return overall, fold_aucs, proba


def shuffle_auc_band(model_factory, X, y, n_shuffles=20, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_shuffles):
        y_shuf = rng.permutation(y)
        a, _, _ = oof_auc(model_factory, X, y_shuf)
        if not np.isnan(a):
            aucs.append(a)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (np.nan, np.nan)


def run_study(price_df, thresholds, cost_rt, train_frac=0.6, seed=42, label="",
              vol_window=96, rate_window=None):
    closes = price_df['Close'].values.astype(np.float64)
    dates = price_df.index
    n = len(closes)
    train_end = int(n * train_frac)
    rate_window = rate_window or 5 * vol_window

    models = {
        'logistic': lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)),
        'gboost': lambda: HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, random_state=seed),
    }

    results = []
    for ti, theta in enumerate(thresholds):
        df = build_event_dataset(closes, thresholds, ti, dates=dates,
                                 vol_window=vol_window, rate_window=rate_window)
        if df.empty:
            continue
        df = df[df['conf_idx'] < train_end].reset_index(drop=True)
        if len(df) < 120:
            print(f"[{label}] theta={theta:.2%}: only {len(df)} train events - skipped")
            continue
        y = label_exceeds_cost(df, cost_rt).values
        X = df[FEATURE_COLS].values.astype(np.float64)
        base_rate = float(y.mean())

        row = {'label': label, 'theta': theta, 'n_events': int(len(df)),
               'base_rate': base_rate, 'cost_rt': cost_rt,
               'mean_os_move': float(df['os_move'].mean()),
               'mean_ret_next_conf': float(df['ret_to_next_conf'].mean())}

        for name, factory in models.items():
            auc, folds, proba = oof_auc(factory, X, y)
            row[f'auc_{name}'] = float(auc) if not np.isnan(auc) else None
            # Economic check: tradeable outcome in the top-confidence tercile
            mask = ~np.isnan(proba)
            if mask.sum() > 30:
                dfx = df.loc[mask].copy()
                dfx['p'] = proba[mask]
                top = dfx[dfx['p'] >= dfx['p'].quantile(2 / 3)]
                row[f'top_tercile_net_{name}'] = float(
                    (top['ret_to_next_conf'] - cost_rt).mean())
        lo, hi = shuffle_auc_band(models['logistic'], X, y, seed=seed)
        row['shuffle_auc_95band'] = [lo, hi]
        results.append(row)
        print(f"[{label}] theta={theta:.2%}: n={len(df)}, base={base_rate:.2f}, "
              f"AUC logit={row.get('auc_logistic')}, gboost={row.get('auc_gboost')}, "
              f"noise band=({lo:.3f},{hi:.3f})")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='ETHUSDT_15m.csv')
    p.add_argument('--thresholds', type=float, nargs='+',
                   default=[0.01, 0.015, 0.02, 0.03])
    p.add_argument('--cost-rt', type=float, default=0.0012,
                   help='round-trip cost incl. slippage (default: perp taker 2x0.05% + 0.02%)')
    p.add_argument('--train-frac', type=float, default=0.6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--label', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    label = args.label or args.data.split('/')[-1].split('.')[0]
    prices = load_prices(args.data)
    print(f"{label}: {len(prices)} bars, {prices.index[0]} to {prices.index[-1]}")
    res = run_study(prices, args.thresholds, args.cost_rt,
                    train_frac=args.train_frac, seed=args.seed, label=label)

    def cell_passes(r):
        floor = max(0.52, r['shuffle_auc_95band'][1])
        return any((r.get(f'auc_{m}') or 0) > floor for m in ('logistic', 'gboost'))

    gate_pass = any(cell_passes(r) for r in res)
    print(f"\nKILL-GATE ({label}): "
          f"{'SURVIVES (AUC beats 0.52 AND the shuffle noise band)' if gate_pass else 'FAILED (no AUC beats both 0.52 and its shuffle noise band)'}")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)
