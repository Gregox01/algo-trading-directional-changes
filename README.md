# Multi-Threshold Directional Changes (MTDC) Trading System

An algorithmic trading system for ETH/USDT built from the design in
[`predictor_algo_documentation.md`](predictor_algo_documentation.md):

1. Detect **Directional Change (DC)** events at multiple price thresholds
   (0.5%–3%).
2. Predict each event's **Overshoot (OS)** length — how much longer the trend
   runs — with symbolic regression (DEAP genetic programming).
3. Turn predictions into per-threshold buy/sell/hold signals, aggregate them
   with **weighted voting**, and optimize the weights plus stop-loss/take-profit
   with a **genetic algorithm**.
4. Backtest with realistic execution (decide on bar close, fill at next open,
   per-side transaction costs) against buy-and-hold, single-threshold DC and
   RSI benchmarks.

## Verdict: not profitable

The full, honest backtest (chronological train/validation/test split, all
fitting on training data only, 0.1% per-side costs) says the system **loses
money out of sample**: −15.6% on the untouched May–Oct 2024 test window
(buy-and-hold: −37.0%), negative in 2 of 3 walk-forward folds, and still
negative even with zero transaction costs. Full numbers, charts, significance
tests and the pre-registered criteria are in [`RESULTS.md`](RESULTS.md).

The most important number in the report is in the appendix: implementing the
design document's pseudocode literally — acting at the DC *extremum* instead of
the *confirmation* bar — reports **+1,168%** on the same test window. That gap
is look-ahead bias, not alpha. The original half-finished implementation also
had target leakage in its regression dataset (the feature contained the value
being predicted). Both defects are documented and fixed in this codebase; the
honest result is the negative one above.

## Layout

| File | Purpose |
|---|---|
| `directional_changes.py` | DC/OS event detection (confirmation-aware), data loading |
| `symbolic_regression.py` | GP predictor of OS length from DC length (leak-free, vectorized) |
| `signals.py` | Event-space → bar-space signal series per threshold |
| `strategy.py` | Weighted voting + trading simulator (costs, SL/TP, long & short) |
| `ga_optimizer.py` | GA over threshold weights and risk parameters |
| `benchmarks.py` | Buy & hold, single-threshold DC, RSI, random-timing null |
| `metrics.py` | Sharpe/Sortino/drawdown/profit factor, bootstrap CIs, Wilcoxon |
| `run_backtest.py` | End-to-end seeded pipeline; writes `RESULTS.md` + `results/` |
| `tests/` | pytest suite (causality, no-look-ahead, exact simulator arithmetic) |

## Usage

```bash
pip install -r requirements.txt

python run_backtest.py            # full run, ~1-2 minutes, deterministic (seed 42)
python run_backtest.py --quick    # fast plumbing check
python run_backtest.py --help     # thresholds, costs, GP/GA budgets, seed

pytest                            # run the test suite
```

Data: `ETHUSDT_15m.csv` — 59,703 fifteen-minute ETH/USDT bars,
2023-01-01 to 2024-10-02.

## Caveats

Single asset, single ~21-month period, one seed family: even the negative
verdict is conditional on this regime. Shorting is modeled at spot prices
without borrow/funding costs (the long-only variant is also reported and is
likewise unprofitable).
