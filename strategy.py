"""Weighted multi-threshold voting and the trading simulator.

Execution model: an action is decided on bar t's close (using only information
available then) and filled at bar t+1's open. Transaction costs are charged per
side on traded notional. Short accounting tracks cash plus signed units and
marks to market every bar (the design document's pseudocode double-counted short
proceeds; this is written from scratch).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def weighted_voting(B, S, H, weights):
    """
    Aggregate per-threshold confidences into bar-level actions.

    B, S, H: (n_thresholds, n_bars) buy/sell/hold confidence matrices
    weights:  (n_thresholds,) non-negative threshold weights

    Returns int8 actions per bar: +1 buy, -1 sell, 0 hold. Ties go to hold.
    """
    weights = np.asarray(weights, dtype=np.float64)
    buy_score = weights @ B
    sell_score = weights @ S
    hold_score = weights @ H

    actions = np.zeros(B.shape[1], dtype=np.int8)
    actions[(buy_score > sell_score) & (buy_score > hold_score)] = 1
    actions[(sell_score > buy_score) & (sell_score > hold_score)] = -1
    return actions


@dataclass
class Trade:
    direction: int          # +1 long, -1 short
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    pnl: float              # cash difference over the trade, costs included
    ret: float              # pnl relative to equity at entry


@dataclass
class SimResult:
    equity: np.ndarray          # mark-to-market equity at each bar close
    positions: np.ndarray       # int8 position held during each bar (+1/0/-1)
    trades: List[Trade] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)


def simulate_trading(actions, open_prices, close_prices, initial_capital=10_000.0,
                     transaction_cost=0.001, stop_loss_pct: Optional[float] = None,
                     take_profit_pct: Optional[float] = None, allow_short=True) -> SimResult:
    """
    Simulate trading a single asset with all-in position sizing.

    actions[t] is the decision made on bar t's close: +1 want long, -1 want short
    (flat if allow_short is False), 0 keep the current position. Position changes
    fill at bar t+1's open. Stop-loss/take-profit are evaluated against bar
    closes relative to the entry fill and exit at the next open; after a risk
    exit the strategy stays flat until a bar whose action differs from the
    position just closed (no instant re-entry on a stale signal).
    """
    actions = np.asarray(actions)
    n = len(actions)
    assert len(open_prices) == n and len(close_prices) == n

    cash = float(initial_capital)
    units = 0.0                 # signed asset units
    position = 0                # +1, 0, -1
    entry_price = 0.0
    entry_bar = -1
    entry_equity = float(initial_capital)
    blocked_direction = 0       # direction we refuse to re-enter until the signal changes

    equity = np.empty(n, dtype=np.float64)
    positions = np.zeros(n, dtype=np.int8)
    trades: List[Trade] = []

    pending_target = None       # position to establish at the next open

    for t in range(n):
        price_open = float(open_prices[t])
        price_close = float(close_prices[t])

        # 1) Execute the pending order at this bar's open
        if pending_target is not None and pending_target != position:
            if position != 0:
                # Close the current position
                if position == 1:
                    cash += units * price_open * (1.0 - transaction_cost)
                else:
                    cash -= (-units) * price_open * (1.0 + transaction_cost)
                exit_equity = cash
                trades.append(Trade(
                    direction=position, entry_bar=entry_bar, exit_bar=t,
                    entry_price=entry_price, exit_price=price_open,
                    pnl=exit_equity - entry_equity,
                    ret=(exit_equity - entry_equity) / entry_equity,
                ))
                units = 0.0
                position = 0
            if pending_target != 0:
                entry_equity = cash
                entry_price = price_open
                entry_bar = t
                if pending_target == 1:
                    units = cash / (price_open * (1.0 + transaction_cost))
                    cash = 0.0
                    position = 1
                else:
                    short_units = cash / price_open
                    cash += short_units * price_open * (1.0 - transaction_cost)
                    units = -short_units
                    position = -1
        pending_target = None

        positions[t] = position
        equity[t] = cash + units * price_close

        # 2) Decide what to do at the next open
        action = int(actions[t])
        if action != blocked_direction:
            blocked_direction = 0

        # Risk management first: stop-loss / take-profit on the close
        risk_exit = False
        if position != 0:
            if position == 1:
                pl_pct = (price_close - entry_price) / entry_price
            else:
                pl_pct = (entry_price - price_close) / entry_price
            if stop_loss_pct is not None and pl_pct <= -stop_loss_pct:
                risk_exit = True
            elif take_profit_pct is not None and pl_pct >= take_profit_pct:
                risk_exit = True

        if risk_exit:
            pending_target = 0
            blocked_direction = position
        elif action == 1 and blocked_direction != 1:
            pending_target = 1 if position != 1 else None
        elif action == -1 and blocked_direction != -1:
            desired = -1 if allow_short else 0
            pending_target = desired if position != desired else None
        # action == 0 (or blocked): keep the current position

    # Close any open position at the final close for reporting purposes
    if position != 0:
        if position == 1:
            final_cash = cash + units * float(close_prices[-1]) * (1.0 - transaction_cost)
        else:
            final_cash = cash - (-units) * float(close_prices[-1]) * (1.0 + transaction_cost)
        trades.append(Trade(
            direction=position, entry_bar=entry_bar, exit_bar=n - 1,
            entry_price=entry_price, exit_price=float(close_prices[-1]),
            pnl=final_cash - entry_equity,
            ret=(final_cash - entry_equity) / entry_equity,
        ))

    return SimResult(equity=equity, positions=positions, trades=trades)
