import numpy as np
import pytest

from benchmarks import buy_and_hold
from strategy import simulate_trading, weighted_voting


def test_all_hold_stays_flat():
    n = 50
    actions = np.zeros(n, dtype=np.int8)
    opens = np.linspace(100, 200, n)
    closes = np.linspace(100, 200, n)
    sim = simulate_trading(actions, opens, closes, initial_capital=10_000.0)
    assert np.allclose(sim.equity, 10_000.0)
    assert sim.n_trades == 0


def test_long_trade_exact_arithmetic_zero_cost():
    actions = np.array([1, 0, 0, 0, 0], dtype=np.int8)
    opens = np.array([100.0, 100.0, 120.0, 120.0, 80.0])
    closes = np.array([100.0, 110.0, 120.0, 90.0, 80.0])
    sim = simulate_trading(actions, opens, closes, initial_capital=10_000.0,
                           transaction_cost=0.0)
    # Decided on bar 0, filled at open of bar 1 -> 100 units
    assert sim.positions.tolist() == [0, 1, 1, 1, 1]
    assert np.allclose(sim.equity, [10_000, 11_000, 12_000, 9_000, 8_000])
    assert sim.n_trades == 1
    assert sim.trades[0].pnl == pytest.approx(-2_000.0)


def test_long_trade_with_costs():
    actions = np.array([1, 0, 0], dtype=np.int8)
    opens = np.array([100.0, 100.0, 100.0])
    closes = np.array([100.0, 110.0, 110.0])
    cost = 0.001
    sim = simulate_trading(actions, opens, closes, initial_capital=10_000.0,
                           transaction_cost=cost)
    units = 10_000.0 / (100.0 * (1 + cost))
    assert sim.equity[1] == pytest.approx(units * 110.0)
    # Final mark-to-close trade pays the exit cost too
    assert sim.trades[-1].pnl == pytest.approx(units * 110.0 * (1 - cost) - 10_000.0)


def test_short_trade_exact_arithmetic_zero_cost():
    actions = np.array([-1, 0, 0, 0], dtype=np.int8)
    opens = np.array([100.0, 100.0, 100.0, 100.0])
    closes = np.array([100.0, 90.0, 95.0, 90.0])
    sim = simulate_trading(actions, opens, closes, initial_capital=10_000.0,
                           transaction_cost=0.0)
    # Short 100 units at 100: cash 20k, equity = 20k - 100 * close
    assert np.allclose(sim.equity, [10_000, 11_000, 10_500, 11_000])
    assert sim.trades[-1].pnl == pytest.approx(1_000.0)


def test_short_entry_charges_fee_immediately():
    actions = np.array([-1, 0], dtype=np.int8)
    opens = np.array([100.0, 100.0])
    closes = np.array([100.0, 100.0])
    cost = 0.001
    sim = simulate_trading(actions, opens, closes, initial_capital=10_000.0,
                           transaction_cost=cost)
    # Flat price: equity should be down exactly the entry fee on shorted notional
    assert sim.equity[1] == pytest.approx(10_000.0 - 10_000.0 * cost)


def test_allow_short_false_maps_sell_to_flat():
    actions = np.array([1, 0, -1, 0], dtype=np.int8)
    opens = np.array([100.0, 100.0, 100.0, 100.0])
    closes = np.array([100.0, 100.0, 100.0, 100.0])
    sim = simulate_trading(actions, opens, closes, transaction_cost=0.0,
                           allow_short=False)
    assert sim.positions.tolist() == [0, 1, 1, 0]


def test_higher_cost_lower_equity():
    rng = np.random.default_rng(5)
    n = 500
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    actions = rng.choice(np.array([1, 0, -1], dtype=np.int8), size=n)
    eq_cheap = simulate_trading(actions, opens, closes, transaction_cost=0.0).equity[-1]
    eq_costly = simulate_trading(actions, opens, closes, transaction_cost=0.002).equity[-1]
    assert eq_costly < eq_cheap


def test_stop_loss_exits_at_next_open_and_blocks_reentry():
    actions = np.ones(5, dtype=np.int8)  # signal stays long the whole time
    opens = np.array([100.0, 100.0, 100.0, 94.0, 94.0])
    closes = np.array([100.0, 100.0, 93.0, 94.0, 94.0])
    sim = simulate_trading(actions, opens, closes, initial_capital=10_000.0,
                           transaction_cost=0.0, stop_loss_pct=0.05)
    # -7% on bar 2 close triggers the stop; exit fills at bar 3 open (94)
    assert sim.positions.tolist() == [0, 1, 1, 0, 0]
    assert sim.equity[-1] == pytest.approx(9_400.0)
    # The stale long signal must not re-enter
    assert sim.trades[0].exit_bar == 3


def test_take_profit_exits():
    actions = np.array([1, 0, 0, 0], dtype=np.int8)
    opens = np.array([100.0, 100.0, 100.0, 111.0])
    closes = np.array([100.0, 100.0, 111.0, 111.0])
    sim = simulate_trading(actions, opens, closes, transaction_cost=0.0,
                           take_profit_pct=0.10)
    assert sim.positions.tolist() == [0, 1, 1, 0]


def test_buy_and_hold_matches_closed_form():
    rng = np.random.default_rng(9)
    n = 300
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    cost = 0.001
    sim = buy_and_hold(opens, closes, transaction_cost=cost)
    units = 10_000.0 / (opens[1] * (1 + cost))
    assert sim.equity[-1] == pytest.approx(units * closes[-1])


def test_weighted_voting_ties_go_to_hold():
    B = np.array([[1.0], [0.0]])
    S = np.array([[0.0], [1.0]])
    H = np.array([[0.0], [0.0]])
    actions = weighted_voting(B, S, H, np.array([0.5, 0.5]))
    assert actions.tolist() == [0]


def test_weighted_voting_majority():
    # threshold 0 says strong buy, threshold 1 says hold
    B = np.array([[2.0], [0.0]])
    S = np.array([[0.0], [0.0]])
    H = np.array([[0.0], [1.0]])
    assert weighted_voting(B, S, H, np.array([0.6, 0.4])).tolist() == [1]
    assert weighted_voting(B, S, H, np.array([0.2, 0.8])).tolist() == [0]
