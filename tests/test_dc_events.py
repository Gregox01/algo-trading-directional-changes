import numpy as np
import pytest

from directional_changes import identify_dc_events, identify_os_events


def test_hand_computed_zigzag():
    prices = np.array([100.0, 103.0, 100.0, 104.0, 101.0])
    events = identify_dc_events(prices, 0.02)
    assert events == [
        ('upturn', 0, 100.0, 1, 103.0),
        ('downturn', 1, 103.0, 2, 100.0),
        ('upturn', 2, 100.0, 3, 104.0),
        ('downturn', 3, 104.0, 4, 101.0),
    ]


def test_flat_series_produces_no_events():
    prices = np.full(100, 100.0)
    assert identify_dc_events(prices, 0.01) == []


@pytest.mark.parametrize("threshold", [0.005, 0.02])
def test_confirmation_invariants(threshold):
    rng = np.random.default_rng(7)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000)))
    events = identify_dc_events(prices, threshold)
    assert len(events) > 5
    for event_type, ext_idx, ext_price, conf_idx, conf_price in events:
        assert conf_idx > ext_idx
        move = (conf_price - ext_price) / ext_price
        if event_type == 'upturn':
            assert move >= threshold
        else:
            assert move <= -threshold
    # Events are confirmed in order and alternate after the first
    conf_indices = [e[3] for e in events]
    assert conf_indices == sorted(conf_indices)
    types = [e[0] for e in events]
    for a, b in zip(types, types[1:]):
        assert a != b


def test_causality_under_truncation():
    """Truncating the series must not change any already-confirmed event."""
    rng = np.random.default_rng(11)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    full_events = identify_dc_events(prices, 0.01)
    for cut in (500, 1500, 2500):
        truncated_events = identify_dc_events(prices[:cut], 0.01)
        expected = [e for e in full_events if e[3] <= cut - 1]
        assert truncated_events == expected


def test_os_events_span_between_extrema():
    prices = np.array([100.0, 103.0, 100.0, 104.0, 101.0])
    events = identify_dc_events(prices, 0.02)
    os_events = identify_os_events(events, len(prices))
    for start, end in os_events:
        assert start < end
        assert 0 <= start < len(prices)
        assert end <= len(prices) - 1
