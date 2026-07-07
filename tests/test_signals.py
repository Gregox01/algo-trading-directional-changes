import numpy as np

from directional_changes import identify_dc_events
from signals import (build_confidence_matrices, build_signal_series,
                     compute_event_features, signal_strengths)


def constant_predictor(value):
    return lambda dc_lengths: np.full(len(np.atleast_1d(dc_lengths)), float(value))


def test_signal_strengths_thresholds():
    # Comparison happens in log1p space around avg_os_log = log1p(10)
    strengths = signal_strengths([25.0, 12.0, 8.0],
                                 avg_os_log=np.log1p(10.0), sig_level_log=0.5)
    assert list(strengths) == [2, 1, 0]


def test_expiry_and_override():
    events = [
        ('upturn', 10, 100.0, 15, 103.0),
        ('downturn', 40, 110.0, 50, 106.0),
    ]

    def predict(dc_lengths):
        return np.array([20.0, 100.0])

    sig = build_signal_series(events, predict, 100,
                              avg_os_log=np.log1p(10.0), sig_level_log=0.5)
    assert np.all(sig[:15] == 0)
    assert np.all(sig[15:35] == 2)          # predicted horizon 20 bars
    assert np.all(sig[35:50] == 0)          # expired before next confirmation
    assert np.all(sig[50:] == -2)           # capped by end of series, not horizon


def test_no_lookahead_when_future_changes():
    rng = np.random.default_rng(3)
    steps = rng.normal(0, 0.01, 600)
    prices_a = 100.0 * np.exp(np.cumsum(steps))
    steps_b = steps.copy()
    steps_b[300:] = rng.normal(0, 0.02, 300)  # different future
    prices_b = 100.0 * np.exp(np.cumsum(steps_b))

    predict = constant_predictor(20.0)
    events_a = identify_dc_events(prices_a, 0.02)
    events_b = identify_dc_events(prices_b, 0.02)
    sig_a = build_signal_series(events_a, predict, 600,
                                avg_os_log=np.log1p(10.0), sig_level_log=0.5)
    sig_b = build_signal_series(events_b, predict, 600,
                                avg_os_log=np.log1p(10.0), sig_level_log=0.5)
    assert np.array_equal(sig_a[:300], sig_b[:300])


def test_event_features_are_causal_quantities():
    prices = np.array([100.0, 103.0, 100.0, 104.0, 101.0])
    events = identify_dc_events(prices, 0.02)
    feats = compute_event_features(events)
    assert np.all(feats['dc_len'] > 0)
    # os_len for event k measures next extremum minus this confirmation
    assert feats['os_len'][0] == max(0, events[1][1] - events[0][3])
    assert np.isnan(feats['os_len'][-1])


def test_confidence_matrices():
    sig1 = np.array([2, 0, -1], dtype=np.int8)
    sig2 = np.array([0, 1, 0], dtype=np.int8)
    B, S, H = build_confidence_matrices([sig1, sig2])
    assert B.tolist() == [[2, 0, 0], [0, 1, 0]]
    assert S.tolist() == [[0, 0, 1], [0, 0, 0]]
    assert H.tolist() == [[0, 1, 0], [1, 0, 1]]
