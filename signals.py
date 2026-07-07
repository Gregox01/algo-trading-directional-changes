"""Turn per-threshold DC events into bar-indexed trading signal series.

The design document's weighted_voting assumes the per-threshold signal lists are
aligned element by element, which is false: each threshold confirms events at
different bars. The fix is to project every threshold's events onto the common
bar axis. sig[t] is the signal from the latest event confirmed at or before bar t
(so it only uses information available at bar t), and a signal expires once the
predicted OS horizon has elapsed.

Signal values (int8): +2 strong buy, +1 buy, 0 hold, -1 sell, -2 strong sell.
"""

import numpy as np


def compute_event_features(dc_events):
    """
    Per-event features from confirmation-aware DC events (5-tuples).

    Returns a dict of aligned arrays; os_len (bars from confirmation to the next
    extremum, clipped at 0) is NaN for the final event, whose next extremum is
    unknown.
    """
    n = len(dc_events)
    direction = np.array([1 if e[0] == 'upturn' else -1 for e in dc_events], dtype=np.int8)
    ext_idx = np.array([e[1] for e in dc_events], dtype=np.int64)
    conf_idx = np.array([e[3] for e in dc_events], dtype=np.int64)
    dc_len = conf_idx - ext_idx
    os_len = np.full(n, np.nan)
    if n > 1:
        os_len[:-1] = np.maximum(0, ext_idx[1:] - conf_idx[:-1])
    return {
        'direction': direction,
        'ext_idx': ext_idx,
        'conf_idx': conf_idx,
        'dc_len': dc_len,
        'os_len': os_len,
    }


def signal_strengths(predicted_os, avg_os_log, sig_level_log):
    """
    Map predicted OS lengths to strengths per the design doc: 2 (strong) if the
    prediction exceeds the average OS length by more than one significance level,
    1 if it is merely above average, 0 (hold) otherwise.

    The comparison happens in log1p space: OS lengths are heavily right-skewed
    (train median ~9 bars vs mean ~22), so a predictor fit in log space almost
    never exceeds the raw arithmetic mean and the raw-space rule would emit no
    signals at all. avg_os_log and sig_level_log are the mean and std of
    log1p(os_length) over training events only.
    """
    predicted_os = np.asarray(predicted_os, dtype=float)
    deviation = np.log1p(np.maximum(predicted_os, 0.0)) - avg_os_log
    return np.where(deviation > sig_level_log, 2, np.where(deviation > 0, 1, 0)).astype(np.int8)


def build_signal_series(dc_events, predict_fn, n_bars, avg_os_log, sig_level_log):
    """
    Build a causal int8 signal series over bars for one threshold.

    Event k confirmed at bar c_k emits sign(direction) * strength from bar c_k
    until the earlier of the next confirmation c_{k+1} or the predicted horizon
    c_k + max(1, round(predicted_os)). sig[t] therefore depends only on events
    with confirmation index <= t. avg_os_log/sig_level_log are log1p-space train
    statistics (see signal_strengths).
    """
    sig = np.zeros(n_bars, dtype=np.int8)
    if not dc_events:
        return sig

    feats = compute_event_features(dc_events)
    predicted = np.atleast_1d(np.asarray(predict_fn(feats['dc_len']), dtype=float))
    strengths = signal_strengths(predicted, avg_os_log, sig_level_log)

    conf_idx = feats['conf_idx']
    direction = feats['direction']
    n_events = len(dc_events)

    for k in range(n_events):
        s = int(strengths[k]) * int(direction[k])
        if s == 0:
            continue
        start = int(conf_idx[k])
        if start >= n_bars:
            break
        horizon = int(max(1, round(predicted[k])))
        end = min(start + horizon, n_bars)
        if k + 1 < n_events:
            end = min(end, int(conf_idx[k + 1]))
        if end > start:
            sig[start:end] = s
    return sig


def build_confidence_matrices(signal_series_list):
    """
    Stack per-threshold signal series into the three confidence matrices used by
    weighted voting: B[j, t] / S[j, t] are the buy/sell confidences (0, 1 or 2)
    and H[j, t] is the hold confidence (1 where the threshold votes hold).
    """
    sig = np.vstack(signal_series_list).astype(np.float64)
    B = np.maximum(sig, 0.0)
    S = np.maximum(-sig, 0.0)
    H = (sig == 0).astype(np.float64)
    return B, S, H
