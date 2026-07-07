"""Causal per-event feature engineering for the cost-aware classification target.

For each DC event confirmed at bar c, every feature uses only information
available at c (trailing windows, previously confirmed events, and the just-
revealed extremum of the finished trend). The label asks the economically
relevant question: will the overshoot price move from the confirmation point
exceed a round-trip cost before the trend ends?

Label subtlety: the "trend end" price is the next extremum, which is only
*revealed* at the next confirmation — so the label is generous (it assumes a
perfect exit at the extremum). That is deliberate: the classifier study is a
KILL-GATE, testing a necessary condition. If even this generous label cannot be
predicted better than chance (AUC <= 0.52), no implementable exit rule can
monetize DC overshoots at this frequency. A secondary, fully tradeable outcome
(enter at confirmation, exit at next confirmation) is also recorded.
"""

import numpy as np
import pandas as pd

from directional_changes import identify_dc_events


def build_event_dataset(closes, thresholds, trade_idx, dates=None,
                        vol_window=96, rate_window=500, htf_multiple=4.0):
    """
    Build the per-event feature/label table for one trading threshold.

    closes:      full close-price array
    thresholds:  the full threshold list (used for cross-threshold features)
    trade_idx:   index into `thresholds` of the threshold whose events we trade
    dates:       optional DatetimeIndex aligned with closes (for time features)

    Returns a DataFrame with one row per event (excluding the last, whose
    outcome is unknown), sorted by confirmation index.
    """
    closes = np.asarray(closes, dtype=np.float64)
    theta = thresholds[trade_idx]
    events = identify_dc_events(closes, theta)
    if len(events) < 3:
        return pd.DataFrame()

    # Cross-threshold confirmation indices (for event-rate and HTF features)
    other_confs = {}
    for j, th in enumerate(thresholds):
        if j == trade_idx:
            other_confs[j] = None
            continue
        evs = identify_dc_events(closes, th)
        other_confs[j] = np.array([e[3] for e in evs], dtype=np.int64)

    htf_theta = theta * htf_multiple
    htf_events = identify_dc_events(closes, htf_theta)
    htf_confs = np.array([e[3] for e in htf_events], dtype=np.int64)
    htf_dirs = np.array([1 if e[0] == 'upturn' else -1 for e in htf_events], dtype=np.int8)

    log_close = np.log(closes)
    log_ret = np.diff(log_close, prepend=log_close[0])

    own_confs = np.array([e[3] for e in events], dtype=np.int64)
    cross_conf_arrays = [c for c in other_confs.values() if c is not None]

    rows = []
    for k in range(len(events) - 1):
        etype, ext_idx, ext_price, c, conf_price = events[k]
        direction = 1 if etype == 'upturn' else -1
        next_ext_idx = events[k + 1][1]
        next_ext_price = events[k + 1][2]
        next_conf_idx = events[k + 1][3]
        next_conf_price = events[k + 1][4]

        # --- outcomes (future; used as labels only) ---
        os_move = direction * (next_ext_price - conf_price) / conf_price
        ret_to_next_conf = direction * (next_conf_price - conf_price) / conf_price
        os_len = max(0, next_ext_idx - c)

        # --- causal features at bar c ---
        dc_len = c - ext_idx
        dc_move = abs(conf_price / ext_price - 1.0)
        overshoot_at_conf = dc_move - theta

        lo = max(1, c - vol_window)
        vol = float(np.std(log_ret[lo:c + 1]))
        trend = float(log_close[c] - log_close[max(0, c - vol_window)])

        rate_lo = c - rate_window
        own_rate = int(k + 1 - np.searchsorted(own_confs, rate_lo, side='right'))
        cross_rate = 0
        for confs in cross_conf_arrays:
            cross_rate += int(np.searchsorted(confs, c, side='right')
                              - np.searchsorted(confs, rate_lo, side='right'))

        # Previous event's finished overshoot (revealed by this confirmation)
        if k > 0:
            prev_conf = events[k - 1][3]
            prev_os_len = max(0, ext_idx - prev_conf)
            prev_os_move = abs(ext_price / events[k - 1][4] - 1.0)
        else:
            prev_os_len = 0
            prev_os_move = 0.0

        # Higher-timeframe state: direction of last HTF event confirmed at or before c
        pos = np.searchsorted(htf_confs, c, side='right') - 1
        htf_state = int(htf_dirs[pos]) if pos >= 0 else 0

        row = {
            'conf_idx': c,
            'direction': direction,
            'dc_len': dc_len,
            'log_dc_len': np.log1p(dc_len),
            'dc_move': dc_move,
            'overshoot_at_conf': overshoot_at_conf,
            'vol': vol,
            'trend': trend,
            'dc_move_over_vol': dc_move / vol if vol > 0 else 0.0,
            'own_event_rate': own_rate,
            'cross_event_rate': cross_rate,
            'prev_os_len': prev_os_len,
            'log_prev_os_len': np.log1p(prev_os_len),
            'prev_os_move': prev_os_move,
            'htf_state': htf_state,
            'htf_aligned': htf_state * direction,
            # outcomes
            'os_move': os_move,
            'os_len': os_len,
            'ret_to_next_conf': ret_to_next_conf,
        }
        if dates is not None:
            ts = dates[c]
            row['hour'] = ts.hour
            row['dow'] = ts.dayofweek
        rows.append(row)

    return pd.DataFrame(rows)


FEATURE_COLS = ['log_dc_len', 'dc_move', 'overshoot_at_conf', 'vol', 'trend',
                'dc_move_over_vol', 'own_event_rate', 'cross_event_rate',
                'log_prev_os_len', 'prev_os_move', 'htf_aligned', 'hour', 'dow']


def label_exceeds_cost(df, cost_round_trip):
    """1 if the (generous) overshoot move covers the round-trip cost."""
    return (df['os_move'] > cost_round_trip).astype(int)
