import numpy as np
from typing import List, Tuple, Callable

def generate_signals_multi_threshold(
    dc_events_list: List[List[Tuple[str, int, float]]],
    scaler_X_list: List[object],
    scaler_y_list: List[object],
    funcs: List[Callable],
    thresholds: List[float],
    price_series: np.ndarray,
    weights: List[float]
) -> List[Tuple[int, int, float]]:
    signals = []

    # Calculate short and long moving averages
    short_window = 10
    long_window = 50
    short_ma = np.convolve(price_series, np.ones(short_window), 'valid') / short_window
    long_ma = np.convolve(price_series, np.ones(long_window), 'valid') / long_window

    # Find the minimum length among all dc_events lists
    min_length = min(len(dc_events) for dc_events in dc_events_list)

    for i in range(min_length):
        dict_forecast = {}
        wb, ws = 0, 0
        list_b, list_s = [], []

        for j, threshold in enumerate(thresholds):
            dc_events = dc_events_list[j]
            func = funcs[j]
            scaler_X = scaler_X_list[j]
            scaler_y = scaler_y_list[j]
            weight = weights[j]

            dc_event = dc_events[i]
            dc_length = np.array([dc_events[i + 1][1] - dc_events[i][1] if i + 1 < len(dc_events) else len(price_series) - dc_events[i][1]]).reshape(-1, 1)
            dc_length_normalized = scaler_X.transform(dc_length)
            predicted_os_length_normalized = func(dc_length_normalized[0][0])
            trp = scaler_y.inverse_transform([[predicted_os_length_normalized]])[0][0]

            if dc_event[0] == 'upturn':
                list_s.append(trp)
                ws += weight
            else:  # downturn
                list_b.append(trp)
                wb += weight

        if ws > wb:
            trp_optimal = optimize_trp(list_s)
            signal = -1  # Sell signal
        else:
            trp_optimal = optimize_trp(list_b)
            signal = 1  # Buy signal

        # Apply moving average crossover filter
        if i >= long_window - 1:
            ma_index = i - (long_window - 1)
            if short_ma[ma_index] > long_ma[ma_index] and signal == 1:
                signals.append((dc_events_list[0][i][1], signal, trp_optimal))
            elif short_ma[ma_index] < long_ma[ma_index] and signal == -1:
                signals.append((dc_events_list[0][i][1], signal, trp_optimal))
            else:
                signals.append((dc_events_list[0][i][1], 0, trp_optimal))  # No trade
        else:
            signals.append((dc_events_list[0][i][1], 0, trp_optimal))  # No trade

    return signals

def optimize_trp(trp_list: List[float]) -> float:
    # Implement the optimization according to Equation 3
    # For simplicity, we'll use the mean of the TRPs
    return np.mean(trp_list) if trp_list else 0.0
