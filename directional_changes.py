import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

# A DC event is a 5-tuple:
#   (event_type, extremum_index, extremum_price, confirmation_index, confirmation_price)
# The extremum is where the trend actually reversed, but that is only knowable in
# hindsight: in real time the event becomes known at the confirmation bar, when the
# price has already moved `threshold` away from the extremum. Any trading logic must
# therefore act on confirmation_index, never on extremum_index.
DCEvent = Tuple[str, int, float, int, float]


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file and prepare the data for analysis.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Prepared DataFrame with datetime index, open and close prices.
    """
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df['Close'] = df['Close'].astype(float)
    df['Open'] = df['Open'].astype(float)

    # Handle missing values (a zero price is treated as missing)
    df['Close'] = df['Close'].replace(0, np.nan).ffill()
    df['Open'] = df['Open'].replace(0, np.nan).ffill()
    df.dropna(subset=['Open', 'Close'], inplace=True)

    return df


def segment_data_monthly(df: pd.DataFrame) -> dict:
    """
    Segment the data into monthly groups.

    Args:
    df (pd.DataFrame): DataFrame with datetime index and close prices.

    Returns:
    dict: Dictionary with monthly data segments.
    """
    monthly_groups = df.groupby(pd.Grouper(freq='ME'))
    return {month.strftime('%Y-%m'): group['Close'].values for month, group in monthly_groups}


def identify_dc_events(price_series: np.ndarray, threshold: float) -> List[DCEvent]:
    """
    Identify Directional Change events.

    Each event records both where the trend reversed (the extremum) and where the
    reversal became observable (the confirmation bar, at which the price had moved
    `threshold` relative to the extremum). Events are appended in confirmation
    order, so truncating the price series at bar t yields exactly the events with
    confirmation_index <= t (causality).
    """
    dc_events: List[DCEvent] = []
    event_type = None
    extremum_price = price_series[0]
    extremum_index = 0

    for i in range(1, len(price_series)):
        current_price = price_series[i]
        price_change = (current_price - extremum_price) / extremum_price

        if event_type == 'upturn':
            if price_change <= -threshold:
                # Downturn confirmed at bar i; the peak was the extremum
                dc_events.append(('downturn', extremum_index, extremum_price, i, current_price))
                event_type = 'downturn'
                extremum_price = current_price
                extremum_index = i
            elif current_price > extremum_price:
                # New high during upturn
                extremum_price = current_price
                extremum_index = i
        elif event_type == 'downturn':
            if price_change >= threshold:
                # Upturn confirmed at bar i; the trough was the extremum
                dc_events.append(('upturn', extremum_index, extremum_price, i, current_price))
                event_type = 'upturn'
                extremum_price = current_price
                extremum_index = i
            elif current_price < extremum_price:
                # New low during downturn
                extremum_price = current_price
                extremum_index = i
        else:
            # Initial event type determination
            initial_price_change = (current_price - extremum_price) / extremum_price
            if initial_price_change >= threshold:
                event_type = 'upturn'
                dc_events.append(('upturn', extremum_index, extremum_price, i, current_price))
                extremum_price = current_price
                extremum_index = i
            elif initial_price_change <= -threshold:
                event_type = 'downturn'
                dc_events.append(('downturn', extremum_index, extremum_price, i, current_price))
                extremum_price = current_price
                extremum_index = i

    return dc_events


def identify_os_events(dc_events: List[DCEvent], price_series_length: int) -> List[Tuple[int, int]]:
    """
    Identify Overshoot periods between consecutive DC event extrema.

    Note: these extremum-to-extremum spans are only knowable in hindsight and are
    used for descriptive analysis/visualization. The trading pipeline derives its
    own causal OS definition from confirmation indices (see signals.py).
    """
    os_events = []
    for i in range(len(dc_events) - 1):
        os_start = dc_events[i][1] + 1  # Start immediately after the DC event extremum
        os_end = dc_events[i + 1][1]    # End at the next DC event extremum
        if os_start < os_end:
            os_events.append((os_start, os_end))
    # Handle the last OS period
    if dc_events:
        os_start = dc_events[-1][1] + 1
        os_end = price_series_length - 1
        if os_start < os_end:
            os_events.append((os_start, os_end))
    return os_events


def analyze_os_lengths(os_events: List[Tuple[int, int]]) -> dict:
    """
    Analyze the statistical properties of OS lengths.

    Args:
    os_events (List[Tuple[int, int]]): List of OS periods.

    Returns:
    dict: Statistical properties of OS lengths.
    """
    os_lengths = [end - start for start, end in os_events]
    return {
        'mean': np.mean(os_lengths),
        'median': np.median(os_lengths),
        'std': np.std(os_lengths),
        'min': np.min(os_lengths),
        'max': np.max(os_lengths)
    }


def multi_scale_analysis(price_series: np.ndarray, thresholds: List[float]) -> dict:
    """
    Perform multi-scale analysis using different thresholds.

    Args:
    price_series (np.ndarray): Array of price values.
    thresholds (List[float]): List of thresholds to analyze.

    Returns:
    dict: DC and OS events for each threshold.
    """
    results = {}
    for threshold in thresholds:
        dc_events = identify_dc_events(price_series, threshold)
        os_events = identify_os_events(dc_events, len(price_series))
        results[threshold] = {
            'dc_events': dc_events,
            'os_events': os_events,
            'os_stats': analyze_os_lengths(os_events)
        }
    return results


def handle_flat_market(price_series: np.ndarray, threshold: float, max_flat_period: int) -> List[DCEvent]:
    """
    Handle flat market periods by forcing events after a maximum flat period.

    Forced events are anticipatory by construction (they have no genuine
    confirmation), so they are for descriptive analysis only and must not be fed
    into the trading pipeline. A forced event's confirmation fields equal its own
    bar.
    """
    dc_events = identify_dc_events(price_series, threshold)
    forced_events: List[DCEvent] = []
    last_event_index = 0

    for i in range(1, len(price_series)):
        if i - last_event_index > max_flat_period:
            if price_series[i] > price_series[last_event_index]:
                forced_events.append(('upturn', i, price_series[i], i, price_series[i]))
            else:
                forced_events.append(('downturn', i, price_series[i], i, price_series[i]))
            last_event_index = i
        elif any(event[1] == i for event in dc_events):
            last_event_index = i

    all_events = sorted(dc_events + forced_events, key=lambda x: x[1])
    return all_events


def visualize_dc_os(price_series: np.ndarray, dc_events: List[DCEvent],
                    os_events: List[Tuple[int, int]], save_path: str = None):
    """
    Visualize the price series with DC events and OS periods.

    Extrema are drawn as triangles, confirmation bars as small dots connected to
    their extremum. If save_path is given the figure is written to disk instead of
    shown interactively.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(price_series, label='Price')

    for event in dc_events:
        event_type, index, price = event[0], event[1], event[2]
        color = 'g' if event_type == 'upturn' else 'r'
        plt.scatter(index, price, c=color, marker='^' if event_type == 'upturn' else 'v')
        if len(event) >= 5:
            conf_index, conf_price = event[3], event[4]
            plt.scatter(conf_index, conf_price, c=color, marker='.', s=30)
            plt.plot([index, conf_index], [price, conf_price], c=color, lw=0.5, alpha=0.5)

    for start, end in os_events:
        plt.axvspan(start, end, alpha=0.1, color='yellow')

    plt.legend()
    plt.title('Price Series with DC Events and OS Periods')
    plt.xlabel('Time')
    plt.ylabel('Price')
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Modified main section
if __name__ == "__main__":
    # Load and prepare the data
    file_path = 'ETHUSDT_15m.csv'
    df = load_and_prepare_data(file_path)

    # Segment data monthly
    monthly_data = segment_data_monthly(df)

    # Example analysis for the first month
    first_month = list(monthly_data.keys())[0]
    price_series = monthly_data[first_month]
    threshold = 0.015  # 1.5% threshold

    # Identify DC and OS events
    dc_events = identify_dc_events(price_series, threshold)
    os_events = identify_os_events(dc_events, len(price_series))

    print(f"Analysis for {first_month}:")
    print("Number of DC Events:", len(dc_events))
    print("Number of OS Events:", len(os_events))

    # Analyze OS lengths
    os_stats = analyze_os_lengths(os_events)
    print("OS Length Statistics:", os_stats)

    # Multi-scale analysis
    thresholds = [0.015]
    multi_scale_results = multi_scale_analysis(price_series, thresholds)
    for threshold, results in multi_scale_results.items():
        print(f"Threshold {threshold}:")
        print(f"  Number of DC Events: {len(results['dc_events'])}")
        print(f"  OS Length Statistics: {results['os_stats']}")

    # Handle flat market
    max_flat_period = 50
    dc_events_with_flat = handle_flat_market(price_series, threshold, max_flat_period)
    print("Number of DC Events (including flat market handling):", len(dc_events_with_flat))

    # Visualize
    visualize_dc_os(price_series, dc_events, os_events, save_path='dc_os_first_month.png')
