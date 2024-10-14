import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file and prepare the data for analysis.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Prepared DataFrame with datetime index and close prices.
    """
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df['Close'] = df['Close'].astype(float)

    # Handle missing values
    df['Close'].replace(0, method='ffill', inplace=True)
    df.dropna(inplace=True)

    return df

def segment_data_monthly(df: pd.DataFrame) -> dict:
    """
    Segment the data into monthly groups.

    Args:
    df (pd.DataFrame): DataFrame with datetime index and close prices.

    Returns:
    dict: Dictionary with monthly data segments.
    """
    monthly_groups = df.groupby(pd.Grouper(freq='M'))
    return {month.strftime('%Y-%m'): group['Close'].values for month, group in monthly_groups}

def identify_dc_events(price_series: np.ndarray, threshold: float) -> List[Tuple[str, int, float]]:
    dc_events = []
    event_type = None
    extremum_price = price_series[0]
    extremum_index = 0

    for i in range(1, len(price_series)):
        current_price = price_series[i]
        price_change = (current_price - extremum_price) / extremum_price

        if event_type == 'upturn':
            if price_change <= -threshold:
                # Downturn detected
                dc_events.append(('downturn', extremum_index, extremum_price))
                event_type = 'downturn'
                extremum_price = current_price
                extremum_index = i
            elif current_price > extremum_price:
                # New high during upturn
                extremum_price = current_price
                extremum_index = i
        elif event_type == 'downturn':
            if price_change >= threshold:
                # Upturn detected
                dc_events.append(('upturn', extremum_index, extremum_price))
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
                dc_events.append(('upturn', extremum_index, extremum_price))
                extremum_price = current_price
                extremum_index = i
            elif initial_price_change <= -threshold:
                event_type = 'downturn'
                dc_events.append(('downturn', extremum_index, extremum_price))
                extremum_price = current_price
                extremum_index = i

    return dc_events

def identify_os_events(dc_events: List[Tuple[str, int, float]], price_series_length: int) -> List[Tuple[int, int]]:
    os_events = []
    for i in range(len(dc_events) - 1):
        os_start = dc_events[i][1] + 1  # Start immediately after the DC event
        os_end = dc_events[i + 1][1]    # End at the next DC event
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

def handle_flat_market(price_series: np.ndarray, threshold: float, max_flat_period: int) -> List[Tuple[str, int, float]]:
    """
    Handle flat market periods by forcing events after a maximum flat period.

    Args:
    price_series (np.ndarray): Array of price values.
    threshold (float): The threshold for identifying DC events.
    max_flat_period (int): Maximum number of periods to allow without an event.

    Returns:
    List[Tuple[str, int, float]]: List of DC events including forced events.
    """
    dc_events = identify_dc_events(price_series, threshold)
    forced_events = []
    last_event_index = 0

    for i in range(1, len(price_series)):
        if i - last_event_index > max_flat_period:
            if price_series[i] > price_series[last_event_index]:
                forced_events.append(('upturn', i, price_series[i]))
            else:
                forced_events.append(('downturn', i, price_series[i]))
            last_event_index = i
        elif any(event[1] == i for event in dc_events):
            last_event_index = i

    all_events = sorted(dc_events + forced_events, key=lambda x: x[1])
    return all_events

def visualize_dc_os(price_series: np.ndarray, dc_events: List[Tuple[str, int, float]], os_events: List[Tuple[int, int]]):
    """
    Visualize the price series with DC events and OS periods.

    Args:
    price_series (np.ndarray): Array of price values.
    dc_events (List[Tuple[str, int, float]]): List of DC events.
    os_events (List[Tuple[int, int]]): List of OS periods.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(price_series, label='Price')

    for event_type, index, price in dc_events:
        color = 'g' if event_type == 'upturn' else 'r'
        plt.scatter(index, price, c=color, marker='^' if event_type == 'upturn' else 'v')

    for start, end in os_events:
        plt.axvspan(start, end, alpha=0.1, color='yellow')

    plt.legend()
    plt.title('Price Series with DC Events and OS Periods')
    plt.xlabel('Time')
    plt.ylabel('Price')
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
    threshold = 0.015  # 3% threshold

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
    visualize_dc_os(price_series, dc_events, os_events)
