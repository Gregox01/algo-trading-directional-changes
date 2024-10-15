from utils import load_checkpoint
from signal_generation import generate_signals_multi_threshold

def live_trading():
    # Load saved models
    models = []
    for i in range(len(thresholds)):
        best_individual, toolbox = load_checkpoint(f'best_model_{i}.pkl')
        models.append(toolbox.compile(expr=best_individual))

    # Use models for live trading
    # This is a simplified example; you'll need to adapt it to your live data feed
    while True:
        # Get latest price data
        latest_price_data = get_latest_price_data()

        # Generate signals
        signals = generate_signals_multi_threshold(dc_events_list, scaler_X_list, scaler_y_list, models, thresholds, latest_price_data, weights)

        # Execute trades based on signals
        execute_trades(signals)

        # Wait for next update
        time.sleep(update_interval)
