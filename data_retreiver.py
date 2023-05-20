from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd

# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# request connection status and parameters
print(mt5.terminal_info())
# get data on MetaTrader 5 version
print(mt5.version())

# get bars from EURUSD within a specific range of time
eurusd_rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H1, datetime(2023,1,1,0), datetime(2023,5,20,0))

# shut down connection to MetaTrader 5
mt5.shutdown()

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(eurusd_rates)

# convert time in seconds into the datetime format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')

# save the data to a CSV file
rates_frame.to_csv('EURUSD_H1.csv', index=False)
