import pandas as pd
import requests
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from autosklearn.classification import AutoSklearnClassifier
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

csv_file_path = "/content/EURUSD_H1.csv"  # Replace this with your actual path

# Load data from the CSV file
data = pd.read_csv(csv_file_path)

# Convert 'time' column to datetime if it's not
data['time'] = pd.to_datetime(data['time'])

# Set datetime as index if it's not
data.set_index('time', inplace=True)

# Define the thresholds for DC and OS events
dc_threshold = 0.0001041666667
os_threshold = 0.0002

# Define the window length for calculating returns
window_length = 5

data.dropna(subset=['close'], inplace=True)

# Calculate the returns over the specified window length
data['return'] = data['close'].pct_change(window_length)

# Identify DC and OS events
data['dc_event'] = np.where(data['return'] > dc_threshold, 1, np.where(data['return'] < -dc_threshold, -1, 0))
data['os_event'] = np.where(data['return'] > os_threshold, 1, np.where(data['return'] < -os_threshold, -1, 0))

# Print the indicator in the chart

# Create a new figure
plt.figure(figsize=(14, 7))

# Plot the closing price
plt.plot(data['close'], label='Close Price', color='blue')

# Plot the DC events
dc_events = data[data['dc_event'] != 0]
plt.scatter(dc_events.index, dc_events['close'], color='red', label='DC Events')

# Plot the OS events
os_events = data[data['os_event'] != 0]
plt.scatter(os_events.index, os_events['close'], color='green', label='OS Events')

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Show the plot
plt.show()




# Assume the last column is the target
X = data[['return', 'dc_event', 'os_event']]
y = data['dc_event']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the AutoSklearnClassifier
auto_classifier = AutoSklearnClassifier(time_left_for_this_task=90, per_run_time_limit=40)

# Fit the classifier to the training data
auto_classifier.fit(X_train, y_train)

# Use the classifier to make predictions on the test data
predictions = auto_classifier.predict(X_test)



# Calculate the lengths of the DC and OS events
data['dc_length'] = data['dc_event'].rolling(window=10).sum()
data['os_length'] = data['os_event'].rolling(window=10).sum()

# Fill NaN values with 0
data = data.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the symbolic regressor
regressor = SymbolicRegressor(population_size=5000, function_set=['add', 'sub', 'mul', 'div'], generations=20, stopping_criteria=0.01, p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=0.9, verbose=1, parsimony_coefficient=0.01, random_state=0)

# Fill NaN values with 0 in the training data
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)

# Fit the regressor to the training data
regressor.fit(X_train, y_train)

# Fill NaN values with 0 in the test data
X_test = X_test.fillna(0)

# Use the regressor to make predictions on the test data
predictions = regressor.predict(X_test)



# Define the fitness function
def fitness(weights):
    # Initialize the total profit and position
    total_profit = 0
    position = 0

    # Loop over the price data
    for i in range(1, len(data)):
        # Determine the action to take based on the weights and the DC and OS events
        if data.iloc[i]['dc_event'] == 1 and position == 0:
            # Buy when a DC event occurs
            position = 1 / data.iloc[i]['close']
        elif data.iloc[i]['os_event'] == 1 and position > 0:
            # Sell half of the position when an OS event occurs
            total_profit += position * data.iloc[i]['close'] / 2
            position /= 2
        elif data.iloc[i]['dc_event'] == 1 and position > 0:
            # Sell the rest of the position when the next DC event occurs
            total_profit += position * data.iloc[i]['close']
            position = 0

    # Return the total profit as the fitness
    return total_profit,


# Create the toolbox
toolbox = base.Toolbox()
# Define the individual and population


# Define the individual and population
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the genetic operators
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create the initial population
pop = toolbox.population(n=50)

# Run the genetic algorithm
result = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)

# Extract the fitness of the best individual in each generation
best_fitnesses = [ind.fitness.values[0] for ind in result[0]]

# Calculate the price change
data['price_change'] = data['close'].diff()

# Create a new column that indicates whether the price goes up (1) or down (-1) in the next time step
data['trend'] = np.sign(data['price_change'].shift(-1))

# Remove the rows with NaN values
data = data.dropna()

# Initialize 'dc_start' and 'dc_end' columns
data['dc_start'] = np.nan
data['dc_end'] = np.nan

# Loop through the DataFrame
for i in range(1, len(data)):
    # If 'dc_event' is 1, set 'dc_start' to the current index
    if data.iloc[i]['dc_event'] == 1:
        data.loc[i, 'dc_start'] = i
    # If 'dc_event' is -1 and 'dc_start' is not NaN, set 'dc_end' to the current index
    elif data.iloc[i]['dc_event'] == -1 and not np.isnan(data.iloc[i]['dc_start']):
        data.loc[i, 'dc_end'] = i

# Initialize 'os_start' and 'os_end' columns
data['os_start'] = np.nan
data['os_end'] = np.nan

# Loop through the DataFrame
for i in range(1, len(data)):
    # If 'os_event' is 1, set 'os_start' to the current index
    if data.iloc[i]['os_event'] == 1:
        data.loc[i, 'os_start'] = i
    # If 'os_event' is -1 and 'os_start' is not NaN, set 'os_end' to the current index
    elif data.iloc[i]['os_event'] == -1 and not np.isnan(data.iloc[i]['os_start']):
        data.loc[i, 'os_end'] = i


# Calculate the DC and OS lengths
data['dc_length'] = data['dc_end'] - data['dc_start']
data['os_length'] = data['os_end'] - data['os_start']

# Assume 'dc_length' and 'os_length' are your features and 'trend' is your target
X = data[['dc_length']]
y = data['os_length']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Also split the original data into training and testing sets
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

# Create a new figure
plt.figure(figsize=(14, 7))

# Plot the closing price
plt.plot(data['close'], label='Close Price', color='blue')

# Initialize the position
position = 0
# Initialize the position and total profit
total_profit = 0

data['close'].fillna(method='ffill', inplace=True)

# Loop over the test data
for i in range(len(predictions)):  # change this line
    # Get the current price and the predicted trend reversal point
    price = data_test.iloc[i]['close']
    predicted_trend_reversal = predictions[i]
    print(f"Price: {price}, Predicted Trend Reversal: {predicted_trend_reversal}")


    # If a DC event is predicted and we don't have a position, mark a buy point
    if predicted_trend_reversal == 1 and position == 0:
        position = 1 / price
        print(f"Buy at Price: {price}, Position: {position}")
        plt.scatter(data_test.index[i], price, color='green', label='Buy')

    # If an OS event is predicted and we have a position, mark a sell point
    elif predicted_trend_reversal == -1 and position > 0:
        position = 0
        print(f"Sell at Price: {price}, Position: {position}")
        plt.scatter(data_test.index[i], price, color='red', label='Sell')

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.gca().invert_xaxis()

# Show the plot
plt.show()
