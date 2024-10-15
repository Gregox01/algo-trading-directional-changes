import numpy as np
from sklearn.preprocessing import MinMaxScaler
from directional_changes import load_and_prepare_data, identify_dc_events, identify_os_events
import operator
import functools
import random
from deap import algorithms, base, creator, tools, gp
import math

# Define protected functions
def protected_div(left, right):
    try:
        return left / right if right != 0 else 1.0
    except ZeroDivisionError:
        return 1.0

def protected_log(x):
    return math.log(abs(x)) if x != 0 else 0.0

def prepare_data_for_regression(dc_events, os_events, price_series_length):
    dc_lengths = []
    os_lengths = []
    for i in range(len(os_events)):
        dc_start = dc_events[i][1]
        dc_end = dc_events[i + 1][1] if i + 1 < len(dc_events) else price_series_length - 1
        os_start, os_end = os_events[i]

        dc_length = dc_end - dc_start
        os_length = os_end - os_start  # OS length is from os_start to os_end

        dc_lengths.append(dc_length)
        os_lengths.append(os_length)

    X = np.array(dc_lengths).reshape(-1, 1)
    y = np.array(os_lengths)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X_normalized, y_normalized, scaler_X, scaler_y

# Set up the symbolic regression
def setup_symbolic_regression():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    # Remove trigonometric functions for simplicity
    # pset.addPrimitive(protected_log, 1)
    # pset.addPrimitive(math.sin, 1)
    # pset.addPrimitive(math.cos, 1)
    pset.addEphemeralConstant("rand101", functools.partial(random.randint, -1, 1))
    pset.renameArguments(ARG0='x')

    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox, pset

def prepare_all_data(file_path, thresholds):
    df = load_and_prepare_data(file_path)
    price_series = df['Close'].values

    data = []
    for threshold in thresholds:
        dc_events = identify_dc_events(price_series, threshold)
        os_events = identify_os_events(dc_events, len(price_series))
        X, y, scaler_X, scaler_y = prepare_data_for_regression(dc_events, os_events, len(price_series))
        data.append((dc_events, os_events, X, y, scaler_X, scaler_y))

    return price_series, data
