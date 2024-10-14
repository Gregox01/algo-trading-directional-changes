import numpy as np
from deap import algorithms, base, creator, tools, gp
import operator
import math
import random
import matplotlib.pyplot as plt
from directional_changes import load_and_prepare_data, identify_dc_events, identify_os_events
import functools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
        os_length = os_end - os_start + 1  # +1 to include both start and end points

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
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addEphemeralConstant("rand101", functools.partial(random.randint, -1, 1))
    pset.renameArguments(ARG0='x')

    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox, pset

def eval_symbolic_regression(individual, toolbox, X, y):
    func = toolbox.compile(expr=individual)
    predictions = []
    for x in X:
        try:
            prediction = func(x[0])
            if not np.isfinite(prediction):
                prediction = np.nan
        except (OverflowError, ZeroDivisionError, ValueError):
            prediction = np.nan
        predictions.append(prediction)

    predictions = np.array(predictions)
    valid_indices = ~np.isnan(predictions)
    if np.sum(valid_indices) == 0:
        return (float('inf'),)  # Penalize individuals that produce all invalid predictions
    mse = ((predictions[valid_indices] - y[valid_indices]) ** 2).mean()
    return (mse,)

def run_symbolic_regression(X, y, population_size=300, num_generations=50):
    toolbox, pset = setup_symbolic_regression()

    toolbox.register("evaluate", eval_symbolic_regression, toolbox=toolbox, X=X, y=y)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, num_generations, stats=mstats,
                                   halloffame=hof, verbose=True)

    return pop, log, hof, toolbox

def plot_results(X, y, best_individual, toolbox):
    func = toolbox.compile(expr=best_individual)
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = [func(x[0]) for x in X_plot]

    # Sort X and y for plotting
    sorted_indices = X.flatten().argsort()
    X_sorted = X.flatten()[sorted_indices]
    y_sorted = y[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_sorted, y_sorted, alpha=0.5, label='Actual')
    plt.plot(X_plot, y_pred, 'r-', label='Predicted')
    plt.xlabel('DC Length')
    plt.ylabel('OS Length')
    plt.title('DC Length vs OS Length')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load and prepare the data
    file_path = 'ETHUSDT_15m.csv'
    df = load_and_prepare_data(file_path)

    # Use the entire dataset for this example
    price_series = df['Close'].values
    threshold = 0.015  # 1.5% threshold

    # Identify DC and OS events
    dc_events = identify_dc_events(price_series, threshold)
    os_events = identify_os_events(dc_events, len(price_series))

    # Prepare data for regression
    X, y, scaler_X, scaler_y = prepare_data_for_regression(dc_events, os_events, len(price_series))

    print(f"Number of Data Points: {len(X)}")

    # After preparing X and y in the main function
    print("Sample DC and OS Lengths:")
    for dc_length, os_length in zip(X.flatten()[:10], y[:10]):
        print(f"DC Length: {dc_length}, OS Length: {os_length}")

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run symbolic regression on training data
    pop, log, hof, toolbox = run_symbolic_regression(X_train, y_train)

    best_individual = hof[0]
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_individual.fitness.values[0])
    print(f"Best Individual Expression: {str(best_individual)}")

    # Plot results
    plot_results(X, y, best_individual, toolbox)

    # Evaluate on test data
    func = toolbox.compile(expr=best_individual)
    y_pred = [func(x[0]) for x in X_test]
    mse = ((np.array(y_pred) - y_test) ** 2).mean()
    print(f"Mean Squared Error on test data: {mse}")

    # Generate predictions for new DC lengths
    new_dc_lengths = np.linspace(X.min(), X.max(), 10).reshape(-1, 1)
    func = toolbox.compile(expr=best_individual)
    predicted_os_lengths = [func(x[0]) for x in new_dc_lengths]

    print("\nPredictions for new DC lengths:")
    for dc_length, os_length in zip(new_dc_lengths, predicted_os_lengths):
        print(f"DC Length: {dc_length[0]:.2f}, Predicted OS Length: {os_length:.2f}")

    # TODO: Implement trading strategy based on these predictions
    # TODO: Perform backtesting and evaluate strategy performance
