import numpy as np
from deap import algorithms, base, creator, tools, gp
import operator
import math
import random
import functools


# Define protected functions
def protected_div(left, right):
    try:
        return left / right if right != 0 else 1.0
    except ZeroDivisionError:
        return 1.0

def protected_log(x):
    return math.log(abs(x)) if x != 0 else 0.0

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

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, num_generations, stats=mstats,
                                   halloffame=hof, verbose=True)

    return pop, log, hof, toolbox