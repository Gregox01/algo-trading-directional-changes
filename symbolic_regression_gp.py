import numpy as np
from deap import base, creator, gp, tools, algorithms
import operator
import random
import math
import functools
import operator
import random
import math
import functools
import deap.benchmarks.tools as benchmarks
from utils import save_checkpoint, load_checkpoint
from sklearn.model_selection import train_test_split



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
    pset.addEphemeralConstant("rand101", functools.partial(random.randint, -1, 1))
    pset.renameArguments(ARG0='x')

    # Check if classes exist before creating
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
    predictions = [func(x[0]) for x in X]
    mse = np.mean((np.array(predictions) - y) ** 2)
    return (mse,)

def evolve_srgp(X, y, checkpoint_file='srgp_checkpoint.pkl'):
    toolbox, pset = setup_symbolic_regression()
    toolbox.register("evaluate", eval_symbolic_regression, toolbox=toolbox, X=X, y=y)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    population_size = 300
    num_generations = 50

    # Try to load a checkpoint if it exists
    try:
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            pop, _ = checkpoint
            print(f"Loaded checkpoint from {checkpoint_file}. Continuing evolution...")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_file}. Starting new evolution...")
        pop = toolbox.population(n=population_size)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, num_generations, stats=stats,
                                   halloffame=hof, verbose=True)

    # Save the final population as a checkpoint
    save_checkpoint((pop, toolbox), checkpoint_file)

    return hof[0]

def run_symbolic_regression(X, y, checkpoint_file='srgp_checkpoint.pkl'):
    best_individual = evolve_srgp(X, y, checkpoint_file)
    toolbox, _ = setup_symbolic_regression()
    return best_individual, toolbox
