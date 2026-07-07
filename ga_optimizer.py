"""Genetic algorithm optimizing threshold weights and risk parameters (doc §4D).

Genome: [w_1 .. w_J, stop_loss_pct, take_profit_pct]. Weights are kept in [0, 1]
and renormalized to sum to 1 after every operator; risk parameters are clipped to
their bounds. Fitness (evaluated on the training window only) is
annualized Sharpe - 2 * max drawdown, with a heavy penalty below a minimum trade
count so that degenerate always-flat genomes cannot win on zero drawdown.

The caller re-evaluates the Hall of Fame on a validation window to pick the
champion — the GA itself never sees validation or test data.
"""

import random

import numpy as np
from deap import algorithms, base, creator, tools

from strategy import weighted_voting, simulate_trading
from metrics import compute_metrics

SL_BOUNDS = (0.01, 0.10)
TP_BOUNDS = (0.01, 0.25)
DD_PENALTY = 2.0
MIN_TRADES = 10


def decode_genome(genome, n_thresholds):
    weights = np.asarray(genome[:n_thresholds], dtype=np.float64)
    stop_loss = float(genome[n_thresholds])
    take_profit = float(genome[n_thresholds + 1])
    return weights, stop_loss, take_profit


def evaluate_genome(genome, B, S, H, open_prices, close_prices, transaction_cost,
                    allow_short=True, min_trades=MIN_TRADES):
    """Run the full voting + simulation for one genome. Returns (fitness, sim, metrics)."""
    n_thresholds = B.shape[0]
    weights, stop_loss, take_profit = decode_genome(genome, n_thresholds)
    actions = weighted_voting(B, S, H, weights)
    sim = simulate_trading(actions, open_prices, close_prices,
                           transaction_cost=transaction_cost,
                           stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                           allow_short=allow_short)
    m = compute_metrics(sim.equity, sim.trades, sim.positions)
    fitness = m['sharpe'] - DD_PENALTY * m['max_drawdown']
    if m['n_trades'] < min_trades:
        fitness -= 5.0
    return fitness, sim, m


def _repair(individual, n_thresholds):
    """Clip genes to bounds and renormalize the weights to sum to 1."""
    for j in range(n_thresholds):
        individual[j] = min(1.0, max(0.0, individual[j]))
    total = sum(individual[:n_thresholds])
    if total > 0:
        for j in range(n_thresholds):
            individual[j] /= total
    else:
        for j in range(n_thresholds):
            individual[j] = 1.0 / n_thresholds
    individual[n_thresholds] = min(SL_BOUNDS[1], max(SL_BOUNDS[0], individual[n_thresholds]))
    individual[n_thresholds + 1] = min(TP_BOUNDS[1], max(TP_BOUNDS[0], individual[n_thresholds + 1]))
    return individual


def optimize_weights(B, S, H, open_prices, close_prices, transaction_cost=0.001,
                     allow_short=True, seed=42, population_size=40,
                     num_generations=30, hof_size=10, verbose=False):
    """
    Run the GA on the training window (B/S/H and prices must already be sliced).

    Returns (hall_of_fame_genomes, logbook): the genomes are plain lists, best
    train fitness first.
    """
    random.seed(seed)
    n_thresholds = B.shape[0]

    if not hasattr(creator, 'FitnessMaxGA'):
        creator.create("FitnessMaxGA", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'IndividualGA'):
        creator.create("IndividualGA", list, fitness=creator.FitnessMaxGA)

    def make_genome():
        weights = [random.random() for _ in range(n_thresholds)]
        total = sum(weights)
        weights = [w / total for w in weights]
        return creator.IndividualGA(
            weights + [random.uniform(*SL_BOUNDS), random.uniform(*TP_BOUNDS)])

    def eval_wrapped(individual):
        fitness, _, _ = evaluate_genome(
            individual, B, S, H, open_prices, close_prices,
            transaction_cost, allow_short)
        return (fitness,)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_genome)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_wrapped)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    sigma = [0.1] * n_thresholds + [0.01, 0.02]
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=sigma, indpb=0.3)

    def repair_decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                _repair(child, n_thresholds)
            return offspring
        return wrapper

    toolbox.decorate("mate", repair_decorator)
    toolbox.decorate("mutate", repair_decorator)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(hof_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3,
                                   ngen=num_generations, stats=stats,
                                   halloffame=hof, verbose=verbose)

    return [list(ind) for ind in hof], log


def select_on_validation(hof_genomes, B_val, S_val, H_val, open_val, close_val,
                         transaction_cost=0.001, allow_short=True):
    """Re-evaluate Hall of Fame genomes on the validation window; return the best
    genome and its validation fitness."""
    best_genome, best_fitness = None, -np.inf
    for genome in hof_genomes:
        fitness, _, _ = evaluate_genome(
            genome, B_val, S_val, H_val, open_val, close_val,
            transaction_cost, allow_short)
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = genome
    return best_genome, best_fitness
