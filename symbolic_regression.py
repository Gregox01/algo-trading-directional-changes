"""Symbolic regression (DEAP genetic programming) predicting OS length from DC length.

Honest problem definition
-------------------------
For DC event k, everything known at its confirmation bar c_k is fair game; anything
later is not. So:

    feature  dc_length_k = confirmation_index_k - extremum_index_k
    target   os_length_k = max(0, extremum_index_{k+1} - confirmation_index_k)

i.e. "how much longer does the trend run after we learn about it". The original
version of this module measured dc_length extremum-to-extremum, which *contains*
the OS period being predicted (X ~= y + small term) — any fit quality obtained
that way is target leakage, not skill. It also fit its MinMaxScaler on the full
dataset before splitting. Both are fixed here: lengths are used raw through a
log1p transform, and all fitting happens on whatever subset the caller passes in.

Expressions are evaluated as numpy ufunc trees, so each individual is scored in a
single vectorized call over the whole training set.
"""

import functools
import math
import operator
import random

import numpy as np
from deap import algorithms, base, creator, tools, gp
import matplotlib.pyplot as plt


# --- Protected numpy primitives -------------------------------------------------

def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(np.abs(right) > 1e-12, np.divide(left, right), 1.0)
    return out


def protected_log(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(np.abs(x) > 1e-12, np.log(np.abs(np.where(np.abs(x) > 1e-12, x, 1.0))), 0.0)
    return out


def _rand_const():
    return random.uniform(-1, 1)


def prepare_data_for_regression(dc_events, min_pairs=2):
    """
    Build the leak-free (dc_length, os_length) dataset from confirmation-aware
    DC events (5-tuples). The last event has no known next extremum and is dropped.

    Returns:
    (dc_lengths, os_lengths): two int arrays of equal length.
    """
    dc_lengths = []
    os_lengths = []
    for k in range(len(dc_events) - 1):
        conf_idx = dc_events[k][3]
        ext_idx = dc_events[k][1]
        next_ext_idx = dc_events[k + 1][1]
        dc_lengths.append(conf_idx - ext_idx)
        os_lengths.append(max(0, next_ext_idx - conf_idx))
    if len(dc_lengths) < min_pairs:
        raise ValueError(f"Not enough DC events to build a dataset ({len(dc_lengths)} pairs)")
    return np.asarray(dc_lengths, dtype=float), np.asarray(os_lengths, dtype=float)


# --- GP setup --------------------------------------------------------------------

_PSET = None


def _get_pset():
    global _PSET
    if _PSET is None:
        pset = gp.PrimitiveSet("MAIN", 1)
        pset.addPrimitive(np.add, 2, name="add")
        pset.addPrimitive(np.subtract, 2, name="sub")
        pset.addPrimitive(np.multiply, 2, name="mul")
        pset.addPrimitive(protected_div, 2, name="div")
        pset.addPrimitive(protected_log, 1, name="plog")
        pset.addPrimitive(np.sin, 1, name="sin")
        pset.addPrimitive(np.cos, 1, name="cos")
        pset.addEphemeralConstant("randc", _rand_const)
        pset.renameArguments(ARG0='x')
        _PSET = pset
    return _PSET


def setup_symbolic_regression():
    pset = _get_pset()

    if not hasattr(creator, 'FitnessMinSR'):
        creator.create("FitnessMinSR", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'IndividualSR'):
        creator.create("IndividualSR", gp.PrimitiveTree, fitness=creator.FitnessMinSR)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.IndividualSR, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox, pset


def _safe_eval(func, x_vec):
    """Evaluate a compiled expression over a vector, returning an array with NaN
    where the expression misbehaves."""
    with np.errstate(all='ignore'):
        try:
            pred = func(x_vec)
        except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError):
            return np.full_like(x_vec, np.nan)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 0:
        pred = np.full_like(x_vec, float(pred))
    pred = np.where(np.isfinite(pred), pred, np.nan)
    return pred


def eval_symbolic_regression(individual, toolbox, X, y, parsimony=0.001):
    func = toolbox.compile(expr=individual)
    predictions = _safe_eval(func, X)
    valid = ~np.isnan(predictions)
    if not np.any(valid):
        return (float('inf'),)
    mse = float(((predictions[valid] - y[valid]) ** 2).mean())
    # Invalid predictions and bloat are both penalized
    penalty = 1.0 + parsimony * len(individual) + (1.0 - valid.mean())
    return (mse * penalty,)


def run_symbolic_regression(X, y, population_size=200, num_generations=30, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)
    toolbox, pset = setup_symbolic_regression()

    toolbox.register("evaluate", eval_symbolic_regression, toolbox=toolbox, X=X, y=y)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=12))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=12))

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, num_generations, stats=mstats,
                                   halloffame=hof, verbose=verbose)

    return pop, log, hof, toolbox


def fit_os_predictor(dc_lengths, os_lengths, seed=42, population_size=200,
                     num_generations=30, verbose=False):
    """
    Fit a GP expression predicting OS length (bars) from DC length (bars).

    Both quantities pass through log1p to tame heavy tails; predictions are mapped
    back with expm1 and clipped to [0, max training OS length].

    Returns:
    predict_fn: vectorized, raw bar counts in -> predicted OS bar counts out
    expr_str:   the evolved expression (in log space)
    info:       dict with train MSE (log space), the mean-baseline MSE, and stats
    """
    dc_lengths = np.asarray(dc_lengths, dtype=float)
    os_lengths = np.asarray(os_lengths, dtype=float)
    X = np.log1p(dc_lengths)
    y = np.log1p(os_lengths)

    pop, log, hof, toolbox = run_symbolic_regression(
        X, y, population_size=population_size, num_generations=num_generations,
        seed=seed, verbose=verbose)

    best = hof[0]
    func = toolbox.compile(expr=best)

    y_mean = float(y.mean())
    max_os = float(os_lengths.max())
    fallback = float(np.expm1(y_mean))

    preds_train = _safe_eval(func, X)
    valid = ~np.isnan(preds_train)
    train_mse = float(((preds_train[valid] - y[valid]) ** 2).mean()) if np.any(valid) else float('inf')
    baseline_mse = float(((y - y_mean) ** 2).mean())

    def predict_fn(raw_dc_lengths):
        x = np.log1p(np.asarray(raw_dc_lengths, dtype=float))
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        pred_log = _safe_eval(func, x)
        pred = np.expm1(pred_log)
        pred = np.where(np.isfinite(pred), pred, fallback)
        pred = np.clip(pred, 0.0, max_os)
        return float(pred[0]) if scalar else pred

    info = {
        'train_mse_log': train_mse,
        'baseline_mse_log': baseline_mse,
        'beats_mean_baseline': train_mse < baseline_mse,
        'n_train': int(len(X)),
    }
    return predict_fn, str(best), info


def plot_results(dc_lengths, os_lengths, predict_fn, save_path=None, title='DC Length vs OS Length'):
    x_plot = np.linspace(max(1.0, dc_lengths.min()), dc_lengths.max(), 200)
    y_pred = predict_fn(x_plot)

    plt.figure(figsize=(10, 6))
    plt.scatter(dc_lengths, os_lengths, alpha=0.3, s=10, label='Actual')
    plt.plot(x_plot, y_pred, 'r-', label='GP prediction')
    plt.xlabel('DC Length (bars, confirmation - extremum)')
    plt.ylabel('OS Length (bars after confirmation)')
    plt.title(title)
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    from directional_changes import load_and_prepare_data, identify_dc_events

    file_path = 'ETHUSDT_15m.csv'
    df = load_and_prepare_data(file_path)

    price_series = df['Close'].values
    threshold = 0.015  # 1.5% threshold

    dc_events = identify_dc_events(price_series, threshold)
    dc_lengths, os_lengths = prepare_data_for_regression(dc_events)
    print(f"Number of Data Points: {len(dc_lengths)}")

    # Chronological split: events are already in confirmation order
    n_train = int(len(dc_lengths) * 0.8)
    predict_fn, expr, info = fit_os_predictor(
        dc_lengths[:n_train], os_lengths[:n_train], seed=42, verbose=True)

    print("Best Individual:", expr)
    print("Train MSE (log space):", info['train_mse_log'])
    print("Mean-baseline MSE (log space):", info['baseline_mse_log'])
    print("Beats predicting the mean:", info['beats_mean_baseline'])

    # Honest out-of-sample check on the last 20% of events
    y_test_log = np.log1p(os_lengths[n_train:])
    pred_test_log = np.log1p(predict_fn(dc_lengths[n_train:]))
    test_mse = float(((pred_test_log - y_test_log) ** 2).mean())
    test_baseline = float(((y_test_log - np.log1p(os_lengths[:n_train]).mean()) ** 2).mean())
    print(f"Test MSE (log space): {test_mse:.4f} vs train-mean baseline {test_baseline:.4f}")

    plot_results(dc_lengths, os_lengths, predict_fn, save_path='gp_fit.png')
