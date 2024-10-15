import random

def initialize_weights(number_of_thresholds, population_size):
    population = []

    for _ in range(population_size):
        chromosome = []

        # Initialize weights for threshold positions
        for i in range(number_of_thresholds):
            chromosome.append(1.0)

        # Initialize remaining weights randomly
        for _ in range(population_size - number_of_thresholds):
            chromosome.append(random.uniform(0.0, 1.0))

        population.append(chromosome)

    return population

def uniform_crossover(parent1, parent2):
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append(gene1)
        else:
            child.append(gene2)
    return child

def uniform_mutation(parent):
    child = parent.copy()
    for i in range(len(child)):
        if random.random() < 0.5:
            child[i] = random.uniform(0.0, 1.0)
    return child

def select_best_individual(population, fitness_func):
    return max(population, key=fitness_func)

def genetic_algorithm(population_size, number_of_thresholds, generations, fitness_func):
    population = initialize_weights(number_of_thresholds, population_size)

    for _ in range(generations):
        new_population = []

        while len(new_population) < population_size:
            parent1 = select_best_individual(population, fitness_func)
            parent2 = select_best_individual(population, fitness_func)

            child = uniform_crossover(parent1, parent2)
            child = uniform_mutation(child)

            new_population.append(child)

        population = new_population

    best_individual = select_best_individual(population, fitness_func)
    return best_individual