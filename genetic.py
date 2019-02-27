#!/usr/bin/python3

'''
P15/37272/2016
Genetic algorithm
One Max Problem
'''

'''
Creates a population of individuals
Consisting of random integer vectors filled with (0|1).
The population then evolves until one of its members contains only 1s (no 0s)
'''

import sys
import random
import numpy

from deap import base, creator, tools, algorithms

# Genealogy plotting
import matplotlib.pyplot as mplt
import networkx as nx

# Define some sufficient range
POPULATION_SIZE = 100
GENERATIONS_LIMIT = int(25 + (1000 - POPULATION_SIZE) * 0.13)


def evaluate_ones(individual):
    # Return a sum of constituent integers
    return sum(individual),
# End evaluate_ones


def main():
    global POPULATION_SIZE
    global GENERATIONS_LIMIT

    print("[*] Genetic algorithm: One Max Problem")

    # Create classes for evolution
    # Inherit fitness class with additionalatribute weights
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # create Individual class inheriting the class "list" & FitnessMax class in its fitness attribute.
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    history = tools.History()

    # Register creates aliases to functions
    # Register generation function, random integer of 1s &/or 0s
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Register individual initialization function
    toolbox.register("individual", tools.initRepeat, creator.Individual,
        toolbox.attr_bool, 100)
    # Register population inintialization function
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function
    toolbox.register("evaluate", evaluate_ones)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # For plotting
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    # Instantiate a population (list) of POPULATION_SIZE members
    pop = toolbox.population(n=POPULATION_SIZE)
    # Update history
    history.update(pop)

    # The underlying algorithm
    '''
    cross_probability = 0.5
    mutation_probability = 0.2

    # Evaluate the population
    fitnesses = map(toolbox.evaluate, pop)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Obtain fitness values for individuals
    fits = [ind.fitness.values[0] for ind in pop]

    generations = 0
    target = 100

    while (max(fits) < target and generations < GENERATIONS_LIMIT):
        generations += 1

        print("[+] Generation: " + str(generations))

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        # Individuals are modified inplace, ensures references aren't used
        offspring = list(map(toolbox.clone, offspring))

        # Crossover offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cross_probability:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutate the offspring
        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population with current offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        sum2 = sum(x*x for x in fits)

        print("[+] Min %s" % min(fits))
        print("[+] Max %s" % max(fits))
    # End while

    mean = sum(fits) / length
    std = abs(sum2 / length - mean**2)**0.5
    print("[+] Mean %s" % mean)
    print("[+] Std dev %s" % std)
    '''
    # End of the underlying algorithm

    # '''
    # Simpler & cleaner implementation
    # Output formatting in columns
    # Iterates

    # The hall of fame contains the best individual that ever lived
    # in the population during the evolution.
    hof = tools.HallOfFame(1)  # Only one individual
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Mean", numpy.mean)
    stats.register("Std dev", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    # cxpb  - Crossover probability
    # mutpb - Mutation probability
    # ngen  - Max generations number, always iterates to this point
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS_LIMIT, stats=stats, halloffame=hof, verbose=True)
    # '''

    #'''
    # Takes some time to plot
    mplt.title("Genetic Algorithm, one max")
    graph = nx.DiGraph(history.genealogy_tree)
    # Make the grah top-down
    graph = graph.reverse()
    colours = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    nx.draw(graph, node_color=colours)
    mplt.show()
    #'''
# End main


if(__name__ == "__main__"):
    main()
