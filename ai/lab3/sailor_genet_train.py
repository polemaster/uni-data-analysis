"""
    Genetic algoritm for sailor problem.
    Sailor sails from left to right side of the map bypassing dangerous places (with negative rewards).
    The environment is nondeterministic due to the wind and waves which means that the boat moves to chosen
    places only with constant probabilities. If you want to evaluate sailor's strategy in the reliable way,
    you need to use many episodes for strategy evaluation.

    In this task you need to invent crossover and mutation operations, invent the evaluation method, choose
    the best parameters values of the evolution, maybe add more parameters and operations. Your solutions
    should work for different maps and different gamma values.
"""

import os
import pdb
import time

import matplotlib.pyplot as plt
import numpy as np
import sailor_funct as sf

# Evolution parameters ..... (find the best values)
number_of_simulations = 100000  # do not change (time complexity parameter)
number_of_individuals = 20  # number of individuals in population (each individual conatains sailor strategy)
number_of_episodes_for_eval = 5  # number of epizodes for strategy evaluation

p_cross = 0.6  # crossover probability
p_mut = 0.5  # mutation probability
if_elitism = False  # the best individual goes to the next population unchanged
selection_pressure = 1.0  # if higher -> more copies of the best individuals in new population expense of worst individuals

fitness_limit = 7


# Task definition parameters ............ (begin from easy one, and try mid one later)
# file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
file_name = "map_simple.txt"
# file_name = 'map_mid.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'
gamma = 1  # discount factor (part of a task - do not change). If gamma < 1, rewards with longer time distance are less important
# (it pays an agent to get positive rewards as soon as possible and penalties as long as possible)

# number fo epochs of evolution with time complexity preservation:
number_of_epochs = number_of_simulations // (
    number_of_individuals * number_of_episodes_for_eval
)
# number_of_epochs = 10
reward_map = sf.load_data(file_name)  # load map of rewards from file
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(
    2.5 * (num_of_rows + num_of_columns)
)  # maximum number of steps in an episode
Popul = np.random.randint(
    1, 5, (number_of_individuals, num_of_rows, num_of_columns)
)  # population of strategies

# print("Initial population = " + str(Popul))


def reproduction(Popul, fitnesses):  # new population based on fittness values
    fit_cum = np.copy(fitnesses)
    for i in range(fitnesses.size - 1):  # cumulative sum
        fit_cum[i + 1] += fit_cum[i]

    max_cum_value = fit_cum[fitnesses.size - 1]
    Popul_new = np.copy(Popul)

    for i in range(fitnesses.size):
        rand_value = np.random.random() * max_cum_value
        for j in range(fit_cum.size):
            prev_val = 0
            if j > 0:
                prev_val = fit_cum[j - 1]
            if (rand_value > prev_val) & (rand_value <= fit_cum[j]):
                parent_index = j
                break
        Popul_new[i, ...] = np.copy(Popul[j, ...])
    return Popul_new


def fitness_func(value):
    # return max(0, value)
    return 1 / (1 + np.exp(-value))


def crossover_func(parent_1, parent_2):
    # print(f"before:\nparent1: {parent_1}\nparent2: {parent_2}")
    end_row = num_of_rows // 2
    end_col = num_of_columns // 2

    # Selecting the left top corner (a quarter of the matrix)
    corner1 = np.copy(parent_1[:end_row, :end_col])
    corner2 = np.copy(parent_2[:end_row, :end_col])

    # Switching the values
    parent_1[:end_row, :end_col] = corner2
    parent_2[:end_row, :end_col] = corner1
    # print(f"after:\noffspring1: {parent_1}\noffsprint2: {parent_2}")

    return parent_1, parent_2


# end of reproduction

maximum_mean_sum_of_rewards = -1000000000


for epoch in range(number_of_epochs):
    # evaluation of individuals:
    mean_sums_of_rewards = np.zeros([number_of_individuals])
    minimum_mean_sum_of_rewards = 1000000000
    fitnesses = np.zeros([number_of_individuals])
    for individual in range(number_of_individuals):
        mean_sums_of_rewards[individual] = sf.sailor_test(
            reward_map, Popul[individual, ...], number_of_episodes_for_eval, gamma
        )
        if minimum_mean_sum_of_rewards > mean_sums_of_rewards[individual]:
            minimum_mean_sum_of_rewards = mean_sums_of_rewards[individual]
        if maximum_mean_sum_of_rewards < mean_sums_of_rewards[individual]:
            maximum_mean_sum_of_rewards = mean_sums_of_rewards[individual]
            best_strategy = np.copy(Popul[individual, ...])
            # if maximum_mean_sum_of_rewards >= fitness_limit:
            #     break
            print(
                "new best individual = "
                + str(maximum_mean_sum_of_rewards)
                + " in epoch "
                + str(epoch)
            )
        # mean_sums_of_rewards += minimum_mean_sum_of_rewards
        fitnesses[individual] = fitness_func(mean_sums_of_rewards[individual])
        # fitnesses[individual] = mean_sums_of_rewards[individual]
    # print(fitnesses)
    # print(sum(fitnesses))

    # Fitness values must be >= 0 and higher as individual better. You can use selection_factor as an exponent of
    # mean_sums_of_rewards to adjust selection pressure - expected number of copies of the best individuals against the
    # existence of the worst.
    # for individual in range(number_of_individuals):
    #     fitnesses[individual] = 1  # for now ............
    # selection pressure can be used ....
    # rank reproduction can be used ...

    # print(
    #     "epoch = "
    #     + str(epoch)
    #     + " avarage sum of rewards over population = "
    #     + str(np.mean(mean_sums_of_rewards))
    # )

    # Reproduction ...... can be rank or roulette version
    Popul = reproduction(Popul, fitnesses)  # roulette version

    # Elitism
    next_popul = np.copy(Popul)

    sorted_indices = np.argsort(mean_sums_of_rewards)
    sorted_pop = [Popul[i] for i in sorted_indices]

    next_popul[0] = np.copy(sorted_pop[0])
    next_popul[1] = np.copy(sorted_pop[1])

    # Crossover .......
    # for j in range(0, number_of_individuals, 2):
    for j in range(2, number_of_individuals, 2):
        if np.random.random() < p_cross:
            parent_1 = np.copy(Popul[j])
            parent_2 = np.copy(Popul[j + 1])
            offspring_1, offspring_2 = crossover_func(parent_1, parent_2)
            next_popul[j, ...] = offspring_1
            next_popul[j + 1, ...] = offspring_2
        # else:
        #     next_popul[j, ...] = parent_1
        #     next_popul[j + 1, ...] = parent_2

    # if epoch == 0:
    #     print("popul:", Popul)
    #     print("next_popul:", next_popul)

    Popul = np.copy(next_popul)

    # Mutation .......
    for j in range(number_of_individuals):
        for row in range(num_of_rows):
            for col in range(num_of_columns):
                if np.random.random() < p_mut:
                    Popul[j, row, col] = np.random.randint(1, 5)

    # Other operations (elitism, niches, parameter changing functions etc.) ...

# end of evolution loop

mean_sum_of_rewards = sf.sailor_test(reward_map, best_strategy, 1000, gamma)
print(
    "Test 1000 episodes for best strategy - avg.sum of rewards = "
    + str(mean_sum_of_rewards)
)
sf.draw(reward_map, best_strategy, mean_sum_of_rewards)
# print("Final population = " + str(Popul))
