import random

import matplotlib.pyplot as plt
import numpy as np
import sailor_funct as sf

number_of_episodes = 4000  # number of training epizodes DO NOT INCREASE!
gamma = 0.99  # discount factor

# Training data parameters - these may be time-dependent functions:
alpha = 0.1  # training speed factor
epsilon = 1  # exploration factor
min_epsilon = 0.005
max_epsilon = 1
decay_rate = 0.005
best_reward_map = None

best_reward = 0

# file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
# file_name = "map_simple.txt"
# file_name = 'map_middle.txt'
# file_name = 'map_mid.txt'
file_name = "map_big.txt"
# file_name = "map_spiral.txt"

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape
print(num_of_columns)

num_of_steps_max = int(
    2.5 * (num_of_rows + num_of_columns)
)  # maximum number of steps in an episode
Q = np.zeros(
    [num_of_rows, num_of_columns, 4], dtype=float
)  # trained usability table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

for episode in range(number_of_episodes):
    random_row = np.random.randint(0, num_of_rows)
    # random_col = np.random.randint(0, num_of_columns - 1)
    random_col = 0
    state = np.array([random_row, random_col])
    # state = np.zeros(
    #     [2], dtype=int
    # )  # initial state here [1 1] but rather random due to exploration

    # print('initial state = ' + str(state) )
    the_end = False
    nr_pos = 0
    # reward_map_curr = reward_map
    while not the_end:
        nr_pos = nr_pos + 1
        # move number

        # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom):
        if np.random.random() < epsilon:
            # if random.uniform(0, 1) < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state[0], state[1], :])

        state_next, reward = sf.environment(state, action + 1, reward_map)

        # State-action usability modifcication:
        Q[state[0], state[1], action] = (1 - alpha) * Q[
            state[0], state[1], action
        ] + alpha * (reward + gamma * np.max(Q[state_next[0], state_next[1], :]))

        # print('state = ' + str(state) + ' action = ' + str(action) +  ' -> next state = ' + str(state_next) + ' reward = ' + str(reward))

        state = state_next
        # going to the next state

        # end of episode if maximum number of steps is reached or last column
        # is reached
        if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns - 1):
            the_end = True

        sum_of_rewards[episode] += reward
    if episode % 500 == 0:
        print(
            "episode = "
            + str(episode)
            + " average sum of rewards = "
            + str(np.mean(sum_of_rewards))
        )
        print("epsilon = ", epsilon)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    if sum_of_rewards[episode] > -0.5 and sum_of_rewards[episode] > best_reward:
        best_reward_map = reward_map
        best_reward = sum_of_rewards[episode]

print("average sum of rewards = " + str(np.mean(sum_of_rewards)))
# print(Q)

# x = np.arange(len(sum_of_rewards))
# plt.plot(x, sum_of_rewards)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.show()

mean_sum_of_rewards = sf.sailor_test(reward_map, Q, 1000)
sf.draw(reward_map, Q, mean_sum_of_rewards, file_name)

# if best_reward_map is not None:
#     best_mean_sum_of_rewards = sf.sailor_test(best_reward_map, Q, 1000)
#     sf.draw(reward_map, Q, best_mean_sum_of_rewards, "other_" + file_name)
