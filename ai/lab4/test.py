import numpy as np


def create_random_array(num_of_rows, num_of_cols):
    random_row = np.random.randint(0, num_of_rows)
    random_col = np.random.randint(0, num_of_cols)
    return np.array([random_row, random_col])


num_of_rows = 2
num_of_cols = 1
random_array = create_random_array(num_of_rows, num_of_cols)
print("Random array:", random_array)
