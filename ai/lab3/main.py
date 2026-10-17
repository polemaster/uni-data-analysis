import numpy as np

# Example lists
mean_sums_of_rewards = [3, 1, 2, 5, 4]
Popul = ["a", "b", "c", "d", "e"]

# Get the sorted indices based on mean_sums_of_rewards
sorted_indices = np.argsort(mean_sums_of_rewards)
print(sorted_indices)

# Rearrange Popul based on the sorted indices
sorted_Popul = [Popul[i] for i in sorted_indices]

print("Original Popul:", Popul)
print("Sorted Popul based on mean_sums_of_rewards:", sorted_Popul)
