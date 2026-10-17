import numpy as np
import dec_tree_func as dt

from build_xxx import prune_by_entropy, is_leaf, entropy

file_name = "lenses/lenses.txt"
#file_name = "car/car.txt"

# print("\n\n\nbuilding tree for " + file_name)
# loading data from file and dividing it into training and test sets:
# examples with odd numbers (1,3,5,...) for training
# examples with even numbers for test
train_vectors, train_classes, test_vectors, test_classes = dt.load_data_odd_even(file_name)

num_of_train_examples = train_classes.__len__()
num_of_test_examples = test_classes.__len__()

tree = dt.build_tree(train_vectors, train_classes)
dist = dt.distribution(train_vectors,train_classes,tree)
d =  dt.depth(tree)


num_of_training_errors = dt.calc_error(train_vectors,train_classes,tree)
num_of_test_errors = dt.calc_error(test_vectors,test_classes,tree)

print("training error = " + str(num_of_training_errors/num_of_train_examples) + " test error = " +
str(num_of_test_errors/num_of_test_examples))

print("tree:\n" + str(tree))
print("dist:\n" + str(dist))
print("depth:" + str(d))
# dt.print_tree('lenses_tree.txt',tree)
# pruning:

tree_pruned = np.copy(tree)
num_of_rows, num_of_nodes = tree_pruned.shape

max_depth = 2
removed_nodes = []
for node in range(num_of_nodes):
    # if any node is below max_depth then set all its values to 0
    if d[node] > max_depth:
        removed_nodes.append(node+1)
        for row in range(num_of_rows):
            tree_pruned[row][node] = 0
tree_pruned_tr = np.transpose(tree_pruned)
dist_tr = np.transpose(dist)
for node in range(num_of_nodes):
    # if node contains any of the removed nodes and it is not removed, make it a leaf
    # if tree_pruned[:-1][node] contains any of removed and node not in removed_nodes:
    col = tree_pruned_tr[:][node][:-1]
    if np.any(np.isin(col, removed_nodes)) and node not in removed_nodes:
        for row in range(num_of_rows):
            tree_pruned[row][node] = 0
        tree_pruned[-1][node] = np.argmax(dist[...,node]) + 1

pruned_tree = tree_pruned
pruned_tree = prune_by_entropy(tree, dist, d)
for i in range(num_of_nodes):
    print(i, is_leaf(pruned_tree, i))
dt.print_tree('lenses_tree_pruned.txt',pruned_tree)
print('pruned tree:\n' + str(pruned_tree))

num_of_training_errors = dt.calc_error(train_vectors,train_classes,pruned_tree)
num_of_test_errors = dt.calc_error(test_vectors,test_classes,pruned_tree)



print("after pruning:\ntraining error = " + str(num_of_training_errors/num_of_train_examples) + " test error = " +
str(num_of_test_errors/num_of_test_examples))




