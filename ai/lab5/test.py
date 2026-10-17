import numpy as np
import dec_tree_func as dt

file_name = 'car/car.txt'

num_of_random_features = 0

number_of_experiments = 10

# part of the examples for building a tree in percentage (0-100%) other examples can be used
# for pruning as validation subset:
part_for_constr = 80 # You can choose the proportions for tree construction and validation

print("\n\n\nBuilding tree for " + file_name)
# Loading data from file and dividing it into training and test sets:
# examples with odd numbers (1, 3, 5, ...) for training
# examples with even numbers for test

example_set = dt.load_data(file_name)

# Adding worthless information:
if num_of_random_features > 0:
    example_set = np.concatenate(
        (np.array(np.ceil(np.random.rand(len(example_set), num_of_random_features) * 12), dtype=int), example_set), axis=1)

[num_of_examples, num_of_columns] = example_set.shape
num_of_training_examples = num_of_examples // 2
num_of_test_examples = num_of_examples - num_of_training_examples
number_for_constr = int(np.ceil(num_of_training_examples * part_for_constr / 100))   # number of examples for tree construction

print("Number of examples: for tree construction = " + str(number_for_constr) + ", for validation = " +
      str(num_of_training_examples - number_for_constr) + ", for test = " + str(num_of_test_examples))

mean_test_error = 0
mean_test_error_prun = 0

for experiment in range(number_of_experiments):

    np.random.shuffle(example_set)  # Random permutation of example set

    train_vectors = example_set[0:num_of_training_examples, 0:num_of_columns-1]
    train_classes = example_set[0:num_of_training_examples, num_of_columns-1]
    test_vectors = example_set[num_of_training_examples:num_of_examples, 0:num_of_columns-1]
    test_classes = example_set[num_of_training_examples:num_of_examples, num_of_columns-1]

    tree = dt.build_tree(train_vectors, train_classes)

    if experiment == 0:
        dt.print_tree("tree_" + str(experiment) + ".txt", tree)

    dist = dt.distribution(train_vectors, train_classes, tree)
    d = dt.depth(tree)

    num_of_training_errors = dt.calc_error(train_vectors, train_classes, tree)
    num_of_test_errors = dt.calc_error(test_vectors, test_classes, tree)

    mean_test_error += num_of_test_errors / num_of_test_examples / number_of_experiments

    # Pruning - training example set division for tree construction and validation:
    train_vectors_constr = train_vectors[:number_for_constr]
    train_classes_constr = train_classes[:number_for_constr]
    train_vectors_valid = train_vectors[number_for_constr:]
    train_classes_valid = train_classes[number_for_constr:]

    tree_pruned = dt.build_tree(train_vectors_constr, train_classes_constr)
    num_of_rows, num_of_nodes = tree_pruned.shape

    # Here write code for pruning:

    def is_leaf(tree, node_index):
        arr = tree[..., node_index]
        return np.all(arr[:-1] == 0)

    def prune_tree(tree, valid_vectors, valid_classes, node):
        # If node is a leaf, return the tree
        if is_leaf(tree, node):
            return tree

        # Recursively prune left and right subtrees
        for child in tree[..., node][:-1]:
            tree[..., child]
        left_index = tree[2][0]
        right_index = tree[3][0]
        left_tree = tree[:, left_index:right_index]
        right_tree = tree[:, right_index:]

        # Prune left and right subtrees
        pruned_left_tree = prune_tree(left_tree, valid_vectors, valid_classes)
        pruned_right_tree = prune_tree(right_tree, valid_vectors, valid_classes)

        # Create new tree with pruned subtrees
        new_tree = np.copy(tree)
        new_tree[:, left_index:right_index] = pruned_left_tree
        new_tree[:, right_index:] = pruned_right_tree

        # Calculate errors before and after pruning
        error_before = dt.calc_error(valid_vectors, valid_classes, tree)
        error_after = dt.calc_error(valid_vectors, valid_classes, new_tree)

        # If pruning reduces the error, prune the node
        if error_after < error_before:
            new_tree[4][0] = 0  # Mark node as leaf
            new_tree[2][0] = 0  # Remove left subtree
            new_tree[3][0] = 0  # Remove right subtree

        return new_tree

    # Prune the tree using the validation set
    tree_pruned = prune_tree(tree_pruned, train_vectors_valid, train_classes_valid)

    if experiment == 0:
        dt.print_tree("tree_pruned_" + str(experiment) + ".txt", tree_pruned)

    num_of_training_errors_prun = dt.calc_error(train_vectors, train_classes, tree_pruned)
    num_of_test_errors_prun = dt.calc_error(test_vectors, test_classes, tree_pruned)

    print("Error: train = " + str(np.round(num_of_training_errors / num_of_training_examples, 4)) + " test = " +
          str(np.round(num_of_test_errors / num_of_test_examples, 4)) + "  after pruning: train = " +
          str(np.round(num_of_training_errors_prun / num_of_training_examples, 4)) + " test = " + str(np.round(num_of_test_errors_prun / num_of_test_examples, 4)))

    mean_test_error_prun += num_of_test_errors_prun / num_of_test_examples / number_of_experiments

print("Mean test error: before pruning = " + str(np.round(mean_test_error, 4)) + " after pruning = " + str(np.round(mean_test_error_prun, 4)) +
      "\nReduction of average test error = " + str(100 * (mean_test_error - mean_test_error_prun) / mean_test_error) + "%")

