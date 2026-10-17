import numpy as np
import dec_tree_func as dt



def is_leaf(tree, node_index):
    arr = tree[..., node_index]
    return np.all(arr[:-1] == 0)

def entropy(dist):
    total = np.sum(dist)
    if total == 0:
        return 0

    dist_normalized = dist / total
    dist_nonzero = dist_normalized[dist_normalized > 0]
    entropy_value = -np.sum(dist_nonzero * np.log2(dist_nonzero))
    return entropy_value

def prune_tree(tree, dist, depths, entropy_thr=1.2):
    pruned_tree = tree.copy()
    num_of_rows, num_of_nodes = tree.shape

    to_prune = 400
    for i in range(num_of_nodes-1, -1, -1):
        if not is_leaf(pruned_tree, i):
            if depths[i] > 2 and sum(dist[..., i]) < 5:
                to_prune -= 1
                if to_prune == 0:
                    break

                # turn the node into a leaf
                pruned_tree[..., i] = 0
                pruned_tree[num_of_rows-1,i] = np.argmax(dist[...,i]) + 1
            if depths[i] > 3 and entropy(dist[...,i]) >= entropy_thr:
            # if max(dist[..., i]) / sum(dist[..., i]) < 0.85:
                to_prune -= 1
                if to_prune == 0:
                    break
                # turn the node into a leaf
                pruned_tree[..., i] = 0
                pruned_tree[num_of_rows-1,i] = np.argmax(dist[...,i]) + 1
    return pruned_tree


def main():
    #file_name = "lenses\lenses.txt"
    file_name = 'car/car.txt'
    #file_name = 'tic/tic.txt'
    # file_name = 'letters1.txt'
    #file_name = 'letters2.txt'
    #file_name = 'letters3.txt'
    #file_name = 'letters5.txt'

    num_of_random_features = 0

    number_of_experiments = 10


    # part of the examples for building a tree in percentage (0-100%) other axamples can be used
    # for pruning as validation subset:
    part_for_constr = 100 # (..... you can choose the proportions for tree construction and validation )



    print("\n\n\nbuilding tree for " + file_name)
    # loading data from file and dividing it into training and test sets:
    # examples with odd numbers (1,3,5,...) for training
    # examples with even numbers for test

    example_set = dt.load_data(file_name)
    # Adding worthless information:
    if num_of_random_features > 0:
        example_set = np.concatenate(\
            (np.array(np.ceil(np.random.rand(len(example_set),num_of_random_features)*12),dtype=int),example_set), axis = 1)
    #print(" examples = \n" + str(examples))

    [num_of_examples, num_of_columns] = example_set.shape
    num_of_training_examples = num_of_examples//2
    num_of_test_examples = num_of_examples - num_of_training_examples
    number_for_constr = int(np.ceil(num_of_training_examples*part_for_constr/100))   # number of examples for tree construction
    print("number of examples: for tree construction = "+ str(number_for_constr) + ", for validation = "+ \
          str(num_of_training_examples-number_for_constr) + ", for test = " + str(num_of_test_examples))
    #print("num_of_training_examples = " + str(num_of_training_examples)+ " num_of_test_examples = "+str(num_of_test_examples))

    mean_test_error = 0
    mean_test_error_prun = 0


    for experiment in range(number_of_experiments):

        np.random.shuffle(example_set)       # random permutation of example set
        #print(" examples after shuffle = \n" + str(examples))

        train_vectors = example_set[0:num_of_training_examples,0:num_of_columns-1]
        train_classes = example_set[0:num_of_training_examples,num_of_columns-1]
        test_vectors = example_set[num_of_training_examples:num_of_examples,0:num_of_columns-1]
        test_classes = example_set[num_of_training_examples:num_of_examples,num_of_columns-1]

        #print("train_vectors = \n" + str(train_vectors))
        #print("train_classes = \n" + str(train_classes))
        #print("test_vectors = \n" + str(test_vectors))
        #print("test_classes = \n" + str(test_classes))

        tree = dt.build_tree(train_vectors, train_classes)

        #print("final tree = \n" + str(tree))
        if experiment == 0:
            dt.print_tree("tree_"+str(experiment)+".txt",tree)

        dist = dt.distribution(train_vectors,train_classes,tree)
        #print("distribution = \n" + str(dist))
        d =  dt.depth(tree)
        #print("depth = " + str(d))

        num_of_training_errors = dt.calc_error(train_vectors,train_classes,tree)
        num_of_test_errors = dt.calc_error(test_vectors,test_classes,tree)

        mean_test_error += num_of_test_errors/num_of_test_examples/number_of_experiments


        # pruning - training example set division for tree construction and validation:
        train_vectors_constr = train_vectors[:number_for_constr]
        train_classes_constr = train_classes[:number_for_constr]
        train_vectors_valid =  train_vectors[number_for_constr:]
        train_classes_valid =  train_classes[number_for_constr:]


        tree_pruned = dt.build_tree(train_vectors_constr, train_classes_constr)
        num_of_rows, num_of_nodes = tree_pruned.shape

        # create this for getting columns
        # tree_pruned_tr = np.transpose(tree_pruned)
        # dist_tr = np.transpose(dist)

        # max_depth = 5
        # # pruning:
        # removed_nodes = []
        # for node in range(num_of_nodes):
        #     # if any node is below max_depth then set all its values to 0
        #     if d[node] > max_depth:
        #         removed_nodes.append(node+1)
        #         for row in range(num_of_rows):
        #             tree_pruned[row][node] = 0
        # for node in range(num_of_nodes):
        #     # if node contains any of the removed nodes and it is not removed, make it a leaf
        #     # if tree_pruned[:-1][node] contains any of removed and node not in removed_nodes:
        #     col = tree_pruned_tr[:][node][:-1]
        #     if np.any(np.isin(col, removed_nodes)) and node not in removed_nodes:
        #         for row in range(num_of_rows):
        #             tree_pruned[row][node] = 0
        #         tree_pruned[-1][node] = np.argmax(dist[...,node]) + 1


        # entropies = [x / 20.0 for x in range(20, 31)]
        # min_err = 0
        # pruned_tree = dt.build_tree(train_vectors_constr, train_classes_constr)

        # dist = dt.distribution(train_vectors_constr,train_classes_constr,pruned_tree)
        # d =  dt.depth(pruned_tree)

        # final_tree = pruned_tree
        # final_ent = 0
        # for ent in entropies:
        #     tree_pruned = prune_by_entropy(pruned_tree, dist, d, ent)
        #     error = dt.calc_error(test_vectors,test_classes,tree_pruned)
        #     if experiment == 3:
        #         print(ent, error)
        #     if min_err == 0 or error < min_err:
        #         min_err = error
        #         final_tree = tree_pruned
        #         final_ent = ent
        # tree_pruned = final_tree

        # print('final_ent', final_ent)

        tree_pruned = prune_tree(tree, dist, d)

        if experiment == 0:
            dt.print_tree("tree_pruned_"+str(experiment)+".txt",tree_pruned)
        num_of_training_errors_prun = dt.calc_error(train_vectors,train_classes,tree_pruned)
        # num_of_training_errors_prun = dt.calc_error(train_vectors_constr,train_classes_constr,tree_pruned)
        num_of_test_errors_prun = dt.calc_error(test_vectors,test_classes,tree_pruned)

        # print("error: train = " + str(np.round(num_of_training_errors/num_of_training_examples,4)) + " test = " +
        # str(np.round(num_of_test_errors/num_of_test_examples,4)) + "  after pruning: train = " +
        # str(np.round(num_of_training_errors_prun/num_of_training_examples,4)) + " test = " + str(np.round(num_of_test_errors_prun/num_of_test_examples,4)))

        mean_test_error_prun += num_of_test_errors_prun/num_of_test_examples/number_of_experiments

    print("mean test error: before pruning = " + str(np.round(mean_test_error,4)) + " after pruning = " + str(np.round(mean_test_error_prun,4)) +
    "\nreduction of average test error = " + str(100*(mean_test_error-mean_test_error_prun)/mean_test_error) + "%")

if __name__ == "__main__":
    main()
