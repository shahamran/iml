# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ID3

# constants
TRAIN_FILE = 'train.txt'
VALIDATION_FILE = 'validation.txt'
CSV_PARAMS = {'delim_whitespace': True, 'header': None}
FEATURE_VALUES = ['y', 'n', 'u']
TREES_DIR = 'trees'
if not os.path.exists(TREES_DIR):
    os.mkdir(TREES_DIR)
# for the random stuff to be consistent in submission
np.random.seed(25)

def TREE_FILE(question, name):
    return os.path.join(TREES_DIR, 'q%d_%s.svg' % (question, name))

def LOSS(predictions, true_labels):
    m = len(predictions)
    assert(len(predictions) == len(true_labels))
    return sum(predictions != true_labels) / m

# read and tidy up the data
train_data = pd.read_csv(TRAIN_FILE, **CSV_PARAMS)
validation_data = pd.read_csv(VALIDATION_FILE, **CSV_PARAMS)
# change the last column's name to "label"
for data in [train_data, validation_data]:
    data.rename(columns={(data.shape[1]-1): 'label'}, inplace=True)
labels = pd.unique(train_data.label)
features = train_data.columns[:-1]
d = len(features)
d_values = np.array(range(d+1))

# question 1 - 2

# initialize the ID3 class
T = ID3.ID3Classifier(labels, FEATURE_VALUES)
trees = [None] * (d+1)

m_train = train_data.shape[0]
m_valid = validation_data.shape[0]
train_error = [None] * (d+1)
valid_error = train_error.copy()
# run the algorithm for each tree height
for max_height in d_values:
    train_predictions = T.fit(train_data, max_height).predict(train_data)
    valid_predictions = T.predict(validation_data)
    train_error[max_height] = LOSS(train_predictions, train_data.label)
    valid_error[max_height] = LOSS(valid_predictions, validation_data.label)
    T.save_to_file(TREE_FILE(question=2, name=str(max_height)))
    trees[max_height] = T.copy()

# plot the errors graph
plt.plot(d_values, train_error, label='training')
plt.plot(d_values, valid_error, label='validation')
plt.xlabel('maximal tree height (d)')
plt.ylabel('error')
plt.legend()


# question 3
def error_bound(test_data):
    def bound(tree):
        predictions = tree.predict(test_data)
        true_labels = test_data.label
        return LOSS(predictions, true_labels)
    return bound

def generalization_error(tree):
    return np.abs(LOSS(tree.predict(train_data), train_data.label) -
                  LOSS(tree.predict(validation_data), validation_data.label))

pruned_tree = trees[-1].prune(error_bound(validation_data))
pruned_tree.save_to_file(TREE_FILE(question=3, name='pruned'))

# print the resulting errors
print('question 3')
print('----------')
print('generalization error of the un-pruned tree:')
print(generalization_error(trees[-1]))

print('generalization error of the pruned tree:')
print(generalization_error(pruned_tree))
print()

# TODO: question 4

def perform_kfold(clf, S, k):

    def concat(arr):
        """
        concatenates the given array of arrays if possible (if `arr' is not empty),
        otherwise doesn't do anything
        :param arr: an array of arrays to concatenate
        :return: a concatenated array if len(arr)>0, an empty array otherwise
        """
        if len(arr) > 0:
            return np.concatenate(arr)
        else:
            return np.array([])


    folds = np.array_split(np.arange(S.shape[0]), k)
    loss = [np.inf] * k
    error = np.ones((k, d+2)) * np.inf
    trees = [None] * (d+1)
    for i in range(k):
        # split to train & test sets
        train_1 = concat(folds[:i])
        train_2 = concat(folds[i + 1:])
        train_idx = concat((train_1, train_2))
        test_idx = folds[i]
        train_data = S.loc[train_idx, :].reset_index(drop=True)
        test_data = S.loc[test_idx, :].reset_index(drop=True)
        true_labels = test_data.label
        for j in range(d+1):
            trees[j] = clf.fit(train_data, max_height=j).copy()
            predictions = trees[j].predict(test_data)
            error[i, j] = LOSS(predictions, true_labels)
        pruned = trees[-1].prune(error_bound(test_data))
        predictions = pruned.predict(test_data)
        error[i, -1] = LOSS(predictions, true_labels)
    return error.mean(axis=0)

# perform 8-fold cross-validation
folds = 8
# unify train & validation and shuffle the array
unified = pd.concat([train_data, validation_data], ignore_index=True)
unified = unified.iloc[np.random.permutation(len(unified))].reset_index(
    drop=True)
error = perform_kfold(T, unified, folds)
plt.figure()
plt.plot(range(d+2), error)
plt.xticks(range(d+2), list(range(d+1)) + ['pruned'])
plt.xlabel('max_height / pruned')
plt.ylabel('cross-validation error')
plt.show()
