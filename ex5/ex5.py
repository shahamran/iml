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

def TREE_FILE(question, name):
    return os.path.join(TREES_DIR, 'q%d_%s.svg' % (question, name))

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
    train_error[max_height] = sum(train_predictions !=
                                  train_data.label) / m_train
    valid_error[max_height] = sum(valid_predictions !=
                                  validation_data.label) / m_valid
    T.save_to_file(TREE_FILE(question=2, name=str(max_height)))
    trees[max_height] = T.copy()

# plot the errors graph
plt.plot(d_values, train_error, label='training')
plt.plot(d_values, valid_error, label='validation')
plt.xlabel('maximal tree height (d)')
plt.ylabel('error')
plt.legend()


# TODO: question 3


# TODO: question 4
folds = 8
# unify train & validation and shuffle the array
unified = pd.concat([train_data, validation_data], ignore_index=True)
unified = unified.iloc[np.random.permutation(len(unified))].reset_index()

def perform_kfold(S, k):
    """
    performs k-fold cross-validation over the given data set
    :param S: data set
    :param k: number of folds
    :return:
    """
    folds = np.array_split(np.arange(Y.shape[0]), k)
    loss = [np.inf] * k
    error = [np.inf] * D
    # iterate the parameters space (d)
    for j, d in enumerate(DEGREES):
        # train & test for each fold
        for i in range(k):
            # split to train & test sets
            train_1 = concat(folds[:i])
            train_2 = concat(folds[i+1:])
            train_idx = concat((train_1, train_2)).astype(int)
            test_idx = folds[i]
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]
            # fit a polynomial to the training set and compute the loss on
            # the test set
            poly, _ = fit_polynomial(X_train, Y_train, d)
            loss[i] = compute_loss(poly(X_test), Y_test)
        # average the "test" losses on all folds
        error[j] = np.mean(loss)
    # get the polynomial degree which minimizes the error
    d_index = np.argmin(error)
    d_cv = DEGREES[d_index]
    # fit a polynomial with the 'best' degree on the whole data set
    return fit_polynomial(X, Y, d_cv)

plt.show()
