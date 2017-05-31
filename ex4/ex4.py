#!/usr/bin/env python3
"""
Fit various polynomials to a a dataset with train-validation-test split
and cross-validation.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt

# constants
OPEN_MODE = 'r'
FILE_DATA = 'X_poly.npy'
FILE_LABELS = 'Y_poly.npy'
D = 15
DEGREES = list(range(1, D+1))
K = 5


def nice_print(msg):
    if not msg:
        return
    delimiter = '=' * len(msg)
    print()
    print(delimiter)
    print(msg)
    print(delimiter)


def compute_loss(predictions, labels):
    return np.mean(np.power(predictions - labels, 2))


def load_data():
    X = np.load(FILE_DATA, mmap_mode=OPEN_MODE)
    Y = np.load(FILE_LABELS, mmap_mode=OPEN_MODE)
    return X, Y


def split_train_test(X, Y):
    m = X.shape[0]
    return np.array_split(np.arange(m), 3)


def fit_polynomial(X, Y, d):
    poly = np.poly1d(np.polyfit(X, Y, d))
    return poly, compute_loss(poly(X), Y)


def perform_validation(X, Y, H):
    loss = [np.inf] * len(H)
    for i, h in enumerate(H):
        loss[i] = compute_loss(h(X), Y)
    h_star = H[np.argmin(loss)]
    return h_star, loss


def concat(arr):
    if len(arr) > 0:
        return np.concatenate(arr)
    else:
        return np.array([])


def perform_kfold(X, Y, k):
    folds = np.array_split(np.arange(Y.shape[0]), k)
    h = [None] * k
    error = [np.inf] * D
    for j, d in enumerate(DEGREES):
        for i in range(k):
            train_1 = concat(folds[:i])
            train_2 = concat(folds[i+1:])
            train_idx = concat((train_1, train_2)).astype(int)
            test_idx = folds[i]
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]
            h[i], _ = fit_polynomial(X_train, Y_train, d)
            loss[i] = compute_loss(h[i](X_test), Y_test)
        error[j] = np.mean(loss)
    d_index = np.argmin(error)
    d_cv = DEGREES[d_index]
    return fit_polynomial(X, Y, d_cv)


def plot_fitted_data(X, Y, h_star, h_cv):
    plt.figure('Data Figure')
    X_sorted = np.sort(X)
    plt.scatter(X, Y, s=.5, label='data points')
    plt.title('All data')
    plt.text(0.62, 0.4, 'h_star:', fontsize=6)
    plt.text(0.7, 0.4, str(h_star), fontsize=6)
    plt.text(0.62, 0, 'h_cv:', fontsize=6)
    plt.text(0.7, 0, str(h_cv), fontsize=6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(X_sorted, h_star(X_sorted), c='orange', label='fitted with validation')
    plt.plot(X_sorted, h_cv(X_sorted), label='fitted with cross-validation')
    plt.legend()


def plot_losses(train_loss, valid_loss):
    plt.figure('Errors Figure')
    plt.title('Errors')
    plt.plot(DEGREES, train_loss, label='training')
    plt.plot(DEGREES, valid_loss, label='validation')
    plt.xlabel('polynomial degree (d)')
    plt.ylabel('mean error (MSE)')
    plt.legend()


if __name__ == "__main__":
    # read the data and divide to train, validation, test
    X, Y = load_data()


    train_idx, validation_idx, test_idx = split_train_test(X, Y)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_valid, Y_valid = X[validation_idx], Y[validation_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # get the best hypothesis for each polynomial degree
    hypotheses = [None] * D
    loss = [np.inf] * D
    for i, d in enumerate(DEGREES):
        hypotheses[i], loss[i] = fit_polynomial(X_train, Y_train, d)

    # perform validation to get the best hypothesis `h_d'
    h_star, valid_loss = perform_validation(
        X_valid, Y_valid, hypotheses)
    test_loss = compute_loss(h_star(X_test), Y_test)
    nice_print(f'h_star loss is: {test_loss}')
    X_cv = np.concatenate((X_train, X_valid))
    Y_cv = np.concatenate((Y_train, Y_valid))
    h_cv, _ = perform_kfold(X=X_cv, Y=Y_cv, k=K)

    # plot the data with the fitted polynomials to check if they're similar
    plot_fitted_data(X, Y, h_star, h_cv)

    # plot losses
    plot_losses(loss, valid_loss)

    plt.show()
