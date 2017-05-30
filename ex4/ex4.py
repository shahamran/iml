#!/usr/bin/env python3
"""
Fit various polynomials to a a dataset
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


class PolyClassifier:
    def psi(self, X):
        m = X.shape[0]
        d = self.d
        X_new = np.zeros((m, d+1))
        for i in range(d+1):
            X_new[:, i] = np.power(X, i)
        return X_new

    def fit(self, X, Y, d):
        self.d = d
        X_new = self.psi(X)
        # get the optimal polynomial coefficients
        self.w = least_squares(X_new, Y)

    def predict(self, X, Y=None):
        predictions = np.dot(self.psi(X), self.w)
        if Y is not None:
            return predictions, compute_loss(predictions, Y)
        else:
            return predictions

    def __str__(self, *a, **kw):
        return str(self.w)

    def __call__(self, X):
        return self.predict(X)

    def __getitem__(self, i):
        return self.w[i]


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


def least_squares(X, Y):
    A = np.transpose(X).dot(X)
    b = np.transpose(X).dot(Y)
    return np.linalg.pinv(A).dot(b)


def fit_polynomial(X, Y, d):
    poly = PolyClassifier()
    poly.fit(X, Y, d)
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
    h_cv, _ = perform_kfold(X_cv, Y_cv, k=K)

    # check if the hypothesis from the kfold cv
    # is the same as h_star
    nice_print(f'h_star equals h_cv: {np.all(np.equal(h_cv, h_star))}')
    nice_print(str(h_cv))
    nice_print(str(h_star))

    # plot losses
    plt.plot(DEGREES, loss)
    plt.plot(DEGREES, valid_loss)
    plt.show()
