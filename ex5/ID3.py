import numpy as np
import DTree


class ID3:
    """
    represents the ID3 classifier and implements its important methods
    """

    def __init__(self, feature_values, label_values):
        """
        creates a new classifier
        :param feature_values: (array_like) possible feature values range
        :param label_values: (array_like) possible label values range
        """
        self.root = None
        self.feature_values = feature_values
        self.label_values = label_values

    def entropy(label_values, S):
        """
        computes the entropy of a given set
        :param label_values: (array_like) possible values for the labels
        :param S: a set of examples
        :return: sum over i of -p_i*log(p_i)
        """
        if S is None or S.shape[0] == 0:
            return 0
        m = S.shape[0]
        result = 0
        for c in label_values:
            # compute the probability for label `c`
            p = sum(S.label == c) / m
            # if p=0 then entropy is 0
            result -= p * np.log(p) if p > 0 else 0
        return result

    def Gain(self, S, i, H=entropy):
        """
        computes the gain of the feature i and example set S.
        by default, computes the information gain
        :param S: (pandas dataframe) the examples set
        :param i: (non-negative int) the feature's index
        :param H: (callable) how to compute the gain
        :return: the gain of feature i w.r.t S
        """
        m = S.shape[0]
        c = 0
        for v in self.feature_values:
            # Sv_indices is the indices of samples in which the feature `i`
            # gets the value `v`
            Sv_indices = S.loc[:, i] == v
            Sv = S.loc[Sv_indices, :]
            c += Sv.shape[0] / m * H(self.label_values, Sv)
        return H(self.label_values, S) - c

    def ID3(self, S, A, height):
        """
        implementation of the ID3 (recursive) algorithm
        :param S: (dataframe) the examples set
        :param A: (set) the features set
        :param height: (int) maximal height of the resulting tree
        :return: a decision tree of height <= `height` as chosen by ID3 with
                 information gain as a gain measurement
        """
        m = S.shape[0]
        max_c, max_c_val = 0, 0
        # check if all labels get the same value. if not, compute the label
        # which has the maximal number of examples
        for c in self.label_values:
            Sc = sum(S.label == c)
            if Sc == m:
                return DTree.create_tree(c)
            if Sc > max_c_val:
                max_c_val = Sc
                max_c = c
        if A is None or len(A) == 0 or height == 0:
            return DTree.create_tree(max_c)
        # take the feature that maximizes the gain and remove it from the
        # feature set (denoted by new_A)
        j = np.argmax([self.Gain(S, i) for i in A])
        new_A = A - {j}
        trees = [None] * len(self.feature_values)
        # for every possible feature value, create a sub-tree
        for i, v in enumerate(self.feature_values):
            j_indices = S.loc[:, j] == v
            trees[i] = (self.ID3(S.loc[j_indices, :], new_A, height - 1), v)
        return DTree.create_tree(trees, j)

    def train(self, S, height=None):
        """
        trains the algorithm on a given set
        :param S: data set where the last column is the labels' column
        :param height: the maximal tree's height (optional) - default value
                       is the number of features (full tree)
        :return: a decision tree fitted to the given data-set using ID3
        """
        features = set(S.columns[:-1])
        if height is None:
            height = len(features)
        self.root = self.ID3(S, features, height)
        return self

    def predict(self, S):
        """
        predict the labels of the given data set according to the fitted tree
        :param S: a data set to predict on
        :return: an array of predictions
        """
        m = S.shape
        # if we're dealing with one sample predict and return
        if len(m) == 1:
            return self.root[S]
        # otherwise, m is the number of samples
        m = m[0]
        predictions = [None] * m
        # get a prediction for each sample (see DTree's implementation)
        for i in range(m):
            predictions[i] = self.root[S.iloc[i]]
        return predictions

    def __str__(self):
        """
        pretty print
        """
        return str(self.root)
