import numpy as np
import DTree


class ID3:
    def __init__(self, feature_values, label_values):
        self.root = None
        self.feature_values = feature_values
        self.label_values = label_values

    def entropy(self, S):
        if S is None or S.shape[0] == 0:
            return 0
        m = S.shape[0]
        result = 0
        for c in self.label_values:
            # compute the probability for label `c`
            p = sum(S.label == c) / m
            # if 0, entropy is 0
            result -= p * np.log(p) if p > 0 else 0
        return result

    def Gain(self, S, i, H=entropy):
        m = S.shape[0]
        c = 0
        for v in self.feature_values:
            Sv_indices = S.loc[:, i] == v
            Sv = S.loc[Sv_indices, :]
            c += Sv.shape[0] / m * H(self, Sv)
        return H(self, S) - c

    def ID3(self, S, A, depth):
        m = S.shape[0]
        max_c, max_c_val = 0, 0
        for c in self.label_values:
            Sc = sum(S.label == c)
            if Sc == m:
                return DTree.create_tree(c)
            if Sc > max_c_val:
                max_c_val = Sc
                max_c = c
        if A is None or len(A) == 0 or depth == 0:
            return DTree.create_tree(max_c)
        j = np.argmax([self.Gain(S, i) for i in A])
        new_A = np.delete(A, j)
        T = [None] * len(self.feature_values)
        for i, v in enumerate(self.feature_values):
            j_indices = S.loc[:, j] == v
            T[i] = self.ID3(S.loc[j_indices, :], new_A, depth-1)
        return DTree.create_tree(T, j)

    def train(self, S, depth=None):
        features = S.columns[:-1]
        if depth is None:
            depth = len(features)
        self.root = self.ID3(S, features, depth)
        return self

    def __str__(self):
        return str(self.root)
