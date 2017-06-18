import numpy as np
from anytree import NodeMixin, RenderTree
from anytree.dotexport import RenderTreeGraph
import pptree


class Node(NodeMixin):
    id = 0
    def __init__(self, value, feature=None, parent=None):
        self._id = Node.id
        Node.id += 1
        if feature is None:
            self.value = value
            self.name = value
            self._i = None
        else:
            self._i = feature
            trees = value
            for a in trees:
                trees[a].parent = self
                trees[a].value = a
            self.name = 'x_%d=?' % feature
        self.parent = parent


def argmax(f, iterable):
    best_value = -np.inf
    best_index = 0
    for x in iterable:
        current_value = f(x)
        if current_value > best_value:
            best_value = current_value
            best_index = x
    return best_index


class ID3Classifier:

    def __init__(self, label_values, feature_values):
        self.tree = None
        self.label_values = label_values
        self.feature_values = feature_values

    def entropy(self, S):
        if S is None or S.shape[0] == 0:
            return 0
        m = S.shape[0]
        result = 0
        for c in self.label_values:
            # compute the probability for label `c`
            p = sum(S.label == c) / m
            # if p=0 then entropy is 0
            result -= p * np.log(p) if p > 0 else 0
        return result

    def Gain(self, S, i):
        H = self.entropy
        m = S.shape[0]
        c = 0
        for v in self.feature_values:
            # Sv_indices is the indices of samples in which the feature `i`
            # gets the value `v`
            Sv_indices = S.loc[:, i] == v
            Sv = S.loc[Sv_indices, :]
            c += Sv.shape[0] / m * H(Sv)
        return H(S) - c

    def _helper(self, S, features, max_height):
        sample_size = S.shape[0]
        best_label, best_label_score = 0, 0
        # check if all labels get the same value. if not, compute the label
        # which has the maximal number of examples
        for label_value in self.label_values:
            sub_sample_size = sum(S.label == label_value)
            if sub_sample_size == sample_size:
                return Node(label_value)
            if sub_sample_size > best_label_score:
                best_label_score = sub_sample_size
                best_label = label_value
        if features is None or len(features) == 0 or max_height == 0:
            return Node(best_label)

        # take the feature that maximizes the gain and remove it from the
        # feature set
        j = argmax(lambda i: self.Gain(S, i), features)
        new_features = features - {j}
        temp = dict()
        # for every possible feature value, create a sub-tree
        for feature_value in self.feature_values:
            j_indices = S.loc[:, j] == feature_value
            temp[feature_value] = self._helper(S.loc[j_indices, :],
                                               new_features,
                                               max_height-1)
        return Node(temp, j)

    def fit(self, S, max_height=None):
        Node.id = 0
        features = set(S.columns[:-1])
        if max_height is None:
            max_height = len(features)
        self.tree = self._helper(S, features, max_height)
        return self

    def _predict_one(self, s):
        temp = self.tree
        while temp._i is not None:
            for child in temp.children:
                if child.value == s[temp._i]:
                    temp = child
                    break
        return temp.name

    def predict(self, S):
        m = S.shape
        if len(m) == 1:
            return self._predict_one(S)
        else:
            m = m[0]
            predictions = [None] * m
            for i in range(m):
                predictions[i] = self._predict_one(S.iloc[i])
            return predictions

    def show(self):
        pptree.print_tree(self.tree)

    def save_to_file(self, filename):

        def edgeattrfunc(node, child):
            return 'label=' + str(child.value)

        def nodeattrfunc(node):
            attr = 'shape='
            if node.is_leaf:
                attr += 'ellipse'
            else:
                attr += 'rectangle'
            return attr

        def nodenamefunc(node):
            if node.is_leaf:
                name = node.name
                #name = num_to_label[node.name]
            else:
                name = node.name
            if node.is_root:
                name += '\n' + 'Height(%d)' % node.height
            name += '\n' + 'id(%d)' % node._id
            return name

        RenderTreeGraph(self.tree,
                        nodeattrfunc=nodeattrfunc,
                        edgeattrfunc=edgeattrfunc,
                        nodenamefunc=nodenamefunc).to_picture(filename)

    def __str__(self):
        return str(RenderTree(self.tree))
