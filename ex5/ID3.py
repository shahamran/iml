import numpy as np
import anytree
from anytree.dotexport import RenderTreeGraph
import copy


class Node(anytree.NodeMixin):
    id = 0

    def __init__(self, value, feature=None, parent=None):
        self._id = Node.id
        Node.id += 1
        self.value = None
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

    def __str__(self):
        return '%s (%d)' % (self.name, self._id)

    def replace(this, that, this_parent=None):
        # switch this node with that node, assigning this_parent to
        # this.parent
        that.parent = this.parent
        this.parent = this_parent
        this.value, that.value = that.value, this.value

    def copy(self):
        return copy.deepcopy(self)


def argmax(f, iterable):
    best_value = -np.inf
    best_index = 0
    for x in iterable:
        current_value = f(x)
        if current_value > best_value:
            best_value = current_value
            best_index = x
    return best_index

def argmin(f, iterable):
    return argmax(lambda x: -f(x), iterable)


class ID3Classifier:

    def __init__(self, label_values, feature_values, tree=None):
        self.tree = tree
        self.label_values = label_values
        self.feature_values = feature_values

    def __str__(self):
        return str(anytree.RenderTree(self.tree))

    def copy(self):
        return copy.deepcopy(self)

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

    def gain(self, S, i):
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
        j = argmax(lambda i: self.gain(S, i), features)
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

    def prune(self, error_bound):
        clf = self.copy()

        def compute_error(clf, original, alternative):
            # just compute the error if  no switch is needed
            if original == alternative:
                return error_bound(clf)
            # switch the original node with the alternative in clf
            old_parents = (original.parent, alternative.parent)
            Node.replace(original, alternative)

            if old_parents[0] is None:
                clf.tree = alternative
            # compute the error with the resulting tree

            output = error_bound(clf)
            # return to the original tree
            Node.replace(alternative, original, old_parents[-1])
            if old_parents[0] is None:
                clf.tree = original

            return output

        # go over all nodes in a bottom-up manner
        for node in anytree.PostOrderIter(clf.tree):
            # add all alternatives to a list
            alternatives = []
            for label in clf.label_values:
                alternatives.append(Node(label))
            for child in node.children:
                alternatives.append(child)
            alternatives.append(node)

            # find the alternative which minimizes the bound on the error
            best_alternative = argmin(lambda x: compute_error(clf, node, x),
                                      alternatives)
            if best_alternative != node:
                Node.replace(node, best_alternative)
                node = best_alternative
                if best_alternative.is_root:
                    clf.tree = best_alternative
        return clf



    def show(self):
        print(self)

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
            name = node.name
            if node.is_root:
                name += '\n' + 'Height(%d)' % node.height
            name += '\n' + 'id(%d)' % node._id
            return name

        RenderTreeGraph(self.tree,
                        nodeattrfunc=nodeattrfunc,
                        edgeattrfunc=edgeattrfunc,
                        nodenamefunc=nodenamefunc).to_picture(filename)
