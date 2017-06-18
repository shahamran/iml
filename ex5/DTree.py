from anytree import NodeMixin, RenderTree

class Node(NodeMixin):
    def __init__(self, name, has_feature=False, parent=None):
        self.feature = None
        self.answer = None
        self.parent = parent
        if has_feature:
            self.name = 'x_%d=?' % name
            self.feature = name
        else:
            self.name = name

    def __getitem__(self, sample):
        if self.feature is None:
            return self.name
        elif len(sample) > self.feature:
            feature_value = sample[self.feature]
            for child in self.children:
                if child.answer == feature_value:
                    return child[sample]

    def __str__(self):
        return str(RenderTree(self).by_attr())


def create_tree(label, j=None):
    if j is not None:
        trees = label
        root = Node(j, has_feature=True)
        for (tree, answer) in trees:
            tree.answer = answer
            tree.parent = root
    else:
        root = Node(label)
    return root
