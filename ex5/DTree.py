class Node:
    DELIM = '  '
    NEW_LINE = '\n'

    def __init__(self, data, parent=None):
        self.name = data
        self.feature_value = None
        self._children = []
        self._feature = None
        self._parent = parent
        self._height = 0
        self._nodes = 1
        self._depth = 0

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value
        for child in self._children:
            child.depth = value + 1

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        if self._parent is not None:
            self._parent.height = max(self._parent.height, value + 1)

    def add_child(self, child, feature_value):
        if self._feature is None:
            self._feature = self.name
            self.name = 'x_%d=?' % self.name
        self._children.append(child)
        child.feature_value = feature_value
        child._parent = self
        child.depth = self.depth + 1
        self._nodes += child._nodes
        self.height = max(self.height, child.height + 1)

    def add_children(self, children):
        for (child, feature_value) in children:
            self.add_child(child, feature_value)

    def __getitem__(self, sample):
        if self._feature is None:
            return self.name
        elif len(sample) > self._feature:
            feature_value = sample[self._feature]
            for child in self._children:
                if child.feature_value == feature_value:
                    return child[sample]

    def __str__(self):
        output = Node.DELIM*self._depth + str(self.name)
        if len(self._children) > 0:
            for child in self._children:
                output += Node.NEW_LINE + str(child)
        return output


def create_tree(label, j=None):
    if j is not None:
        trees = label
        root = Node(j)
        root.add_children(trees)
    else:
        root = Node(label)
    return root
