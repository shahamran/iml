import anytree

class Node:
    DELIM = '  '
    NEW_LINE = '\n'

    def __init__(self, name, parent=None):
        self._children = []
        self._name = name
        self._parent = parent
        self._height = 0
        self._nodes = 0

    @property
    def name(self):
        return self._name

    @name.getter
    def name(self, name):
        self._name = name

    @property
    def parent(self):
        return self.parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        self._parent.add_child(self)

    def add_child(self, child):
        self._children.append(child)
        child.parent = self
        self._nodes += child._nodes
        self._height = max(self._height, self._height + child._height)

    def add_children(self, children):
        if not hasattr(children, '__iter__'):
            self.add_child(children)
        for child in children:
            self.add_child(child)

    def __str__(self):
        output = str(self._name)
        if len(self._children) > 0:
            for child in self._children:
                output += Node.NEW_LINE + Node.DELIM + str(child)
        return output



class Node(anytree.Node):
    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        self.children = []

    def add_children(self, children):
        if not hasattr(children, '__iter__'):
            children = [children]
        for child in children:
            child.parent = self
            self.children.append(child)

    def __str__(self):
        return str(anytree.RenderTree(self))


def create_tree(label, j=None):
    if j is not None:
        trees = label
        tag = 'x_%d=?' % j
        root = Node(tag)
        root.add_children(trees)
    else:
        root = Node(str(label))
    return root
