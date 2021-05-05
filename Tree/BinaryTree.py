
class Node:
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.lchild = None
        self.rchild = None

    def size(self):  # size of a tree rooted in this node
        s = 1
        if self.lchild:
            s += self.lchild.size()
        if self.rchild:
            s += self.rchild.size()
        return s


class BinaryTree:
    def __init__(self):
        self.root = None
