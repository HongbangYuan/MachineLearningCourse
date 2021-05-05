# classification and regression tree
# Testing on the iris dataset
# The tree is simple but powerful
# Choose 60 samples as training data and 90 samples as testing data,we have the final accuracy:
# Final Result: 87/90   Accuracy:96.67%
# But the tree only have depth 3
# Or you can use it to do regression when solving Problem1,Page89
# Remember to set parameter 'classifier' to false to indicate a regression tree

import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from collections import OrderedDict


class Dataset():
    def __init__(self, num_samples):
        if num_samples < 0 or num_samples > 50:
            sys.exit("The choosing number can't be {}! It must be between [0,50]!".format(str(num_samples)))
        iris = load_iris()
        index = [[np.arange(num_samples) + 50 * i] for i in range(3)]  # choose 20 samples for each class
        train_data = np.concatenate([iris.data[index[0][0]], iris.data[index[1][0]], iris.data[index[2][0]]],
                                    axis=0)  # (150,4)
        train_label = np.concatenate([iris.target[index[0][0]], iris.target[index[1][0]], iris.target[index[2][0]]],
                                     axis=0)
        index = [[np.arange(50 - num_samples) + 50 * i + num_samples] for i in range(3)]
        test_data = np.concatenate([iris.data[index[0][0]], iris.data[index[1][0]], iris.data[index[2][0]]],
                                   axis=0)  # (150,4)
        test_label = np.concatenate([iris.target[index[0][0]], iris.target[index[1][0]], iris.target[index[2][0]]],
                                    axis=0)
        self.train_dataset = np.concatenate([train_data, np.expand_dims(train_label, axis=1)], axis=1)
        self.test_dataset = np.concatenate([test_data, np.expand_dims(test_label, axis=1)], axis=1)

    def getTrainData(self):
        return self.train_dataset

    def getTestData(self):
        return self.test_dataset

    def plotDataset(self, dataset):
        # plt.ion()
        fig = plt.figure()
        ax = Axes3D(fig)
        index = [0, 1, 3]  # [0-3] extract 3 feature to visulize
        target_names = ['setosa', 'versicolor', 'virginica']
        for label, color in zip([0, 1, 2], ['r', 'g', 'b']):
            x = dataset[np.where(dataset[:, -1] == label)[0], index[0]]
            y = dataset[np.where(dataset[:, -1] == label)[0], index[1]]
            z = dataset[np.where(dataset[:, -1] == label)[0], index[2]]
            ax.plot(x, y, z, color + '*', label='Hello!')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


def mse(y):
    return np.mean((y - np.mean(y)) ** 2)


def gini(y):
    y = np.array(y, dtype=np.int64)
    return 1 - np.sum(np.square(np.bincount(y) / len(y)))
    # return 1 - np.sum(np.square(np.array([np.bincount(y) / len(set(y))]) ))


def entropy(y):
    """
    Entropy of a label sequence
    """
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(self, left, right, feature, threshold):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        # self.n_classes = None
        self.parent = None
        self.l_r = None

    def set_parent(self, parent, l_r):
        self.parent = parent
        self.l_r = l_r


class Leaf:
    def __init__(self, label, samples):
        self.label = label
        self.samples = samples
        self.parent = None
        self.l_r = None

    def set_parent(self, parent, l_r):
        self.parent = parent
        self.l_r = l_r



class DecisionTree:
    def __init__(self, classifier=True, min_sample=None, n_features=None, criterion='gini'):
        if classifier and criterion not in ['gini']:
            raise ValueError("{} is invalid in a classification tree!".format(criterion))
        if not classifier and criterion not in ['mse']:
            raise ValueError("{} is invalid in a regression tree!".format(criterion))

        self.classifier = classifier  # True:classification  False:regression
        self.min_sample = min_sample if min_sample else 1
        self.n_features = n_features  # use the selected features to train a tree
        self.criterion = criterion
        self.root = None
        # self.depth = 0
        self.num_leaves = 0
        self.false_num_leaves = 0
        self.alpha_subtrees = None  # alpha:num_leaves
        self.false_root = None

    def fit(self, X, Y):
        """
        X : (n_samples,m_features)
        Y : (n_samples,)
        """
        self.n_classes = int(max(Y)) + 1 if self.classifier else None
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow(X, Y)

    def _copy(self,root):
        if isinstance(root,Leaf):
            return copy.deepcopy(root)
        left_child = self._copy(root.left)
        right_child = self._copy(root.right)
        curr_node = copy.deepcopy(root)
        curr_node.left = left_child
        curr_node.right = right_child
        left_child.set_parent(curr_node,'l')
        right_child.set_parent(curr_node,'r')
        return curr_node

    def cost_loss_pruning_path(self, X, Y):
        """
        X:(n_samples,m_features)
        Y:(n_samples,)
        """
        self.false_root = self._grow(X, Y)  # a new decision tree for pruning
        self.alpha_subtrees = [[np.inf, self._copy(self.false_root)]]
        self._prune(self.false_root)
        return self.alpha_subtrees

    def _prune(self, root):
        if not isinstance(root, Leaf):
            self._prune(root.left)
            self._prune(root.right)
        if root.parent is None:
            return
        if isinstance(root, Leaf):
            return

        if self._single_node_error(root) < self._train_error(root):
            print("That should be taken into consideration!")
            return

        g_t = self._gt(root)
        # print(# "Single Node Error C(t):", self._single_node_error(root),
        #       # "    Subtree Error C(Tt):", self._train_error(root),
        #       "    num_leaves:",self._num_leaves(root),
        #       "    g(t):{:.5f}".format(self._gt(root)))

        prev_alpha = self.alpha_subtrees[-1][0]
        alpha = min(prev_alpha,g_t)
        # if g_t < prev_alpha:
        # print("Hi!")
        # print("Before Merging:",self._num_leaves(self.false_root))
        self._merge(root)
        # print("After Merging:",self._num_leaves(self.false_root))

        self.alpha_subtrees.append([alpha, self._copy(self.false_root)])


    def _merge(self, root):
        if root.parent is None:
            print("Careful!Parameter root is already the root of the tree!")
            return
            # print()
            # raise ValueError("Parameter root is already the root of the tree! Please check your code carefully!")
        if isinstance(root, Leaf):
            raise TypeError("You can only merge only non-leaf nodes!")
        samples = self._samples_one_subtree(root)
        label = np.bincount(samples, minlength=self.n_classes) / len(samples) if self.classifier else np.mean(samples)

        if root.l_r == 'r':
            root.parent.right = Leaf(label, samples)
            root.parent.right.set_parent(root.parent, 'r')
        if root.l_r == 'l':
            root.parent.left = Leaf(label, samples)
            root.parent.left.set_parent(root.parent, 'l')

    def _gt(self, root):
        return (self._single_node_error(root) - self._train_error(root)) / (self._num_leaves(root) - 1)

    def _single_node_error(self, root):
        if self.criterion == 'gini':
            loss = gini
        if self.criterion == 'mse':
            loss = mse
        return loss(self._samples_one_subtree(root))

    def _samples_one_subtree(self, root):
        if not isinstance(root, Leaf):
            return np.concatenate([self._samples_one_subtree(root.left),
                                   self._samples_one_subtree(root.right)])
        return root.samples

    def _num_leaves(self, root):
        # the number of leaves of a subtree rooting at Node 'root'
        if not isinstance(root, Leaf):
            return self._num_leaves(root.left) + self._num_leaves(root.right)
        return 1

    def _num_samples(self,root):
        if not isinstance(root, Leaf):
            return self._num_samples(root.left) + self._num_samples(root.right)
        return len(root.samples)

    def _train_error(self, root):
        # the training error of a subtree rooting at Node 'root'
        # num_left = self._num_samples(root.left)
        # num_right = self._num_samples(root.right)
        # all = num_left + num_right
        # return self._single_node_error(root.left) / all * num_left + self._single_node_error(root.right) / all * num_right

        if not isinstance(root, Leaf):
            num_left = self._num_samples(root.left)
            num_right = self._num_samples(root.right)
            all = num_left + num_right
            return self._train_error(root.left) / all * num_left + self._train_error(root.right) / all * num_right
            # return self._train_error(root.left) / self._num_samples(root.left) + self._train_error(root.right) / self._num_samples(root.right)
        # for leaves
        if self.criterion == 'gini':
            loss = gini
        if self.criterion == 'mse':
            loss = mse
        return loss(root.samples)

    def predict(self, X,root=None):  # X:(n_samples,features,)
        root = self.root if root is None else root
        return np.array([self._traverse(X[idx], root, prob=False) for idx in range(X.shape[0])])

    def predict_prob(self, X,root=None):
        root = self.root if root is None else root
        return np.array([self._traverse(X[idx], root, prob=True) for idx in range(X.shape[0])])

    def _traverse(self, X, root, prob):  # X:(features,)
        if isinstance(root, Leaf):
            if self.classifier:
                return root.label if prob else np.argmax(root.label)
            return root.label
        if X[root.feature] <= root.threshold:
            return self._traverse(X, root.left, prob)
        else:
            return self._traverse(X, root.right, prob)

    def _grow(self, X, Y):
        # print(Y)
        if len(set(Y)) == 1:
            # self.num_leaves = self.num_leaves + 1
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
                return Leaf(prob, Y)
            else:
                return Leaf(Y[0], Y)
        if len(Y) < self.min_sample:
            # self.num_leaves = self.num_leaves + 1
            if self.classifier:
                return Leaf(np.bincount(Y, minlength=self.n_classes) / len(Y), Y)
            else:
                return Leaf(np.mean(Y), Y)

        # if self.depth > self.max_depth:
        #     if self.classifier:
        #         return Leaf(np.bincount(Y, minlength=self.n_classes) / len(Y))
        #     else:
        #         return Leaf(np.mean(Y))
        # self.depth = self.depth + 1
        feature_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)  # unique samples
        feature_idx, threshold = self._split(X, Y, feature_idxs)
        if threshold is None:
            print("Ops!Something went wrong!")
        left_idx = np.where(X[:, feature_idx] <= threshold)[0]
        right_idx = np.where(X[:, feature_idx] > threshold)[0]
        # if len(left_idx) == 0 or len(right_idx) == 0:
        # print(left_idx)
        # print(right_idx)
        # print(feature_idx)
        # print(threshold)

        left_child = self._grow(X[left_idx], Y[left_idx])
        right_child = self._grow(X[right_idx], Y[right_idx])
        curr_node = Node(left_child, right_child, feature_idx, threshold)
        left_child.set_parent(curr_node,'l')
        right_child.set_parent(curr_node,'r')
        return curr_node

    def _split(self, X, Y, feature_idxs):
        best_feature, best_threshold = None, None
        best_score = -np.inf
        for i in feature_idxs:
            feature = np.unique(X[:, i])
            if len(feature) == 1:
                continue
            thresholds = (feature[:-1] + feature[1:]) / 2
            scores = [self._score(Y, t, X[:, i]) for t in thresholds]
            # if len(scores) == 0:
            #     print(feature_idxs)
            #     print(feature)
            #     print("current i:",i)
            #     print(thresholds)
            #     print(scores)
            score, idx = np.max(scores), np.argmax(scores)
            if score > best_score:
                best_score = score
                best_threshold = thresholds[idx]
                best_feature = i
        if best_threshold is None:
            print("Ops!Something went wrong!")

        return best_feature, best_threshold

    def _score(self, Y, t, vals):

        if self.criterion == 'gini':
            loss = gini
        if self.criterion == 'mse':
            loss = mse
        init_score = loss(Y)
        left = np.where(vals <= t)[0]
        right = np.where(vals > t)[0]  # tuple -> ndarry:(num,)
        if len(left) == 0 or len(right) == 0:
            return 0
        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        final_score = n_l / n * e_l + n_r / n * e_r

        return init_score - final_score


if __name__ == '__main__':
    # Testing on the iris dataset
    print("Classification Tree On iris dataset:")
    data = Dataset(20)
    train_data = data.getTrainData()
    test_data = data.getTestData()
    # data.plotDataset(train_data)
    # data.plotDataset(test_data)
    dt = DecisionTree(classifier=True, criterion='gini')
    dt.fit(train_data[:, :-1], np.array(train_data[:, -1], dtype=np.int))
    prediction = dt.predict(test_data[:, :-1])
    test_label = np.array(test_data[:, -1], dtype=np.int)
    right = np.sum(prediction == test_label)
    all = len(prediction)
    print("Final Result: {}/{}   Accuracy:{:.2f}%".format(right, all, right / all * 100))
    # print("Testing the merging function:")
    # dt.cost_loss_pruning_path(train_data[:, :-1], np.array(train_data[:, -1], dtype=np.int))
    # dt._merge(dt.root.right)
    # prediction = dt.predict(test_data[:, :-1])
    # test_label = np.array(test_data[:, -1], dtype=np.int)
    # right = np.sum(prediction == test_label)
    # all = len(prediction)
    # print("Final Result: {}/{}   Accuracy:{:.2f}%".format(right, all, right / all * 100))

    train_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           [4.5, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.7, 9.0]])
    X = train_data[0].reshape(10, 1)
    Y = train_data[1].reshape(10, )
    dt = DecisionTree(classifier=False,criterion='mse')
    dt.fit(X, Y)
    print("Regression Tree On Problem 5.2:")
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x = np.expand_dims(x, 1)
    y = dt.predict(x)
    print(x.T, "\n", y)
# dt._score(Y,t,vals)
# print(dt)


# def _false_grow(self, X, Y):
#     # print(Y)
#     if len(set(Y)) == 1:
#         self.false_num_leaves = self.false_num_leaves + 1
#         if self.classifier:
#             prob = np.zeros(self.n_classes)
#             prob[Y[0]] = 1.0
#             return Leaf(prob, Y)
#         else:
#             return Leaf(Y[0], Y)
#     if len(Y) < self.min_sample:
#         self.false_num_leaves = self.false_num_leaves + 1
#         if self.classifier:
#             return Leaf(np.bincount(Y, minlength=self.n_classes) / len(Y), Y)
#         else:
#             return Leaf(np.mean(Y), Y)
#
#     feature_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)  # unique samples
#     feature_idx, threshold = self._split(X, Y, feature_idxs)
#     if threshold is None:
#         print("Ops!Something went wrong!")
#     left_idx = np.where(X[:, feature_idx] <= threshold)[0]
#     right_idx = np.where(X[:, feature_idx] > threshold)[0]
#     # if len(left_idx) == 0 or len(right_idx) == 0:
#     # print(left_idx)
#     # print(right_idx)
#     # print(feature_idx)
#     # print(threshold)
#
#     left_child = self._false_grow(X[left_idx], Y[left_idx])
#     right_child = self._false_grow(X[right_idx], Y[right_idx])
#     return Node(left_child, right_child, feature_idx, threshold)
