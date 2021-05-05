# Constructing the kd-tree and use it to do knn algorithm
# Key point:recursion function in the construction of the tree!
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# %% constructing the kd-tree
class Node:
    def __init__(self, sample, dim, left, right):
        self.sample = sample  # the feature vector of one sample on this node
        self.dim = dim  # the dimension to be compared,in the example,we have four dimensions
        self.left = left  # left child node
        self.right = right  # right child node


class kdTree:
    def __init__(self, dataset):
        self.dataset = dataset
        self.total_dim = dataset.shape[1] - 1
        self.root = self.createNode(0, dataset)
        pass

    def createNode(self, dims, dataset):  # which dimension to compare  dataset:(samples,features+labels)
        if dataset.size == 0:  # the end of the recursion
            return None

        dataset = dataset[dataset[:, dims].argsort(), :]
        median_pos = len(dataset) // 2
        median_sample = dataset[median_pos, :]
        dims_next = (dims + 1) % self.total_dim

        return Node(
            median_sample,
            dims,
            self.createNode(dims_next, dataset[:median_pos, :]),  # left < root
            self.createNode(dims_next, dataset[median_pos + 1:, :]) # root < right
        )

    # def knn(self,sample): # sample:(features,) (4,) in iris dataset


    def nearest(self,n1,n2,sample):
        if n1 is None:
            return n2,np.linalg.norm(n2.sample - sample,ord=2)
        if n2 is None:
            return n1,np.linalg.norm(n1.sample - sample,ord=2)
        d1,d2 = np.linalg.norm(n1.sample - sample,ord=2),np.linalg.norm(n2.sample - sample,ord=2)
        if  d1 < d2:
            return n1,d1
        else:
            return n2,d2

    def nearestNeighbor(self,root,sample,dim):
        if root is None:
            return root
        if sample[dim] >= root.sample[dim]:
            next = root.right
            other = root.left
        if sample[dim] < root.sample[dim]:
            next = root.left
            other = root.right

        tmp = self.nearestNeighbor(next,sample,(dim+1) % self.total_dim)
        print("type(next):",type(next))
        best,radius = self.nearest(tmp,root,sample)
        dist = sample[dim] - root.sample[dim]
        if(radius >= dist * dist):
            tmp = self.nearestNeighbor(other,sample,(dim+1) % self.total_dim)
            best,_ = self.nearest(tmp,best,sample)
        return best

    def dive(self,dim,sample,root):
        if root is None:
            return root
        # print(root.sample)
        if sample[dim] >= root.sample[dim]:
            if root.right is None:
                return root
            else:
                return self.dive((dim+1) % self.total_dim,sample,root.right)
        if sample[dim] < root.sample[dim]:
            if root.left is None:
                return root
            else:
                return self.dive((dim+1) % self.total_dim,sample,root.left)



    def preorder(self,root):
        if root is None:
            return
        print(root.sample)
        self.preorder(root.left)
        self.preorder(root.right)

    def inorder(self,root):
        if root is None:
            return
        self.inorder(root.left)
        print(root.sample)
        self.inorder(root.right)


def plot_dataset(dataset):  # dataset:(samples,features+labels)
    # plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)
    index = [0,1,3] # [0-3] extract 3 feature to visulize
    target_names = ['setosa', 'versicolor', 'virginica']
    for label,color in zip([0,1,2],['r','g','b']):
        x = dataset[np.where(dataset[:,-1] == label)[0],index[0]]
        y = dataset[np.where(dataset[:,-1] == label)[0],index[1]]
        z = dataset[np.where(dataset[:,-1] == label)[0],index[2]]
        ax.plot(x,y,z,color+'*',label='Hello!')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # plt.show()
    # plt.ioff()



# %% constructing the iris dataset   (samples,features) (samples,labels)
print("Preparing the iris dataset...")
num_samples = 2  # [0,50]
print("Every {} samples of each class are extracted for the trainig process...".format(20))
iris = load_iris()
# index = np.arange(20)
index = [[np.arange(num_samples) + 50 * i] for i in range(3)]  # choose 20 samples for each class
train_data = np.concatenate([iris.data[index[0][0]], iris.data[index[1][0]], iris.data[index[2][0]]], axis=0)  # (150,4)
train_label = np.concatenate([iris.target[index[0][0]], iris.target[index[1][0]], iris.target[index[2][0]]], axis=0)
dataset = np.concatenate([train_data, np.expand_dims(train_label, axis=1)], axis=1)

# %% test the kd tree
tree = kdTree(dataset)
print("Constructing the kd-tree model of the given dataset...")
# plot_dataset(dataset)
# tree.inorder(tree.root)
print("Preorder traversal:")
tree.preorder(tree.root)
print("Inorder traversal:")
tree.inorder(tree.root)
sample = dataset[1]
sample[0] = sample[0] + 0.001
node = tree.nearestNeighbor(tree.root,sample,0)
print("Final result:")
print("Selected sample:",node.sample)
print("Original Sample:",sample)
