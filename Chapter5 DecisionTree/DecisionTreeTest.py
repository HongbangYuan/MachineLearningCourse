# Implement a decision tree on the given dataset
# Chapter5 Lihang's book,Page71
# Problem5.1 use c4.5 to construct a decision tree
from sklearn.datasets import load_iris
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

values = [['青年', '否', '否', '一般', '否'],
          ['青年', '否', '否', '好', '否'],
          ['青年', '是', '否', '好', '是'],
          ['青年', '是', '是', '一般', '是'],
          ['青年', '否', '否', '一般', '否'],
          ['中年', '否', '否', '一般', '否'],
          ['中年', '否', '否', '好', '否'],
          ['中年', '是', '是', '好', '是'],
          ['中年', '否', '是', '非常好', '是'],
          ['中年', '否', '是', '非常好', '是'],
          ['老年', '否', '是', '非常好', '是'],
          ['老年', '否', '是', '好', '是'],
          ['老年', '是', '否', '好', '是'],
          ['老年', '是', '否', '非常好', '是'],
          ['老年', '否', '否', '一般', '否'],
          ]
labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
train_data = pd.DataFrame(values, columns=labels)


# print(train_data)


def calEntropy(dataset):  # empirical entropy of one dataset
    length = len(dataset)
    info = {}
    for i in range(length):
        label = dataset.iloc[i, -1]
        if label not in info:
            info[label] = 1
        else:
            info[label] = info[label] + 1
    # print(info)
    # print(length)
    # print([num for num in info.values()])
    # print([num / length * np.log2(num / length) for num in info.values()])
    return -sum([num / length * np.log2(num / length) for num in info.values()])


def calConditionalEntropy(dataset, axis):  # empirical conditional entropy of a dataset and one feature
    # axis=0,1,2,3
    if axis not in [0, 1, 2, 3]:
        raise Exception("The axis must be in [0,1,2,3]!Please check the axis parameter!")
    info = {}
    length = len(dataset)
    for i in range(length):
        feature = dataset.iloc[i, axis]
        label = dataset.iloc[i, -1]
        # print(feature,label)
        if feature not in info:
            info[feature] = [{label: 1}, 1]
        else:
            tmp = info[feature]
            tmp[1] = tmp[1] + 1
            if label not in tmp[0]:
                tmp[0][label] = 1
            else:
                tmp[0][label] = tmp[0][label] + 1
    # print(info)
    # info:{'青年': [{'否': 3, '是': 2}, 5], '中年': [{'否': 2, '是': 3}, 5], '老年': [{'是': 4, '否': 1}, 5]}
    Hx = lambda piece: -sum([num / piece[1] * np.log2(num / piece[1]) for num in piece[0].values()])
    # for piece in info.values():
    #     print(piece,piece[1],length,Hx(piece))
    return sum([piece[1] / length * Hx(piece) for piece in info.values()])


def calInfoGain(entropy, conditional_entropy):
    return entropy - conditional_entropy


def calFeatureEntropy(dataset, axis):  # HA(D) entropy of a feature
    # axis=0,1,2,3
    if axis not in [0, 1, 2, 3]:
        raise Exception("The axis must be in [0,1,2,3]!Please check the axis parameter!")
    info = {}
    length = len(dataset)
    for i in range(length):
        feature = dataset.iloc[i, axis]
        if feature not in info:
            info[feature] = 1
        else:
            info[feature] = info[feature] + 1
    return -sum([num / length * np.log2(num / length) for num in info.values()])


def calInfoGainRatio(info_gain, feature_entropy):
    return info_gain / feature_entropy


class Node:
    def __init__(self, label, subtree=None):
        '''
        if subtree is None:
            label:the final class
        if subtree is not None:
            label:the feature name
            subtree:{feature_value:next_node}
        '''
        self.label = label
        self.subtree = subtree

class LeafNode(Node):
    def __init__(self,label,samples):
        super(LeafNode,self).__init__(label)
        self.samples = samples


def calLabel(dataset):
    return dict(dataset.iloc[:,-1].value_counts())


# def mergeLeaf(leaf1,leaf2):


class DecisionTree():
    def __init__(self,dataset):
        self.epsilon = 0.1
        self.alpha = 0.01
        self.root = self.construct(dataset)
        # self.cutBranch(self.root)

    def construct(self,dataset):
        if len(dataset.iloc[:, -1].value_counts()) == 1:  # if all the instances belong to the same class
            # print(dataset)
            return LeafNode(dataset.iloc[0, -1], calLabel(dataset))
        if dataset.shape[1] == 1:  # if all the features are dropped,choose the most frequent class
            return LeafNode(dataset.iloc[:, -1].value_counts().index[0], calLabel(dataset))
        entropy = calEntropy(dataset)
        conditional_entropies = [calConditionalEntropy(dataset, axis) for axis in range(dataset.shape[1] - 1)]
        feature_entropies = [calFeatureEntropy(dataset, axis) for axis in range(dataset.shape[1] - 1)]
        info_gain_ratios = [calInfoGainRatio(calInfoGain(entropy, conditional_entropies[idx]), feature_entropies[idx])
                            for idx in range(dataset.shape[1] - 1)]
        max_info_gain_ratio, idx = np.max(info_gain_ratios), np.argmax(info_gain_ratios)
        if max_info_gain_ratio < self.epsilon:
            return LeafNode(dataset.iloc[:, -1].values_counts().index[0], calLabel(dataset))
        feature_name = dataset.columns[idx]
        # print("Processing on feature ",feature_name)
        subtree = {}
        for feature_value in dataset.iloc[:, idx].value_counts().index:
            # print("Splitting the dataset according to name ","...")
            new_dataset = dataset.copy(deep=True)
            new_dataset.drop(new_dataset.columns[idx], axis=1, inplace=True)
            indexes = [new_dataset.index[i] for i in range(len(dataset)) if dataset.iloc[i, idx] != feature_value]
            new_dataset.drop(indexes, inplace=True)
            subtree[feature_value] = self.construct(new_dataset)
        return Node(feature_name, subtree)

    # def cutBranch(self,root):

    # def cut(self,node,feature_value):
    #
    #     node.subtree[feature_value] = LeafNode()


DT = DecisionTree(train_data)




# # %% train_data:dataframe  epsilon:threshold
# epsilon = 0.1
#
#
# def construct(dataset):
#     if len(dataset.iloc[:, -1].value_counts()) == 1:  # if all the instances belong to the same class
#         # print(dataset)
#         return LeafNode(dataset.iloc[0, -1],dataset)
#     if dataset.shape[1] == 1:  # if all the features are dropped,choose the most frequent class
#         return LeafNode(dataset.iloc[:, -1].value_counts().index[0],dataset)
#     entropy = calEntropy(dataset)
#     conditional_entropies = [calConditionalEntropy(dataset, axis) for axis in range(dataset.shape[1] - 1)]
#     feature_entropies = [calFeatureEntropy(dataset, axis) for axis in range(dataset.shape[1] - 1)]
#     info_gain_ratios = [calInfoGainRatio(calInfoGain(entropy, conditional_entropies[idx]), feature_entropies[idx])
#                         for idx in range(dataset.shape[1] - 1)]
#     max_info_gain_ratio, idx = np.max(info_gain_ratios), np.argmax(info_gain_ratios)
#     if max_info_gain_ratio < epsilon:
#         return LeafNode(dataset.iloc[:, -1].values_counts().index[0],dataset)
#     feature_name = dataset.columns[idx]
#     # print("Processing on feature ",feature_name)
#     subtree = {}
#     for feature_value in dataset.iloc[:, idx].value_counts().index:
#         # print("Splitting the dataset according to name ","...")
#         new_dataset = dataset.copy(deep=True)
#         new_dataset.drop(new_dataset.columns[idx], axis=1, inplace=True)
#         indexes = [new_dataset.index[i] for i in range(len(dataset)) if dataset.iloc[i, idx] != feature_value]
#         new_dataset.drop(indexes, inplace=True)
#         subtree[feature_value] = construct(new_dataset)
#     return Node(feature_name, subtree)


# root = construct(train_data)  # root node of the decision tree
#
# #%%
# def cutBranch(root):
#     return root
#
#
# root = cutBranch(root)


# # %%
# entropy = calEntropy(train_data)
# print("Entropy of the whole dataset:", entropy)
# columns = ['年龄', '有工作', '有自己的房子', '信贷特征']
# for axis in [0, 1, 2, 3]:
#     conditional_entropy = calConditionalEntropy(train_data, axis=axis)
#     feature_entropy = calFeatureEntropy(train_data, axis=axis)
#     info_gain = calInfoGain(entropy, conditional_entropy)
#     info_gain_ratio = calInfoGainRatio(info_gain, feature_entropy)
#     print("特征：", columns[axis], " 条件熵：", conditional_entropy,
#           " 信息增益：", info_gain, "信息增益比", info_gain_ratio)
