# choose two class in the iris dataset to train a logistic regression model
# use gradient descent algorithm to optimize the loss function
from sklearn.datasets import load_iris
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        return self.train_dataset[:, :-1], self.train_dataset[:, -1]

    def getTestData(self):
        return self.test_dataset[:, :-1], self.test_dataset[:, -1]

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


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.num_iteration = 1000
        self.threshold = 0.00001
        self.pi_x = None
        pass

    def predict(self,X):
        # (N_samples,m_features)
        N, M = X.shape  # N samples,M features
        X = np.concatenate([X, np.ones([N, 1])], axis=1)
        # predict = self.pi_x(X)
        predict = (np.exp(np.dot(X, self.w)) / (1 + np.exp(np.dot(X, self.w)))).flatten()
        predict = np.where(predict >= 0.5,1,0)
        return predict

    def fit(self, X, Y,lr=0.01):  # X:(N_samples,M_features) Y:(N_samples,)
        N,M = X.shape # N samples,M features
        self.w = np.random.randn(M + 1, 1)
        X = np.concatenate([X, np.ones([N , 1])], axis=1)
        # print(self.w.shape,X.shape)
        pi_x = lambda x : (np.exp(np.dot(X,self.w)) / (1 + np.exp(np.dot(X,self.w)))).flatten()
        self.pi_x = pi_x
        # print("np.dot(X,self.w)",np.dot(X,self.w).shape)
        losses = []
        prev_loss = np.inf
        for i in range(self.num_iteration):
            positive = pi_x(X)
            negative = 1 - positive

            loss = -np.sum(Y * np.ma.log(positive).filled(0) + (1 - Y) * np.ma.log(negative).filled(0))
            if loss < 0:
                print("?????There must be a bug here....")
            if prev_loss - loss < self.threshold:
                print("Oh!!!Let's end this iteration!!!")
                break
            gradient = -np.dot(Y - positive,X) / N  # (m_features,)
            # print("gradient[:,None].shape:",gradient[:,None].shape)
            # print("w shape:",self.w.shape)
            self.w = self.w - lr * gradient[:,None]
            # print((Y * np.log(positive) + (1 - Y) * np.log(1 - positive)).shape)
            # print(loss)
            losses.append(loss)
            prev_loss = loss
        plt.plot(losses)
        plt.xlabel("steps")
        plt.ylabel("Loss")
        plt.title("Loss - Steps")

    def _update(self):
        pass

    def _loss(self):
        pass

    def _loss_gradient(self):
        pass


print("Hello World!")
model = LogisticRegression()
num_samples = 20
dataset = Dataset(num_samples)
train_data, train_label = dataset.getTrainData()
train_data = train_data[:2*num_samples,:]
train_label = train_label[:2*num_samples]
print(train_data.shape,train_label.shape)
model.fit(train_data,train_label)
test_data,test_label = dataset.getTestData()
test_data = test_data[:2*(50 - num_samples),:]
test_label = test_label[:2*(50 - num_samples)]
prediction = model.predict(test_data)
print(prediction.shape)
print(test_label.shape)
right = np.sum(prediction == test_label)
all = len(test_label)
print("Final Accuracy:{}/{}  {:.4f}%".format(right,all,right/all * 100))

