import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# 鸢尾花(iris)数据集
# 数据集内包含 3 类共 150 条记录，每类各 50 个数据，
# 每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
# 可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
# 这里只取前100条记录，两项特征，两个类别。
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]


def false_kernel(x1, x2):
    return np.dot(x1, x2)


class SVM:
    def __init__(self):
        self.tol = 0
        self.C = np.inf
        self.kernel = false_kernel  # consider only the linear model
        self.b = None
        self.alpha = None
        self.E = None
        pass

    def fit(self, X, Y):
        self.b = 0
        self.alpha = np.zeros(X.shape[0])
        self.E = np.array([self._predict(X, Y, x_i) - y_i for x_i, y_i in zip(X, Y)])
        return self.SMO(X, Y)
        # pass

    def SMO(self, X, Y):
        m, n = X.shape
        num_changed = 0
        examine_all = 1
        iter = 1
        max_iter = 10000
        while iter < max_iter and (num_changed > 0 or examine_all):
            num_changed = 0
            if examine_all:
                for i in range(m):
                    num_changed += self._examineExample(X, Y, i)
            else:
                for alpha in self.alpha:
                    if 0 < alpha < self.C:
                        num_changed += self._examineExample(X, Y, i)
            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

            print("Iteration {}".format(iter))
            iter = iter + 1
        print("Training Finished!")
        # w = np.array([np.sum(self.alpha * Y * x_i) for x_i in X])
        w = np.dot(self.alpha * Y,X)
        return w

    def predict(self, X, Y):
        return np.array([self._predict(X, Y, x_i) for x_i in X])

    # def _get_linear_w(self):
    #     return np.array(np.sum(self.alpha * Y))

    def _predict(self, X, Y, x_i):
        return np.sum(self.alpha * Y * self.kernel(X, x_i)) + self.b

    def _examineExample(self, X, Y, i2):
        y2 = Y[i2]
        alpha2 = self.alpha[i2]
        x2 = X[i2, :]
        E2 = self.E[i2]
        r2 = E2 * y2

        # find the KKT violators:
        if alpha2 < self.C and r2 < -self.tol or alpha2 > 0 and r2 > self.tol:
            # if len(np.nonzero(self.alpha)[0]) > 1:
            i1 = self._chooseSecond(i2)
            if self._takeStep(X, Y, i1, i2):
                return 1
            for i,alpha in enumerate(self.alpha):
                if 0 < alpha < self.C:
                    i1 = i
                    if self._takeStep(X,Y,i1,i2):
                        return 1
            print("Something unusual happened!")
        return 0

    def _chooseSecond(self, i2):
        E2 = self.E[i2]
        E = np.ma.masked_equal(self.E, E2, copy=False)
        if E2 >= 0:
            i1 = np.argmin(E)
            E1 = E[i1]
        else:
            i1 = np.argmin(E)
            E1 = E[i1]
        if E1 == E2: # choose randomly
            i1 = i2
            while i1 == i2:
                i1 = int(np.random.uniform(0,self.alpha.shape[0]))
                # i1 = int(np.random.uniform(0, X.shape[0]))
            return i1
        else:
            return i1

    def _takeStep(self, X, Y, i1, i2):
        # solve the quadratic optimization problem containing alpha1 and alpha2
        if i1 == i2:
            return 0

        alpha1, alpha2 = self.alpha[i1], self.alpha[i2]
        y1, y2 = Y[i1], Y[i2]
        x1, x2 = X[i1], X[i2]
        E1, E2 = self.E[i1], self.E[i2]
        L = max(0, alpha2 - alpha1) if y1 != y2 else max(0, alpha2 + alpha1 - self.C)
        H = min(self.C, self.C + alpha2 - alpha1) if y1 != y2 else min(self.C, alpha2 + alpha1)
        if L == H:
            return 0

        k11 = self.kernel(x1, x1)
        k22 = self.kernel(x2, x2)
        k12 = self.kernel(x1, x2)
        eta = k11 + k22 - 2 * k12
        if eta < 0:
            print("Eta<0! Ops! Something went wrong!")
            return 0

        # alpha2_new = alpha2 + y2 * np.abs((E1 - E2)) / eta
        alpha2_new = alpha2 + y2 * (E1 - E2) / eta
        if alpha2_new > H:
            alpha2_new = H
        elif alpha2_new < L:
            alpha2_new = L

        if alpha2_new < 1e-8:
            alpha2_new = 0
        if alpha2_new > self.C - 1e-8:
            alpha2_new = self.C

        # if np.abs(alpha2 - alpha2_new) < 0.0001:
        #     return 0
        alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)


        # update the parameters
        b1_new = -E1 - y1 * k11 * (alpha1_new - alpha1) - y2 * k12 * (alpha2_new - alpha2) + self.b
        b2_new = -E2 - y1 * k12 * (alpha1_new - alpha1) - y2 * k22 * (alpha2_new - alpha2) + self.b
        self.b = (b1_new + b2_new) / 2
        self.alpha[i1] = alpha1_new
        self.alpha[i2] = alpha2_new
        print("Alpha1:{}->{}        Alpha2:{}->{}".format(alpha1,alpha1_new,alpha2,alpha2_new))
        self.E[i1] = self._predict(X, Y, x1) - y1
        self.E[i2] = self._predict(X, Y, x2) - y2
        return 1


X, Y = create_data()  # X:(samples,features)  Y:(samples,)
# X = np.concatenate([X,np.ones([X.shape[0],1])],axis=1)
model = SVM()
w = model.fit(X, Y)
prediction = model.predict(X, Y)
# print(prediction)
print("Final Accuracy:{}/{}".format(np.sum(np.sign(prediction) == Y),len(Y)))

positive = X[Y == 1]
negative = X[Y == -1]
plt.scatter(positive[:, 0], positive[:, 1], c='r')
plt.scatter(negative[:, 0], negative[:, 1], c='g')
line_x = np.array([X.min(axis=0)[0],X.max(axis=0)[0]])
line_y = line_x * (-w[0])/w[1] - model.b/w[1]
plt.plot(line_x,line_y)

# # given dataset X,Y and calculate the solution alpha
# # X:(samples,features)  Y:(samples,) labels:±1
# def SMO(X, Y):
#     alpha = np.zeros(X.shape[0])
#     b = 0
#     for i,alpha_i in enumerate(alpha):
#         y_i = Y[i]
#         x_i = X[i,:]
#         g_xi = np.sum(alpha * Y * false_kernel(X,x_i)) + b
#         if y_i * g_xi == 1 and
#
#
#
#     b = 0
#     gx = lambda xj: np.sum(alpha * Y * np.dot(X, xj)) + b
#
#
#     alpha = None
#     return alpha
