# This is a file used for implement the perception algorithm in a 3D space. I run this script in pycharm scientific
# mode and turn the interactive mode off or a bunch of figures will pop out during the optimization ,which is quite
# annoying.The figures in each step is stored in the folder "./figs" and their names representing the optimization
# steps. If you run the script once again,redundant figures may remain as I failed to delete the pictures generated
# in previous running processes and I am reluctant to fix this.

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import os


class Perception:
    def __init__(self, num_samples, eta, w=None, b=None):
        self.eta = eta
        self.num_samples = num_samples
        self.w_opt = np.random.randn(3, 1) if w is None else w
        self.b_opt = np.random.randn(1, 1) if b is None else b
        # self.w = np.zeros([3,1])
        self.w = np.array([[0, 0, 1]]).T
        self.b = np.zeros([1, 1])
        self.global_step = 0
        self.gen_data()

    def train(self):
        self.global_step = 0
        while True:
            print(self.global_step)
            if self.global_step > 1000:
                print("We should break now!")
                break

            pos_x = np.concatenate((self.x, -self.pos), axis=1)  # num_samples,3
            check_pos = np.dot(pos_x, self.w) + self.b
            check_pos = np.array(np.where(check_pos <= 0))

            neg_x = np.concatenate((self.x, -self.neg), axis=1)
            check_neg = np.dot(neg_x, self.w) + self.b
            check_neg = np.array(np.where(check_neg >= 0))

            if not check_pos.size == 0:
                sample = pos_x[check_pos[0, 0]]
                y = 1
                self.update(np.expand_dims(sample,axis=0), y)
                self.global_step = self.global_step + 1
                self.plot_data(savefig=True)
                continue

            if not check_neg.size == 0:
                sample = neg_x[check_neg[0, 0]]
                y = -1
                self.update(np.expand_dims(sample,axis=0), y)
                self.global_step = self.global_step + 1
                self.plot_data(savefig=True)
                continue

            break

        print("Finished!Now go to file {} to see the results!".format(str(os.getcwd()) + "\\figs"))
        self.plot_data()

    def update(self, sample, y):  # sample:(1,3) y:{1,-1}
        self.w = self.w + self.eta * y * sample.T  # (3,1)
        self.b = self.b + self.eta * y

    def gen_data(self):
        x = np.random.randn(self.num_samples, 2)  # (num_samples,2)
        z = np.dot(self.w_opt[:-1].T / self.w_opt[-1, 0], x.T).T + self.b_opt  # (num_samples,1)
        neg = z + 1 + 2 * abs(np.random.randn(*z.shape))  # things should be:x + y - z
        pos = z - 1 - 2 * abs(np.random.randn(*z.shape))
        self.x = x
        # self.z = z
        self.pos = pos  # z coordinate of positive samples (num_samples,1)
        self.neg = neg  # z coordinate of negative samples (num_samples,1)

    def plot_data(self,savefig=False):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = self.x[:, 0]
        y = self.x[:, 1]  # (samples,2)
        ax.plot(x, y, self.pos[:, 0], 'r*')
        ax.plot(x, y, self.neg[:, 0], 'b*')
        x = 1.8 * np.linspace(x.min(), x.max(), 100)
        y = 1.8 * np.linspace(x.min(), x.max(), 100)
        X, Y = np.meshgrid(x, y)
        # ax.plot_surface(X,Y,Z )
        plane1 = ax.plot_surface(X, Y,
                                 Z=self.w_opt[0, 0] / self.w_opt[-1, 0] * X + self.w_opt[1, 0] / self.w_opt[-1, 0] * Y,
                                 alpha=0.8,
                                 label='Target Plane')
        plane2 = ax.plot_surface(X, Y,
                                 Z=self.w[0, 0] / self.w[-1, 0] * X + self.w[1, 0] / self.w[-1, 0] * Y,
                                 alpha=0.3,
                                 label='Current Plane')
        plane1._facecolors2d = plane1._facecolor3d
        plane1._edgecolors2d = plane1._edgecolor3d
        plane2._facecolors2d = plane2._facecolor3d
        plane2._edgecolors2d = plane2._edgecolor3d
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=15, azim=19)
        # plt.show()

        if savefig:
            # if os.path.exists('./figs'):
            #     shutil.rmtree('./figs')
            #     os.mkdir('./figs')
            # else:
            #     os.mkdir('./figs')
            plt.savefig('./figs/{}.png'.format(str(self.global_step)),dpi=fig.dpi)
        # else:
            # plt.show()

if __name__ == '__main__':
    print("Hello World!")
    w = np.array([[1,1,1]]).T
    b = np.array([[0]])
    perception = Perception(300, 1,w,b)
    perception.train()
    perception.plot_data()
