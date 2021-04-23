# implement perception and visualize the results
import numpy as np
from numpy.random import randn
from time import time
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# np.random.seed(0)
w = randn(2,1)
b = randn(1)
sample_len = 100
x = randn(sample_len,2)
y = np.dot(w.T,x.T).T
pos = y + 2 + abs(randn(*y.shape))
neg = y - 2 - abs(randn(*y.shape))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(x[:,0],x[:,1],pos[:,0],'r*')
ax.plot(x[:,0],x[:,1],neg[:,0],'b*')
x = np.linspace(x.min(),x.max(),100)
y = np.linspace(y.min(),y.max(),100)
X,Y = np.meshgrid(x,y)
# ax.plot_surface(X,Y,Z )
ax.plot_surface(X,Y,Z=w[0][0] * X + w[1][0] * Y,alpha=0.5)

# ax.plot(data[:,0],data[:,1],data[:,2],'*')
# ax.plot(1,1,2)
plt.show()


#
# np.random.seed(102039)
#
# def gen_rand_lines(length,dims=2):
#     line_data = np.empty((dims,length))
#     line_data[:,0] = np.random.rand(dims)
#     for index in range(1,length):
#         step = (np.random.rand(dims) - 0.5) * 0.1
#         line_data[:,index] = line_data[:,index-1] + step
#     return line_data
#
# fig = plt.figure()
# ax = Axes3D(fig)
# data = [gen_rand_lines(25,3) for index in range(50)]
# lines = [ax.plot(dat[0,0:1],dat[1,0:1],dat[2,0:1],'o-')[0] for dat in data]
# ax.set_xlim3d([0.0,1.0])
# ax.set_xlabel('X')
# ax.set_ylim3d([0.0,1.0])
# ax.set_ylabel('Y')
# ax.set_zlim([0.0,1.0])
# ax.set_zlabel('Z')
# ax.set_title('3D Test')
# plt.show()
#
#
#
#



