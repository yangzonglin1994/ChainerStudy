import numpy as np
from chainer import Variable

from my_link_based_model import MyLinkBasedModel


def calcu_cost(Y, y):
    Y, y = Y.data, y.data
    loss = -(y * np.log(Y) + (1-y) * np.log(1-Y))
    # print(type(loss))
    # print(loss.shape)
    cost = np.sum(loss) / Y.shape[1]
    return cost

model = MyLinkBasedModel()
X = Variable(np.array([[1, 2, 3], [2, 1, 3], [2, 2, 1]], dtype=np.float32)).T
y = Variable(np.array([0, 1, 0], dtype=np.float32))
iter_num = 100
LR = 0.1
for i in range(iter_num):
    Y = model(X)
    print(calcu_cost(Y, y))
    model.cleargrads()
    Y.grad = (1-y.data)/(1-Y.data)-y.data/Y.data
    Y.backward()
