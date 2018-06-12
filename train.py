import numpy as np
from chainer import Variable

from my_linear import MyLinear
from my_link_based_model import MyLinkBasedModel


def calcu_cost(Y, y):
    """
    计算整个训练集的cost
    :param Y: class Variable object, model output, shape = (1, m)
    :param y: class Variable object, actual value, shape = (1, m)
    :return: cost of entire training set
    """
    Y, y = Y.data, y.data
    loss = -(y * np.log(Y) + (1-y) * np.log(1-Y))
    # print(type(loss))
    # print(loss.shape)
    cost = np.sum(loss) / Y.shape[1]
    return cost

model = MyLinkBasedModel()
X = Variable(np.array([[1, 2, 3], [2, 1, 3], [2, 2, 1]], dtype=np.float32)).T
y = Variable(np.array([0, 1, 0], dtype=np.float32)).reshape(1, 3)
iter_num = 10000
LR = 0.1

# BGD，一轮迭代，只更新一次模型参数
# BGD，只有一个batch，故一轮迭代只有一次forward and backward pass
for i in range(iter_num):
    Y = model(X)  # forward pass
    print(calcu_cost(Y, y))
    model.cleargrads()
    Y.grad = (1-y.data)/(1-Y.data)-y.data/Y.data
    Y.backward()  # backward pass，计算cost func关于W和b的梯度
    # update model param
    for layer in model.children():
        if isinstance(layer, MyLinear):
            layer.W.data -= LR * layer.W.grad
            layer.b.data -= LR * layer.b.grad

# apply model to predict
Y = model(X)
print(Y)
X = Variable(np.array([[1, 2, 3]], dtype=np.float32)).reshape(1, 3).T
Y = model(X)
print(Y)
