from chainer import ChainList

from my_linear import MyLinear
from my_relu import MyReLU
from my_sigmoid import MySigmoid


class MyLinkBasedModel(ChainList):
    def __init__(self):
        super(MyLinkBasedModel, self).__init__(
            MyLinear(3, 5), MyReLU(),
            MyLinear(5, 4), MyReLU(),
            MyLinear(4, 1), MySigmoid()
        )

    def __call__(self, X):
        Y = X
        for layer in ChainList.children(self):
            Y = layer(Y)
        return Y
