import math

import chainer
from chainer import Function, Link
from chainer import initializers


class MyLinearFunction(Function):
    """
    样本作为输入矩阵的列向量
    注意，这里是继承Function，和继承FunctionNode有一些不同
    如果继承FunctionNode，my_linear函数的实现参考Linear Link对应的linear函数
    """
    def forward_cpu(self, inputs):
        """
        注意，参数和返回值皆为numpy.ndarray的元组
        :param inputs: tuple of numpy.ndarray
        :return: tuple of numpy.ndarray
        """
        A, W, b = inputs
        Z = W.dot(A) + b
        return Z,

    def backward_cpu(self, inputs, grad_outputs):
        A, W, b = inputs
        m = A.shape[1]  # the number of sample
        dZ, = grad_outputs
        dA = W.T.dot(dZ)
        dW = dZ.dot(A.T) / m
        db = dZ.sum(axis=1, keepdims=True) / m
        return dA, dW, db

    def forward_gpu(self, inputs):
        pass

    def backward_gpu(self, inputs, grad_outputs):
        pass


def my_linear(A, W, b):
    return MyLinearFunction()(A, W, b)


class MyLinear(Link):
    def __init__(self, in_size, out_size):
        super(MyLinear, self).__init__()
        with self.init_scope():
            self.W = chainer.Parameter(
                initializers.Normal(1. / math.sqrt(in_size)),
                (out_size, in_size))
            self.b = chainer.Parameter(0, (out_size, 1))

    def __call__(self, A):
        return my_linear(A, self.W, self.b)
