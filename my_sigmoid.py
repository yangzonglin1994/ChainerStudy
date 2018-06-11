import chainer.functions as F
from chainer import Function, Link


class MySigmoidFunction(Function):
    def forward_cpu(self, inputs):
        Z, = inputs
        A = F.sigmoid(Z).data
        return A,

    def backward_cpu(self, inputs, grad_outputs):
        Z, = inputs
        dA, = grad_outputs
        A = F.sigmoid(Z).data
        dZ = A * (1-A) * dA
        return dZ,

    def forward_gpu(self, inputs):
        pass

    def backward_gpu(self, inputs, grad_outputs):
        pass


def my_sigmoid(Z):
    return MySigmoidFunction()(Z)


class MySigmoid(Link):
    def __init__(self):
        super(MySigmoid, self).__init__()

    def __call__(self, Z):
        return my_sigmoid(Z)
