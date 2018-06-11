from chainer import Function, Link


class MyReLUFunction(Function):
    def forward_cpu(self, inputs):
        Z, = inputs
        A = Z.copy()
        A[A < 0] = 0
        return A,

    def backward_cpu(self, inputs, grad_outputs):
        Z, = inputs
        dA, = grad_outputs
        dZ = Z.copy()
        dZ[dZ < 0] = 0
        dZ[dZ > 0] = 1
        dZ = dZ * dA
        return dZ,

    def forward_gpu(self, inputs):
        pass

    def backward_gpu(self, inputs, grad_outputs):
        pass


def my_relu(Z):
    return MyReLUFunction()(Z)


class MyReLU(Link):
    def __init__(self):
        super(MyReLU, self).__init__()

    def __call__(self, Z):
        return my_relu(Z)
