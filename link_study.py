import numpy as np
import chainer.links as L
from chainer import Variable


f = L.Linear(3, 2)  # 类似于Scala中，定义于伴生对象中的apply方法
print("f.W:", f.W.data)
print("f.b:", f.b.data)
X = Variable(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
print("X:", X.data)
Y = f(X)  # f.__call__(X)，类似于Scala中定义于伴生类的apply方法
print("Y:", Y.data)

f.cleargrads()  # 必须调用，这里是一个初始化的作用，否则nan，not a number
Y.grad = np.ones((3, 2), dtype=np.float32)  # 链式求导，dX = dY * dY/dX
Y.backward()  # backward pass计算dFinalOutputVar/dVar
# 应该要和1/m相乘
print("1st dW:", f.W.grad)  # dW = dY.T * X，只需确保shape一致即可
print("1st db:", f.b.grad)  # db = np.sum(dY, axis=0)
Y.backward()
Y.backward()
Y.backward()
print("4th dW:", f.W.grad)  # 梯度会累加
print("4th db:", f.b.grad)  # 所以每个min-batch之前，都需要调用f.cleargrads()
