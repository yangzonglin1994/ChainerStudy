import numpy as np
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class MyChain(Chain):  # Python可以多继承，例如class MyClass(Class1, Class2)
    # 如果父类只有有参数构造器，则子类必须定义构造方法；否则子类可以不定义构造器
    # 如果子类定义了构造器，则必须显示地调用父类的构造器
    def __init__(self):
        # super(MyChain, self).__init__()  # 多继承的时候，这种比较方便，一次性调用所有父类的构造器
        Chain.__init__(self)  # 在子类中调用父类的方法，需要加上基类名作为前缀，且还需传入self
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)

    def __call__(self, X):
        H = self.l1(X)  # 除了调用父类方法，其他都不需要传入self
        return self.l2(H)
