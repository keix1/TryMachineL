# -*- coding: utf-8 -*-

# 数値計算関連
import math
import random
import numpy as np
import matplotlib.pyplot as plt
# chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from MyChain import MyChain

# 乱数のシードを固定
random.seed(1)

# 標本データの生成
#   真の関数としてsin関数を
x, y = [], []
for i in np.linspace(-3,3,100):
    x.append([i])
    y.append([math.sin(i)])  # 真の関数
# chainerの変数として再度宣言
x = Variable(np.array(x, dtype=np.float32))
y = Variable(np.array(y, dtype=np.float32))

## NNモデルを宣言
model = MyChain()

# プロット
t = model.predict(x)
plt.plot(x.data, y.data)
plt.scatter(x.data, t.data)
plt.grid(which='major',color='gray',linestyle='-')
plt.ylim(-1.5, 1.5)
plt.xlim(-4, 4)
plt.show()
