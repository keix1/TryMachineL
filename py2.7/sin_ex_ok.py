# -*- coding: utf-8 -*-

# http://qiita.com/yoshizaki_kkgk/items/bfe559d1bdd434be03ed�@�̓��e

# ���l�v�Z�֘A
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

# �����̃V�[�h���Œ�
random.seed(1)

# �W�{�f�[�^�̐���
#   �^�̊֐��Ƃ���sin�֐���
x, y = [], []
for i in np.linspace(-3,3,100):
    x.append([i])
    y.append([math.sin(i)])  # �^�̊֐�
# chainer�̕ϐ��Ƃ��čēx�錾
x = Variable(np.array(x, dtype=np.float32))
y = Variable(np.array(y, dtype=np.float32))

# NN���f����錾
model = MyChain()

# �����֐��̌v�Z
#   �����֐��ɂ͎���덷(MSE)���g�p
def forward(x, y, model):
    t = model.predict(x)
    loss = F.mean_squared_error(t, y)
    return loss

# chainer��optimizer
#   �œK���̃A���S���Y���ɂ� Adam ���g�p
optimizer = optimizers.Adam()
# model�̃p�����[�^��optimizer�ɓn��
optimizer.setup(model)

# �p�����[�^�̊w�K���J��Ԃ�
for i in range(0,1000):
    loss = forward(x, y, model)
    print(loss.data)  # �����MSE��\��
    optimizer.update(forward, x, y, model)

# �v���b�g
t = model.predict(x)
plt.plot(x.data, y.data)
plt.scatter(x.data, t.data)
plt.grid(which='major',color='gray',linestyle='-')
plt.ylim(-1.5, 1.5)
plt.xlim(-4, 4)
plt.show()

