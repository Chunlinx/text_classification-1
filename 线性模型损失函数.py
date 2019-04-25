#coding:utf-8
import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 5))

x = np.arange(-10, 10, 0.01)
# 逻辑回归损失函数
logi = np.log(1 + np.exp(-x))
# 感知机损失函数
y_p = -x
y_p[y_p < 0] = 0
# 线性支持向量机
y_hinge = 1.0 - x
y_hinge[y_hinge < 0] = 0

plt.xlim([-3, 3])
plt.ylim([0, 4])
plt.plot(x, logi, 'r-', mec='k', label='Logistic Loss', lw=2)
plt.plot(x, y_p, 'g-', mec='k', label='0/1 Loss', lw=2)
plt.plot(x, y_hinge, 'b-', mec='k', label='Hinge Loss', lw=2)
plt.grid(True, ls='--')
plt.legend(loc='upper right')
plt.title('损失函数')
plt.xlabel('yf(x,w)')
plt.ylabel('loss')
plt.show()
