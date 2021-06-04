import numpy as np
import matplotlib.pyplot as plt
import math


def dJ(x):
    return 2 * x * math.sin(x) + (x ** 2) * np.cos(x)


def J(x):
    try:
        return (x ** 2) * np.sin(x)
    except:
        return float('inf')


plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x ** 2) * np.sin(plot_x)
plt.plot(plot_x, plot_y)
plt.show()
x = 3  # 随机选取一个起始点
eta = 0.05  # 学习率
epsilon = 1e-12  # 用来判断是否到达二次函数的最小值点的条件
history_x = [x]  # 用来记录使用梯度下降法走过的点的X坐标
while True:
    gradient = dJ(x)  # 梯度（导数）
    last_x = x
    x = x - eta * gradient
    history_x.append(x)
    if abs(J(last_x) - J(x)) < epsilon:  # 用来判断是否逼近最低点
        break
print(history_x)  # 打印到达最低点时x的值
plt.plot(plot_x, plot_y)
plt.plot(np.array(history_x), J(np.array(history_x)), color='r', marker='*')  # 绘制x的轨迹
plt.show()
