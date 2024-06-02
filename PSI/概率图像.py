import matplotlib.pyplot as plt
import numpy as np
from math import comb

# 定义函数
def calculate_p(n):
    p = []
    for k in range(n+1):
        # 第一项
        term1 = comb(n, k) * (1/3)**k * (1/2)**(n-k)
        # 第二项
        term2 = 0
        for j in range(int(n/4)+1):
            term2 += comb(n, 2*j+k) * (1/3)**(j+k) * (1/6)**j * (1/2)**(n-2*j-2*k)
        # 总概率
        total = term1 + term2
        p.append(total)
    return p

# 绘制图像
def plot_p(n):
    x = np.arange(n+1)
    y = calculate_p(n)
    plt.plot(x, y, marker='o')
    plt.xlabel('k')
    plt.ylabel('Probability')
    plt.title('P(k) for n={}'.format(n))
    plt.show()

# 测试函数
plot_p(500) # 将n=50代入函数进行测试
