import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
print("000")
# 定义拟合函数
def model(r, k0, k1, k2, k3, k4):
    return 1 + k0*r**2 + k1*r**4 + k2*r**6 + k3*r**8 + k4*r**10
print("000")
# 准备数据（示例数据）
r_data = np.array([0, 0.0995, 0.200,   0.300,    0.399, 0.499,0.600,  0.700, 0.800,0.900,1])  # 请替换为您的数据
g_data = np.array([1,   1.01, 1.077,   1.2345,  1.5267, 1.801,2.0978, 2.500, 3.080,4.000,4.645])  # 请替换为您的数据
print("000")
# 执行曲线拟合
params, covariance = curve_fit(model, r_data, g_data, p0=None, sigma=0.0000001, absolute_sigma=0.0000001, check_finite=True, method='dogbox')

# 提取拟合参数
k0, k1, k2, k3, k4 = params

# 绘图
plt.scatter(r_data, g_data, label='Data', color='red')
r_fit = np.linspace(min(r_data), max(r_data), 100)
g_fit = model(r_fit, *params)
plt.plot(r_fit, g_fit, label='Fitted Curve', color='blue')
plt.xlabel('r')
plt.ylabel('g')
plt.legend()
plt.show()

# 输出拟合参数
print(f'Fitted parameters: k0={k0}, k1={k1}, k2={k2}, k3={k3}, k4={k4}')