# 引入math模块以使用数学函数

import math

a, b, c = 3, 6, 7

# 余弦定理计算弧度
cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
cos_beta = (a**2 + c**2 - b**2) / (2 * a * c)
cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)

# 转换为角度
alpha = math.degrees(math.acos(cos_alpha))
beta = math.degrees(math.acos(cos_beta))
gamma = math.degrees(math.acos(cos_gamma))

print(f"三边长分别为: {a}, {b}, {c}")
print(f"三个角分别为: {alpha}, {beta}, {gamma}")
print(f"和是否为180: {math.isclose(alpha + beta + gamma, 180)}")
