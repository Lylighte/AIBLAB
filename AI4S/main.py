# from expansion import expand
from expansion import expand, expand_full
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit
from result_plotting import plot_result
import pandas as pd
import os

# ==================== 实验配置 ====================
# 选择系数拟合方法: 'analytical' 解析解, 'nn' 神经网络
FIT_METHOD = 'nn'
# 是否使用 SIS 筛选 (True=开启SIS, False=关闭SIS)
USE_SIS = True
# SIS 保留的特征数
SIS_N = 10
# NN 超参数 (仅 FIT_METHOD='nn' 时生效)
NN_EPOCHS = 100
NN_BATCH_SIZE = 128
NN_LR = 0.1
# 结果保存路径
path = os.getcwd()
# path = "yourfilepath"
# =================================================

# 读取数据
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))

print(f"原始特征数: {data.shape[1]}, 数据量: {data.shape[0]}")

# 初始输入的扩充
# data_expanded = expand(data)
data_expanded = expand_full(data, enable_binary_div=True)
print(f"扩展后特征数: {data_expanded.shape[1]}")

# 确定性独立筛选
if USE_SIS:
    data_selected = sis(data_expanded, focus.to_numpy(), SIS_N)
    print(f"SIS 筛选后特征数: {data_selected.shape[1]}")
else:
    data_selected = data_expanded
    print(f"跳过 SIS，全量特征数: {data_selected.shape[1]}")

# 系数拟合
r2, coef, loss, times = fit(data_selected, focus.to_numpy(),
                             method=FIT_METHOD,
                             num_epochs=NN_EPOCHS,
                             batch_size=NN_BATCH_SIZE,
                             lr=NN_LR)

# 结果整理
results = sort_result(data_selected.to_numpy(),
                      data_selected.columns.to_numpy(),
                      r2, coef, loss, 10)

# 输出日志
display_result(results, 10, path)

# 绘制图像
plot_result(results[0], focus, path)

print("实验完成!")


# TODO: 改图片 — 加R²标签+公式，x/y轴归一化，log里加最优公式+R²+系数