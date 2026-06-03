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
FIT_METHOD = 'analytical'
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

# 自动生成实验标识
_exp_label = f"{'nn' if FIT_METHOD == 'nn' else 'ana'}_{'sis' if USE_SIS else 'full'}"
_exp_config = {
    '方法': '神经网络' if FIT_METHOD == 'nn' else '解析解',
    'SIS': f'开启 (n={SIS_N})' if USE_SIS else '关闭',
    '扩展方式': 'expand_full (含除法)',
}
if FIT_METHOD == 'nn':
    _exp_config.update({
        'NN_EPOCHS': NN_EPOCHS,
        'NN_BATCH_SIZE': NN_BATCH_SIZE,
        'NN_LR': NN_LR,
    })

print(f"{'=' * 60}")
print(f"  AI4S 实验 — 配置: {_exp_label}")
print(f"  FIT_METHOD={FIT_METHOD}, USE_SIS={USE_SIS}, SIS_N={SIS_N}")
print(f"{'=' * 60}")

# 读取数据
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))

print(f"原始特征: {data.shape[0]} 条, {data.shape[1]} 个特征 ({list(data.columns)})")

# 初始输入的扩充
data_expanded = expand_full(data, enable_binary_div=True)
print(f"扩展后特征数: {data_expanded.shape[1]}")

# 确定性独立筛选
if USE_SIS:
    data_selected = sis(data_expanded, focus.to_numpy(), SIS_N)
    print(f"SIS 筛选后: {data_selected.shape[1]} 特征")
else:
    data_selected = data_expanded
    print(f"跳过 SIS，全量: {data_selected.shape[1]} 特征")

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

# 输出日志（带实验标识）
display_result(results, 10, path,
               exp_label=_exp_label, exp_config=_exp_config)

# 绘制图像（带实验标识 + R² + 公式标注）
plot_result(results[0], focus, path,
            title=_exp_label, exp_config=_exp_config)

print(f"\n{'=' * 60}")
print(f"  实验 [{_exp_label}] 完成!")
print(f"  最优 R² = {results[0].get_r2():.6f}")
print(f"  公式: {results[0].get_full_name()}")
print(f"{'=' * 60}")