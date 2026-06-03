import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import os
from feature_information import FeatureInformation


def plot_result(feature: FeatureInformation, target: pd.DataFrame, path,
                title: str = "fit", exp_config: dict = None):
    """
    绘制拟合图，标注 R²、公式。

    参数:
        feature: 最优特征对象
        target: 目标值 DataFrame
        path: 保存路径
        title: 文件名标识（如 'exp_analytical_sis'）
        exp_config: 实验配置信息（用于图片标题）
    """

    # 定义颜色
    commoncolor3 = np.array([[81, 81, 81],
                            [241, 64, 64],
                            [26, 111, 223],
                            [55, 173, 107],
                            [177, 119, 222],
                            [204, 153, 0],
                            [0, 203, 204],
                            [125, 78, 78],
                            [142, 142, 0],
                            [251, 101, 1],
                            [102, 153, 204],
                            [111, 184, 2]]) / 255.0

    threecolors = np.array([[57, 156, 102],
                            [0, 128, 102],
                            [77, 133, 189],
                            [247, 144, 61],
                            [89, 169, 90]]) / 255.0

    colorblue = np.array([36, 103, 180]) / 255.0
    redcolor = np.array([255, 59, 59]) / 255.0

    # 获取预测值（带系数）和真实值
    x = feature.get_full_data().to_numpy().flatten()   # q_pred (带系数)
    y = target.to_numpy().flatten()                     # q_real

    # 创建图形
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_facecolor('w')

    # 绘制数据点
    ax.plot(x, y, 'o', color=threecolors[0, :],
            markersize=6,
            markerfacecolor=threecolors[0, :],
            markeredgecolor=threecolors[0, :], label='Data')

    # 坐标范围（归一化，保证对角线从原点开始）
    data_max = max(x.max(), y.max())
    data_min = min(x.min(), y.min())
    plot_range = data_max - data_min
    margin = plot_range * 0.08
    axis_min = data_min - margin
    axis_max = data_max + margin

    # 绘制拟合线（y = x 对角线）
    ax.plot([axis_min, axis_max], [axis_min, axis_max], '-',
            color=redcolor, linewidth=4, label='y = x (Perfect Fit)')

    # 设置刻度和范围
    tick_step = max(plot_range // 6, 0.5)
    ax.set_xticks(np.arange(0, axis_max + tick_step, tick_step))
    ax.set_yticks(np.arange(0, axis_max + tick_step, tick_step))
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])

    # 设置坐标轴标签
    ax.set_xlabel(r'$q_{pred}$', fontsize=30, fontweight='normal')
    ax.set_ylabel(r'$q_{real}$', fontsize=30, fontweight='normal')

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    # 刻度格式
    ax.tick_params(axis='x', which='both', direction='in',
                   length=6, width=2, labelsize=12, top=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   length=6, width=2, labelsize=12, right=True)
    ax.tick_params(axis='both', which='minor', bottom=False,
                   top=False, left=False, right=False)
    ax.tick_params(axis='both', labelsize=25)
    ax.minorticks_on()

    # === 添加 R² 和公式标注 ===
    r2 = feature.get_r2()
    formula = feature.get_full_name()
    ax.text(0.05, 0.92, f'$R^2 = {r2:.4f}$',
            transform=ax.transAxes, fontsize=28, fontweight='bold',
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray',
                      boxstyle='round,pad=0.5'))

    # 公式分行显示（如果太长）
    formula_display = formula if len(formula) < 80 else formula[:77] + '...'
    ax.text(0.05, 0.80, f'Formula:',
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top')
    ax.text(0.05, 0.72, f'{formula_display}',
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(facecolor='lightyellow', edgecolor='gray',
                      boxstyle='round,pad=0.3'))

    # 显示图片
    full_path = os.path.join(path, f'fit_{title}.png')
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图片已保存: {full_path}")