"""
main_iterative.py — 实验 7.6.4: 迭代式针对性搜索

改进后的 SISSO 算法主循环:
  特征扩展 → SIS 筛选 → 系数拟合 → 引导策略 → 下一轮扩展
                                           ↓
                                      终止判断

用法: python main_iterative.py
配置: 修改下方 CONFIG 字典
"""

from expansion import expand, expand_full
from iterative_expansion import expand_guided
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit
from result_plotting import plot_result
from result_storage import FeatureNode, FeatureTree
from termination import stop_or_not
import pandas as pd
import os

# ==================== 实验配置 ====================
CONFIG = {
    # 系数拟合方法: 'analytical' | 'nn'
    'fit_method': 'analytical',
    # 导向策略: 'layered' | 'crossover' | 'replace'
    'strategy': 'crossover',
    # SIS 每轮保留特征数
    'sis_n': 50,
    # 输出保留 top-N 结果
    'top_n': 10,
    # 导向扩展使用的 top 特征数 (种子数)
    'guide_k': 3,
    # 终止条件
    'max_iter': 20,
    'r2_target': 0.999,
    'window_size': 5,
    'convergence_threshold': 1e-5,
    # 基础 expand 是否启用除法
    'enable_div': True,
    # 运行目录
    'path': os.getcwd(),
    # 'path': "yourfilepath",
}
# =================================================

path = CONFIG['path']

print("=" * 60)
print("  AI4S 实验 7.6.4: 迭代式针对性搜索")
print("=" * 60)
print(f"  策略: {CONFIG['strategy']}")
print(f"  SIS 保留: {CONFIG['sis_n']} 特征/轮")
print(f"  导向种子: top-{CONFIG['guide_k']}")
print(f"  最大迭代: {CONFIG['max_iter']}")
print("=" * 60)

# 读取数据
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))
print(f"\n原始数据: {data.shape[0]} 条, {data.shape[1]} 个特征")
print(f"特征名: {list(data.columns)}")

# 初始化
feature_tree = FeatureTree()
orig_ids = feature_tree.add_original_features(data, list(data.columns))
best_r2_history = []
all_results = []       # 所有轮的 top-N 结果
seen_names = set()     # 已见过的特征名（去重）
iteration = 1
should_stop = False
stop_reason = ""

# ==================== 迭代主循环 ====================
while not should_stop:
    print(f"\n{'—' * 40}")
    print(f"  第 {iteration} 轮迭代")
    print(f"{'—' * 40}")

    # ── Step 1: 特征扩展 ──
    if iteration == 1:
        # 第一轮: 基础扩展 (启用除法)
        data_expanded = expand_full(data, enable_binary_div=CONFIG['enable_div'])
        # 更新 seen_names
        for c in data_expanded.columns:
            seen_names.add(c.replace(' ', '').replace('**', '^'))
    else:
        # 后续轮: 导向性扩展
        prev_top = all_results[-1]  # 上一轮的结果
        top_nodes = []
        for fi in prev_top[:CONFIG['guide_k']]:
            node = FeatureNode(
                node_id=-1,
                name=fi.get_name(),
                data=fi.get_data_array(),
                r2=fi.get_r2(),
                loss=fi.get_loss(),
                coef=fi.get_coef(),
                iteration=iteration - 1
            )
            top_nodes.append(node)

        data_expanded = expand_guided(
            top_nodes=top_nodes,
            feature_tree=feature_tree,
            original_data=data,
            original_feature_ids=orig_ids,
            strategy=CONFIG['strategy'],
            enable_div=CONFIG['enable_div'],
            seen_names=seen_names
        )
        # 更新 seen_names
        for c in data_expanded.columns:
            seen_names.add(c.replace(' ', '').replace('**', '^'))

        if data_expanded.empty:
            print("  [警告] 导向扩展未产生新特征，终止迭代")
            stop_reason = "无法生成新特征"
            break

    n_expanded = data_expanded.shape[1] if not data_expanded.empty else 0
    print(f"  扩展后特征数: {n_expanded}")
    print(f"  累计已见特征: {len(seen_names)}")

    if n_expanded == 0:
        print("  [警告] 扩展后无有效特征，终止迭代")
        stop_reason = "无有效特征"
        break

    # ── Step 2: SIS 筛选 ──
    data_sis = sis(data_expanded, focus.to_numpy(), CONFIG['sis_n'])
    print(f"  SIS 筛选后: {data_sis.shape[1]} 特征")

    if data_sis.shape[1] == 0:
        print("  [警告] SIS 筛选后无特征，终止迭代")
        stop_reason = "SIS 筛选后为空"
        break

    # ── Step 3: 系数拟合 ──
    r2, coef, loss, times = fit(data_sis, focus.to_numpy(),
                                 method=CONFIG['fit_method'])

    # ── Step 4: 结果整理 ──
    results = sort_result(data_sis.to_numpy(),
                          data_sis.columns.to_numpy(),
                          r2, coef, loss, CONFIG['top_n'])

    all_results.append(results)
    best_r2 = results[0].get_r2()
    best_r2_history.append(best_r2)

    # ── Step 5: 存入树结构 ──
    for fi in results:
        feature_tree.add_node(
            name=fi.get_name(),
            data=fi.get_data_array(),
            r2=fi.get_r2(),
            loss=fi.get_loss(),
            coef=fi.get_coef(),
            parent_ids=orig_ids if iteration == 1 else [],
            operation=f'iter{iteration}',
            iteration=iteration
        )

    # ── 输出本轮 top-5 ──
    print(f"\n  本轮 top-5:")
    for rank, fi in enumerate(results[:5]):
        print(f"    {rank+1}. R²={fi.get_r2():.6f} | {fi.get_full_name()[:80]}")

    # ── Step 6: 终止判断 ──
    should_stop, stop_reason = stop_or_not(
        best_r2_history, iteration,
        max_iter=CONFIG['max_iter'],
        r2_target=CONFIG['r2_target'],
        window_size=CONFIG['window_size'],
        convergence_threshold=CONFIG['convergence_threshold']
    )
    print(f"\n  [{stop_reason}]")

    iteration += 1

# ==================== 最终输出 ====================
print(f"\n{'=' * 60}")
print(f"  迭代结束: {stop_reason}")
print(f"  总迭代轮数: {iteration - 1}")
print(f"  累计探索特征: {len(seen_names)}")
print(f"{'=' * 60}")

# 收集所有轮的结果，取全局最优
all_features = []
for round_results in all_results:
    all_features.extend(round_results)

# 按 R² 排序
all_features_sorted = sorted(all_features, key=lambda x: x.get_r2(), reverse=True)
best_feature = all_features_sorted[0]

print(f"\n  ★ 最优公式:")
print(f"    R² = {best_feature.get_r2():.6f}")
print(f"    Loss = {best_feature.get_loss():.6f}")
print(f"    公式: {best_feature.get_full_name()}")

# 输出全局 top-10 日志
print(f"\n  全局 Top-{min(10, len(all_features_sorted))}:")
log_lines = [['Rank', 'R²', 'Loss', 'Score', 'Formula']]
for rank, fi in enumerate(all_features_sorted[:10]):
    log_lines.append([
        rank + 1,
        f"{fi.get_r2():.6f}",
        f"{fi.get_loss():.6f}",
        f"{fi.get_score():.4f}",
        fi.get_full_name()[:100]
    ])
for line in log_lines:
    print('  {:<6}{:<14}{:<14}{:<14}{}'.format(*[str(x) for x in line]))

# ==================== 输出日志文件 ====================
log_path = os.path.join(path, 'log_iterative')
log_content = f"迭代式搜索日志\n{'=' * 60}\n"
log_content += f"策略: {CONFIG['strategy']}\n"
log_content += f"总迭代轮数: {iteration - 1}\n"
log_content += f"累计探索特征: {len(seen_names)}\n"
log_content += f"停止原因: {stop_reason}\n"
log_content += f"\nR² 收敛历史: {[round(r, 6) for r in best_r2_history]}\n"
log_content += f"\n最优公式:\n"
log_content += f"  R² = {best_feature.get_r2():.6f}\n"
log_content += f"  Loss = {best_feature.get_loss():.6f}\n"
log_content += f"  公式: {best_feature.get_full_name()}\n"

with open(log_path, 'w', encoding='utf-8') as f:
    f.write(log_content)
print(f"\n日志已保存到: {log_path}")

# ==================== 绘制拟合图 ====================
if best_feature is not None:
    plot_result(all_features_sorted[0], focus, path,
                title='iterative_layered')
    print(f"拟合图已保存到: {os.path.join(path, 'fit_iterative_layered.png')}")

print(f"\n{'=' * 60}")
print("  实验 7.6.4 完成!")
print(f"{'=' * 60}")