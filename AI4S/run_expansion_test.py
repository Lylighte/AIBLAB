"""
7.6.3 空间膨胀测量脚本
逐步加运算 → 记录特征数、耗时、最优 R²
"""
from expansion import expand, expand_full
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit
from timer import Timer
import pandas as pd
import os

path = os.getcwd()
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))
n0 = data.shape[1]  # 5

report = []
report.append(f"{'='*70}")
report.append(f"  7.6.3 特征空间膨胀分析报告")
report.append(f"  {os.path.join(path, 'data.csv')} — 原始特征: {n0}, 数据量: {data.shape[0]}")
report.append(f"  RAM 上限参考: 32 GB")
report.append(f"{'='*70}")

# ── 配置列表 ──
configs = [
    ("baseline",           "幂次+二元乘(原版 expand)",     False, False, False, False, False, False),
    ("+binary_div",        "+二元除法 a/b, b/a",            True,  False, False, False, False, False),
    ("+ternary_mul",       "+三元乘法 a*b*c",               True,  True,  False, False, False, False),
    ("+ternary_mixed",     "+三元混合 a*b/c, a/(b*c),...", True,  True,  True,  False, False, False),
    ("+log_exp",           "+对数/指数 log, exp",            True,  True,  True,  True,  True,  False),
]

for label, desc, bin_div, tern_mul, tern_mix, do_log, do_exp, _ in configs:
    print(f"\n{'─'*60}")
    print(f"  运行: {label} — {desc}")
    print(f"{'─'*60}")

    ti = Timer()
    ti.start()

    try:
        data_expanded = expand_full(data,
            enable_binary_div=bin_div,
            enable_ternary_mul=tern_mul,
            enable_ternary_mixed=tern_mix,
            enable_log=do_log,
            enable_exp=do_exp,
        )
        elapsed_expand = ti.stop()
        n_feat = data_expanded.shape[1]

        # SIS 保留 10 个
        ti.start()
        data_selected = sis(data_expanded, focus.to_numpy(), 10)
        elapsed_sis = ti.stop()

        # 解析解拟合
        ti.start()
        r2, coef, loss, times = fit(data_selected, focus.to_numpy(), method='analytical')
        elapsed_fit = ti.stop()

        # 排序
        results = sort_result(data_selected.to_numpy(),
                              data_selected.columns.to_numpy(),
                              r2, coef, loss, 10)

        top_r2 = results[0].get_r2()
        top_formula = results[0].get_full_name()

        row = f"[{label}] 特征={n_feat}({n_feat/n0:.0f}x) | 扩展={elapsed_expand:.1f}s | SIS={elapsed_sis:.1f}s | 拟合={elapsed_fit:.1f}s | R²={top_r2:.4f} | {top_formula[:100]}"
        print(f"  >>> {row}")

    except MemoryError:
        n_theory = n0  # placeholder
        row = f"[{label}] ❌ 内存不足 (参考 32GB)，理论预估: ~{n_theory:,} 特征"
        print(f"  >>> {row}")
    except Exception as e:
        row = f"[{label}] ❌ 错误: {e}"
        print(f"  >>> {row}")

    report.append(row)

# ── 理论推算: 三元扩展到全体C(1050,3)的情况 ──
report.append("")
report.append(f"{'─'*70}")
report.append(f"[理论] 三元扩展全体 C(1050,3) 推演 (参考 32GB)")
# C(1050, 3) = 1050*1049*1048/6
c1050_3 = 1050 * 1049 * 1048 // 6
# 如果每个组合产生 count_3 个新特征 (三元乘1 + 三元混合3 = 4)
# 但实际代码中对前5个原始特征做三元，这里仅做理论推算
for factor in [1, 4]:
    total = c1050_3 * factor + 1102500
    mem_gb = 973 * total * 8 / (1024**3)
    status = "✅ 安全" if mem_gb < 32 else f"❌ 超出 ({mem_gb:.1f}GB > 32GB)"
    report.append(f"    三元{factor}运算: {total:,} 特征 ({total/n0:,.0f}x), 内存 ~{mem_gb:.1f}GB — {status}")

report.append("")
report.append(f"{'='*70}")
report.append("  报告结束")
report.append(f"{'='*70}")

# 保存
report_path = os.path.join(path, "expansion_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\n报告已保存: {report_path}")