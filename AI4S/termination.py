"""
终止策略模块

实现三种终止条件:
1. 目标达成: R² > 阈值 → 立即停止
2. 滑动窗口收敛: 最近 N 轮内 R² 提升 < 阈值 → 已无提升空间
3. 最大迭代次数: 防止无限循环
"""


def stop_or_not(best_r2_history: list, iteration: int,
                max_iter: int = 20,
                r2_target: float = 0.999,
                window_size: int = 5,
                convergence_threshold: float = 1e-5) -> tuple:
    '''
    判断是否应该停止迭代

    参数:
        best_r2_history: 每轮迭代的最优 R² 列表
        iteration: 当前迭代轮数
        max_iter: 最大迭代轮数
        r2_target: 目标 R² 值，达到即停止
        window_size: 滑动窗口大小
        convergence_threshold: 收敛阈值 (窗口内 R² 最大-最小 < 此值视为收敛)

    返回:
        (should_stop: bool, reason: str)
    '''
    if not best_r2_history:
        return False, "尚未开始迭代"

    # 策略1: 目标达成 → 立即停止
    if best_r2_history[-1] >= r2_target:
        return True, f"R² = {best_r2_history[-1]:.6f} >= {r2_target}，目标公式已找到"

    # 策略2: 滑动窗口检测收敛
    if len(best_r2_history) >= window_size:
        recent = best_r2_history[-window_size:]
        # 检查窗口内是否有提升
        improvement = max(recent) - min(recent)
        if improvement < convergence_threshold:
            return True, (f"最近 {window_size} 轮 R² 波动 < {convergence_threshold}，"
                          f"已收敛 (best={best_r2_history[-1]:.6f})")

    # 策略3: 达最大迭代数
    if iteration >= max_iter:
        return True, f"达到最大迭代次数 {max_iter} (best R²={best_r2_history[-1]:.6f})"

    return False, (f"继续搜索 (iter={iteration}, "
                   f"best R²={best_r2_history[-1]:.6f})")


def check_early_stop(best_r2_history: list, patience: int = 5,
                     min_improvement: float = 1e-4) -> bool:
    '''
    早停检查：最近 patience 轮内最佳 R² 是否持续未提升超过 min_improvement

    返回: True 表示应该早停
    '''
    if len(best_r2_history) < patience:
        return False

    best_so_far = max(best_r2_history[:-patience], default=0)
    recent_best = max(best_r2_history[-patience:])

    return (recent_best - best_so_far) < min_improvement