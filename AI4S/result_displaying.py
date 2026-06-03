from feature_information import FeatureInformation
import os
import time


def display_result(results: list[FeatureInformation], n: int, path,
                   exp_label: str = "",
                   exp_config: dict = None):
    '''
    输出结果文件，包含实验配置头信息。

    参数:
        results: 排序后的特征结果列表
        n: 输出前 n 个结果
        path: 保存路径
        exp_label: 实验标识（用于文件名，如 'exp_analytical_sis'）
        exp_config: 实验配置字典（key-value 对，写入日志头部）
    '''
    log = ''

    # ── 头部：实验配置 ──
    log += f"{'=' * 60}\n"
    log += f"  AI4S 实验日志\n"
    log += f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"{'=' * 60}\n"
    if exp_config:
        log += "\n[实验配置]\n"
        for k, v in exp_config.items():
            log += f"  {k}: {v}\n"
    log += f"\n[Top-{n} 结果]\n"
    log += f"{'=' * 60}\n"

    # ── 表头 ──
    log += '\n{:<5}{:<15}{:<15}{:<15}{}\n'.format(
        'id', 'r2', 'loss', 'score', 'model'
    )
    log += '-' * 120 + '\n'

    # ── 前 n 个结果 ──
    for ind in range(min(n, len(results))):
        result = results[ind]
        line = '{:<5}{:<15.6f}{:<15.6f}{:<15.4f}{}\n'.format(
            result.get_id(), result.get_r2(), result.get_loss(),
            result.get_score(), result.get_full_name()
        )
        log += line

    # 输出 log
    filename = f'log_{exp_label}' if exp_label else 'log'
    log_path = os.path.join(path, filename)
    with open(log_path, 'w', encoding='utf-8') as file:
        file.write(log)
    print(f"  日志已保存: {log_path}")