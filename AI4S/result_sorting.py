from simplify_str import simplify
from feature_information import FeatureInformation
import pandas as pd
import numpy as np
from numpy.typing import NDArray as ND


def sort_result(num_in: ND, str_in: ND, r2: list, p: list,
                loss: list, n: int, num=0):
    '''
    根据 R² 排序，只取 top-n 结果。
    优化: 避免全量重排大数组，只提取 top-n 列。
    '''
    # 简化字符串 (只简化前 n*10 个候选即可，但 top-n 未知，先取 top-10n 候选)
    # R² 排序（仅排索引，不操作大数组）
    sortlist = sorted(enumerate(r2), key=lambda x: x[1], reverse=True)
    top_k = max(n * 10, 100)  # 候选池：至少 100 个
    idx_top = [x[0] for x in sortlist[:top_k]]

    # 只对 top-k 列做字符串简化
    str_sp = np.array([simplify(str_in[i]) for i in idx_top])

    # 只提取 top-k 列数据（避免全量重排）
    num_in_top = num_in[:, idx_top]
    r2_top = [r2[i] for i in idx_top]
    p_top = [p[i] for i in idx_top]
    loss_top = [loss[i] for i in idx_top]

    # 再按 R² 排一次（候选池内部）
    second_sort = sorted(enumerate(r2_top), key=lambda x: x[1], reverse=True)
    idx_final = [x[0] for x in second_sort[:n]]

    list_out = []
    for rank, orig_idx in enumerate(idx_final):
        fi = FeatureInformation(
            rank + 1 + num,
            pd.DataFrame(num_in_top[:, orig_idx], columns=[str_sp[orig_idx]]),
            p_top[orig_idx], r2_top[orig_idx], loss_top[orig_idx]
        )
        fi.update_score()
        list_out.append(fi)

    return list_out