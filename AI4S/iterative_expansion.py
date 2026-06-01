"""
导向性扩展模块 — 实现"特征生成-筛选-评估-特征生成"循环

三种引导策略:
1. 替换策略: 用 top-K 高分特征替换原始输入再做完整 expand
2. 交叉策略: 高分特征与原始特征的笛卡尔积（保留物理背景）
3. 分层策略 (推荐): 原始特征基础 expand + 高分特征定向组合
"""

import pandas as pd
import numpy as np
import itertools
from numpy.typing import NDArray as ND
from result_storage import FeatureNode, FeatureTree


def expand_guided(top_nodes: list, feature_tree: FeatureTree,
                  original_data: pd.DataFrame,
                  original_feature_ids: list,
                  strategy: str = 'layered',
                  enable_div: bool = True,
                  seen_names: set = None) -> pd.DataFrame:
    '''
    导向性扩展: 基于上轮 top-K 高分特征指导下轮扩展

    参数:
        top_nodes: 上轮 top-K 的 FeatureNode 列表（已排序，最佳在前）
        feature_tree: 特征推导树
        original_data: 原始 5 特征数据
        original_feature_ids: 原始特征在 tree 中的节点 ID 列表
        strategy: 'replace' | 'crossover' | 'layered'
        enable_div: 是否包含除法
        seen_names: 已见过的特征名称集合（避免重复生成）

    返回: 导向性扩展后的 DataFrame
    '''
    if seen_names is None:
        seen_names = set()

    # 获取原始特征的名称和数据
    orig_cols = original_data.columns.tolist()
    orig_np = original_data.to_numpy()

    # 获取 top 节点的表达式名称和数据
    top_names = [n.name for n in top_nodes]
    top_data_list = []
    for n in top_nodes:
        if n.data is not None:
            d = n.data
            if len(d.shape) == 1:
                d = d.reshape(-1, 1)
            top_data_list.append(d)

    if strategy == 'replace':
        # 策略1: 用高分特征替换原始输入，做完整 expand
        return _expand_replace(top_nodes, top_data_list, top_names,
                               original_data, seen_names)

    elif strategy == 'crossover':
        # 策略2: 高分特征 × 原始特征的幂次变体
        return _expand_crossover(top_nodes, top_data_list, top_names,
                                 orig_np, orig_cols, enable_div, seen_names)

    elif strategy == 'layered':
        # 策略3 (推荐): 基础 expand + 高分特征定向组合
        return _expand_layered(top_nodes, top_data_list, top_names,
                               original_data, orig_np, orig_cols,
                               enable_div, seen_names)

    else:
        raise ValueError(f"未知策略: {strategy}，可选: replace/crossover/layered")


def _expand_replace(top_nodes, top_data_list, top_names,
                    orig_data, seen_names):
    """
    替换策略: 将 top-K 高分特征作为新的"原始特征"做完整 expand
    优点: 简单直接
    缺点: 丢失原始物理特征，可解释性下降
    """
    from expansion import expand as full_expand

    # 构建新的 dataframe
    if top_data_list:
        combined_data = np.hstack(top_data_list)
    else:
        combined_data = orig_data.to_numpy()

    new_df = pd.DataFrame(combined_data, columns=top_names)
    # 做一次完整 expand (不显示 verbose 避免重复输出)
    expanded = full_expand(new_df, enable_binary_div=enable_div,
                           enable_ternary_mul=False, verbose=False)

    # 过滤已见过的特征
    new_cols = [c for c in expanded.columns
                if _simplify(c) not in seen_names]
    return expanded[new_cols] if new_cols else pd.DataFrame()


def _expand_crossover(top_nodes, top_data_list, top_names,
                      orig_np, orig_cols, enable_div, seen_names):
    """
    交叉策略: 高分特征 × 原始特征的幂次变体（乘法和除法）
    优点: 保留物理背景，计算量可控
    """
    n_samples = orig_np.shape[0]

    # 为原始特征生成幂次变体
    orig_variants_data = []
    orig_variants_names = []
    powers = [0.5, 2, 1/3, 3, -1]
    for j, col_name in enumerate(orig_cols):
        col = orig_np[:, j:j+1].astype(np.float64)
        # 自身
        orig_variants_data.append(col)
        orig_variants_names.append(col_name)
        # 幂次
        for p in powers:
            orig_variants_data.append(np.power(col, p))
            orig_variants_names.append(f"({col_name}^{p})")

    all_new_num = []
    all_new_str = []

    # 高分特征 × 原始变体 (乘法)
    for ti, tdata in enumerate(top_data_list):
        tname = top_names[ti]
        for vj in range(len(orig_variants_data)):
            # 乘法
            prod = tdata * orig_variants_data[vj]
            name = f"({tname}*{orig_variants_names[vj]})"
            if _simplify(name) not in seen_names:
                all_new_num.append(prod)
                all_new_str.append(name)

            # 除法
            if enable_div:
                eps = 1e-15
                # 高分特征 / 原始变体
                denom = np.where(np.abs(orig_variants_data[vj]) < eps,
                                 eps, orig_variants_data[vj])
                div1 = tdata / denom
                name1 = f"({tname}/{orig_variants_names[vj]})"
                if _simplify(name1) not in seen_names:
                    all_new_num.append(div1)
                    all_new_str.append(name1)
                # 原始变体 / 高分特征
                denom2 = np.where(np.abs(tdata) < eps, eps, tdata)
                div2 = orig_variants_data[vj] / denom2
                name2 = f"({orig_variants_names[vj]}/{tname})"
                if _simplify(name2) not in seen_names:
                    all_new_num.append(div2)
                    all_new_str.append(name2)

    if not all_new_num:
        return pd.DataFrame()

    out_num = np.hstack(all_new_num)
    out_data = pd.DataFrame(out_num, columns=all_new_str)
    out_data = out_data.loc[:, ~(out_data.isna().any())]
    out_data = out_data.loc[:, ~out_data.apply(np.isinf).any()]
    return out_data


def _expand_layered(top_nodes, top_data_list, top_names,
                    orig_data, orig_np, orig_cols,
                    enable_div, seen_names):
    """
    分层策略 (推荐): 原始特征基础 expand + 高分特征与原始特征交叉
    兼顾物理可解释性和搜索效率
    """
    from expansion import expand as full_expand

    # Layer 1: 原始特征的基础扩展 (保守策略)
    expanded_base = full_expand(orig_data, enable_binary_div=enable_div,
                                enable_ternary_mul=False, verbose=False)
    base_cols = [c for c in expanded_base.columns
                 if _simplify(c) not in seen_names]
    if base_cols:
        expanded_base = expanded_base[base_cols]
    else:
        expanded_base = pd.DataFrame()

    # Layer 2: 高分特征与原始特征变体的交叉
    expanded_cross = _expand_crossover(
        top_nodes, top_data_list, top_names,
        orig_np, orig_cols, enable_div, seen_names
    )

    # 合并
    if expanded_base.empty and expanded_cross.empty:
        return pd.DataFrame()
    elif expanded_base.empty:
        return expanded_cross
    elif expanded_cross.empty:
        return expanded_base
    else:
        combined = pd.concat([expanded_base, expanded_cross], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined


def _simplify(name: str) -> str:
    """标准化特征名称"""
    return name.replace(' ', '').replace('**', '^')