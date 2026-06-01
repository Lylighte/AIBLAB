import pandas as pd
import numpy as np
import itertools
from numpy.typing import NDArray as ND


def expand(ori_data: pd.DataFrame):
    '''
    对数据进行扩展（原始版本，一字未改）
    '''
    ## 初始化
    out_data = ori_data.copy()
    out_num = ori_data.to_numpy()
    out_str = ori_data.columns.to_numpy()

    ## 变换：[x1, x2, x3, ...] -> [[x1], [x2], [x3],...]
    out_num = np.hsplit(out_num, out_num.shape[1])
    out_str = np.hsplit(out_str, out_str.shape[0])

    ## 幂
    for index, name in enumerate(out_str):

        num_p1, str_p1 = power(out_num[index], name, 1/2)
        num_p2, str_p2 = power(out_num[index], name, 2)
        num_p3, str_p3 = power(out_num[index], name, 1/3)
        num_p4, str_p4 = power(out_num[index], name, 3)
        out_num[index] = np.hstack((out_num[index], num_p1,
                                    num_p2, num_p3, num_p4))
        out_str[index] = np.hstack((out_str[index], str_p1,
                                    str_p2, str_p3, str_p4))

        num_b, str_b = power(out_num[index], out_str[index], -1)
        out_num[index] = np.hstack((out_num[index], num_b))
        out_str[index] = np.hstack((out_str[index], str_b))

    ## 组合
    # 返回所有组合的可能
    combinations = list(itertools.combinations(range(len(out_num)), 2))
    # 对每一种组合进行计算
    for combination in combinations:
        # 取出一个组合中对应特征的索引
        idx1, idx2 = combination
        # 计算组合特征
        num_c, str_c = combine(out_num[idx1], out_num[idx2],
                               out_str[idx1], out_str[idx2])
        # 将组合特征添加到结果中
        out_num.append(num_c)
        out_str.append(str_c)

    ## 整合数据
    out_num = np.hstack(out_num)
    out_str = np.hstack(out_str)
    out_data = pd.DataFrame(out_num, columns=out_str)

    ## 删除异常数据
    out_data = out_data.loc[:, ~(out_data.isna().any() | \
                                  out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    return out_data

def power(in_num: ND, in_name: ND, p):
    '''
    对数据进行幂运算（原始版本，一字未改）
    '''
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)

    out_num = np.power(in_num, p)
    out_name = [f"({n}**{p})" for n in in_name]

    return out_num, out_name

def combine(in_num1: ND, in_num2: ND, in_name1: ND, in_name2: ND):
    '''
    对数据进行二元组合运算（原始版本，一字未改）
    '''
    if len(in_num1.shape) == 1:
        in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1:
        in_num2 = in_num2.reshape(-1, 1)

    # 组合特征的数量
    num_datapoints = in_num1.shape[0]
    num_featurevars = in_num1.shape[1] * in_num2.shape[1]

    # 初始化组合特征矩阵和名称列表
    out_num = np.zeros((num_datapoints, num_featurevars))
    out_name = ['']*num_featurevars

    # 计算组合特征
    idx = 0
    for i in range(in_num1.shape[1]):
        for j in range(in_num2.shape[1]):
            out_num[:, idx] = in_num1[:, i] * in_num2[:, j]
            out_name[idx] = f"({in_name1[i]}*{in_name2[j]})"
            idx += 1

    return out_num, out_name


# ════════════════════════════════════════════════
# 以下为 7.6.3 / 7.6.4 新增函数，不影响原版
# ════════════════════════════════════════════════

def expand_full(ori_data: pd.DataFrame,
                enable_binary_div: bool = True,
                enable_ternary_mul: bool = False,
                enable_ternary_mixed: bool = False,
                enable_log: bool = False,
                enable_exp: bool = False,
                extra_powers: list = None):
    '''
    扩展版 expand = 原版 expand + 新增可选运算
    '''
    out_data = expand(ori_data)

    out_num = out_data.to_numpy()
    out_str = out_data.columns.to_numpy()
    out_num = list(np.hsplit(out_num, out_num.shape[1]))
    out_str = list(np.hsplit(out_str, out_str.shape[0]))

    n_initial = ori_data.shape[1]
    n_after_basic = sum(len(o) for o in out_num)

    if extra_powers:
        for index, name in enumerate(out_str):
            for p in extra_powers:
                num_p, str_p = power(out_num[index], name, p)
                out_num[index] = np.hstack((out_num[index], num_p))
                out_str[index] = np.hstack((out_str[index], str_p))

    if enable_log:
        for index, name in enumerate(out_str):
            num_l, str_l = log_transform(out_num[index], name)
            out_num[index] = np.hstack((out_num[index], num_l))
            out_str[index] = np.hstack((out_str[index], str_l))

    if enable_exp:
        for index, name in enumerate(out_str):
            num_e, str_e = exp_transform(out_num[index], name)
            out_num[index] = np.hstack((out_num[index], num_e))
            out_str[index] = np.hstack((out_str[index], str_e))

    if enable_binary_div:
        combos = list(itertools.combinations(range(len(out_num)), 2))
        for idx1, idx2 in combos:
            num_c, str_c = combine_divide(out_num[idx1], out_num[idx2],
                                           out_str[idx1], out_str[idx2])
            out_num.append(num_c)
            out_str.append(str_c)

    if enable_ternary_mul or enable_ternary_mixed:
        # 三元只对前 n_initial 组做，控制规模
        combos_3 = list(itertools.combinations(range(n_initial), 3))
        if enable_ternary_mul:
            for idx1, idx2, idx3 in combos_3:
                num_c, str_c = combine_3(
                    out_num[idx1], out_num[idx2], out_num[idx3],
                    out_str[idx1], out_str[idx2], out_str[idx3])
                out_num.append(num_c)
                out_str.append(str_c)
        if enable_ternary_mixed:
            for idx1, idx2, idx3 in combos_3:
                num_c, str_c = combine_3_mixed(
                    out_num[idx1], out_num[idx2], out_num[idx3],
                    out_str[idx1], out_str[idx2], out_str[idx3])
                out_num.append(num_c)
                out_str.append(str_c)

    out_num_arr = np.hstack(out_num)
    out_str_arr = np.hstack(out_str)
    out_data_full = pd.DataFrame(out_num_arr, columns=out_str_arr)
    out_data_full = out_data_full.loc[:, ~(out_data_full.isna().any())]
    out_data_full = out_data_full.loc[:, ~out_data_full.apply(np.isinf).any()]
    out_data_full = out_data_full.loc[:, ~out_data_full.columns.duplicated()]

    print(f"\n[expand_full] 基础expand: {n_after_basic} → 最终: {out_data_full.shape[1]} ({out_data_full.shape[1]/n_initial:.0f}x)")

    return out_data_full


# ── 一元 ──

def log_transform(in_num: ND, in_name: ND, eps: float = 1e-10):
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)
    out_num = np.log(np.abs(in_num.astype(np.float64)) + eps)
    out_name = [f"log(|{n}|)" for n in in_name.flatten()]
    return out_num, out_name


def exp_transform(in_num: ND, in_name: ND, clip_val: float = 5.0):
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)
    out_num = np.exp(np.clip(in_num.astype(np.float64), -clip_val, clip_val))
    out_name = [f"exp({n})" for n in in_name.flatten()]
    return out_num, out_name


# ── 二元 ──

def combine_divide(in_num1: ND, in_num2: ND, in_name1: ND, in_name2: ND):
    if len(in_num1.shape) == 1:
        in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1:
        in_num2 = in_num2.reshape(-1, 1)
    n1 = in_name1.flatten()
    n2 = in_name2.flatten()
    n, nf = in_num1.shape[0], in_num1.shape[1] * in_num2.shape[1] * 2
    out_num = np.zeros((n, nf))
    out_name = [''] * nf
    eps, idx = 1e-15, 0
    for i in range(in_num1.shape[1]):
        for j in range(in_num2.shape[1]):
            d = np.where(np.abs(in_num2[:, j]) < eps, eps, in_num2[:, j])
            out_num[:, idx] = in_num1[:, i] / d
            out_name[idx] = f"({n1[i]}/{n2[j]})"
            idx += 1
            d = np.where(np.abs(in_num1[:, i]) < eps, eps, in_num1[:, i])
            out_num[:, idx] = in_num2[:, j] / d
            out_name[idx] = f"({n2[j]}/{n1[i]})"
            idx += 1
    return out_num, out_name


# ── 三元 ──

def combine_3(in_num1: ND, in_num2: ND, in_num3: ND,
              in_name1: ND, in_name2: ND, in_name3: ND):
    if len(in_num1.shape) == 1:
        in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1:
        in_num2 = in_num2.reshape(-1, 1)
    if len(in_num3.shape) == 1:
        in_num3 = in_num3.reshape(-1, 1)
    n1 = in_name1.flatten()
    n2 = in_name2.flatten()
    n3 = in_name3.flatten()
    n, nf = in_num1.shape[0], in_num1.shape[1] * in_num2.shape[1] * in_num3.shape[1]
    out_num = np.zeros((n, nf))
    out_name = [''] * nf
    idx = 0
    for i in range(in_num1.shape[1]):
        for j in range(in_num2.shape[1]):
            for k in range(in_num3.shape[1]):
                out_num[:, idx] = in_num1[:, i] * in_num2[:, j] * in_num3[:, k]
                out_name[idx] = f"({n1[i]}*{n2[j]}*{n3[k]})"
                idx += 1
    return out_num, out_name


def combine_3_mixed(in_num1: ND, in_num2: ND, in_num3: ND,
                    in_name1: ND, in_name2: ND, in_name3: ND):
    if len(in_num1.shape) == 1:
        in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1:
        in_num2 = in_num2.reshape(-1, 1)
    if len(in_num3.shape) == 1:
        in_num3 = in_num3.reshape(-1, 1)
    n1 = in_name1.flatten()
    n2 = in_name2.flatten()
    n3 = in_name3.flatten()
    n, nf = in_num1.shape[0], in_num1.shape[1] * in_num2.shape[1] * in_num3.shape[1] * 3
    out_num = np.zeros((n, nf))
    out_name = [''] * nf
    eps, idx = 1e-15, 0
    for i in range(in_num1.shape[1]):
        for j in range(in_num2.shape[1]):
            for k in range(in_num3.shape[1]):
                d = np.where(np.abs(in_num3[:, k]) < eps, eps, in_num3[:, k])
                out_num[:, idx] = in_num1[:, i] * in_num2[:, j] / d
                out_name[idx] = f"({n1[i]}*{n2[j]}/{n3[k]})"
                idx += 1
                d = np.where(np.abs(in_num2[:, j] * in_num3[:, k]) < eps,
                             eps, in_num2[:, j] * in_num3[:, k])
                out_num[:, idx] = in_num1[:, i] / d
                out_name[idx] = f"({n1[i]}/({n2[j]}*{n3[k]}))"
                idx += 1
                d = np.where(np.abs(in_num2[:, j]) < eps, eps, in_num2[:, j])
                out_num[:, idx] = (in_num1[:, i] / d) * in_num3[:, k]
                out_name[idx] = f"({n1[i]}/{n2[j]}*{n3[k]})"
                idx += 1
    return out_num, out_name