# AI4S 实验代码变更说明

> 会话日期: 2026-06-01
> 基于已有 SISSO 基础框架，按实验手册第 7 章要求完成 7.6.2 / 7.6.3 / 7.6.4 三个实验的代码实现。

---

## 一、文件变更总览

| 文件 | 操作 | 说明 |
|---|---|---|
| `main.py` | **修改** | 添加实验变体开关 (FIT_METHOD, USE_SIS, SIS_N, NN 超参) |
| `coefficients_fitting.py` | **修改** | 新增 Timer 计时、暴露 NN 超参数、返回 time_out |
| `expansion.py` | **重写** | 重构特征扩展模块：除法/三元/对数/指数 + 空间膨胀分析 |
| `feature_information.py` | **修改** | 新增 `get_coef()`, `get_data_array()` 公开接口 |
| `iterative_expansion.py` | **新增** | 7.6.4 导向性扩展模块 (3 种策略) |
| `result_storage.py` | **新增** | 7.6.4 树状结果存储 (FeatureTree + FeatureNode) |
| `termination.py` | **新增** | 7.6.4 终止策略 (目标达成/滑动窗口/最大迭代) |
| `main_iterative.py` | **新增** | 7.6.4 迭代式搜索主程序 |

### 未修改文件
`sis_ana.py`, `result_sorting.py`, `result_displaying.py`, `result_plotting.py`, `simplify_str.py`, `timer.py`, `test.pbs`, `data.csv`, `focus.csv`

---

## 二、实验 7.6.2 — 解析解 vs 神经网络求解对比

### 算法逻辑

**解析解 (Analytical Solution)**
- 原理：$$c' = (X'^T X')^{-1} X'^T y$$，由最大似然估计推导
- 计算瓶颈：矩阵求逆 $$O(d^3)$$（d 为特征维度）
- 适用场景：小特征空间 (d < 1000)
- 代码：`coefficients_fitting.py` → `analytical_solving()`

**神经网络解 (Neural Network Solution)**
- 原理：单层全连接网络 + SGD 最小化 MSE Loss
- 计算瓶颈：$$O(n \cdot d \cdot epochs)$$
- 适用场景：大特征空间，避免矩阵求逆
- 代码：`coefficients_fitting.py` → `net_solving()`

### 实现方式

**main.py 配置开关：**
```python
FIT_METHOD = 'analytical'  # 'analytical' | 'nn'
USE_SIS = True              # SIS 筛选开关
SIS_N = 10                  # SIS 保留特征数
NN_EPOCHS = 40              # NN 超参数
NN_BATCH_SIZE = 100
NN_LR = 0.1
```

**运行步骤：**
1. 先运行 `FIT_METHOD='analytical', USE_SIS=True` → baseline 日志 (记作 log_ana_sis)
2. 切换 `FIT_METHOD='nn'`，调整 NN 超参直到 R² 收敛到解析解水平
3. 注释 SIS (`USE_SIS=False`)，分别测试两种方法的性能变化

### 结果分析要点
- SIS 开启时 (d≈10)：解析解远快于 NN
- SIS 关闭时 (d≈数千)：NN 避免矩阵求逆，优势明显
- 关键发现：特征空间小时解析解更优，大时 NN 更可扩展 — 两者互补

---

## 三、实验 7.6.3 — 重构特征扩展模块

### 算法逻辑

基础框架仅支持 `x^p` + `a×b`，无法产生目标方程 $$q = a \frac{\alpha^{1/3} \gamma \Omega}{RkT} + b$$ 中的：
- **三元乘积** `α^(1/3) × γ × Ω` ❌
- **除法结构** `÷ (R × kT)` ❌

**解决方法：逐步添加运算，量化每次空间膨胀**

### 实现方式

`expansion.py` 的 `expand()` 函数新增控制参数：

```python
def expand(ori_data,
           enable_power=True,        # 单变量幂次 (1/2, 2, 1/3, 3, -1)
           enable_binary_mul=True,   # 二元乘法 a*b
           enable_binary_div=False,  # 二元除法 a/b, b/a
           enable_ternary_mul=False, # 三元乘法 a*b*c
           enable_ternary_mixed=False,# 三元混合 a*b/c, a/(b*c), (a/b)*c
           enable_log=False,         # 自然对数 log(|x|+ε)
           enable_exp=False,         # 指数 exp(clip(x,5))
           extra_powers=None)        # 额外幂次 [-2, -0.5, -2/3 等]
```

**新运算函数：**
| 函数 | 运算 | 产生数量 |
|---|---|---|
| `power()` | x^p | 每特征 +1 |
| `log_transform()` | log(|x|+ε) | 每特征 +1 |
| `exp_transform()` | exp(clip(x)) | 每特征 +1 |
| `combine()` | a×b | C(n,2) × 1 |
| `combine_divide()` | a/b, b/a | C(n,2) × 2 |
| `combine_3()` | a×b×c | C(n,3) × 1 |
| `combine_3_mixed()` | a×b/c, a/(b×c), a/b×c | C(n,3) × 3 |

**空间膨胀自动输出 (verbose=True)：**
```
========== 特征空间膨胀分析 ==========
  初始特征: 5 特征
  单变量幂次(含[0.5, 2, 0.333..., 3]次): 30 特征
  一元变换后总计: 30 特征
  二元乘法 a*b: 465 特征
  二元除法 a/b, b/a: 1095 特征
  去重/去异常后: 1080 特征
  总膨胀倍率: 216.0x
========================================
```

### 汇报关键数据 (需在 WSL 运行后填入)

| 步骤 | 运算 | 特征数 | 增长倍率 | 能否包含目标? |
|---|---|---|---|---|
| Step 0 (baseline) | 幂次+二元乘 | TBD | 1× | ❌ |
| +二元除 | a/b, b/a | TBD | TBD× | ❌ (还需三元) |
| +三元乘 | a×b×c | TBD | TBD× | ✅ |
| +三元混合 | a×b/c 等 | TBD | TBD× | ✅ |
| +对数/指数 | log, exp | TBD | TBD× | ✅ (冗余大) |

---

## 四、实验 7.6.4 — 迭代式针对性搜索 ⭐

### 算法设计逻辑

**核心思路：** 从"一次性盲目全空间搜索"变为"逐步聚焦"——
用上一轮的高分特征指导下轮扩展方向。

**算法框架 (四模块)：**

```
┌─────────────────────────────────────────────────────┐
│                   Iterative SISSO                   │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 特征扩展  │───▶│ SIS 筛选  │───▶│ 系数拟合  │      │
│  │ (expand) │    │  (sis)   │    │  (fit)   │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       ▲                               │             │
│       │         ┌──────────┐          │             │
│       └─────────│ 引导策略  │◀─────────┘             │
│   下轮扩展种子  │ (guide)   │  本轮 top-K 高分特征    │
│                 └──────────┘                        │
│                       │                             │
│                       ▼                             │
│                 ┌──────────┐                        │
│                 │ 终止判断  │                        │
│                 │ (stop?)  │                        │
│                 └──────────┘                        │
└─────────────────────────────────────────────────────┘
```

### 模块 1: 导向性扩展 `iterative_expansion.py`

**三种策略 (可切换对比)：**

| 策略 | 配置值 | 逻辑 | 优劣 |
|---|---|---|---|
| 替换策略 | `'replace'` | top-K 替换原始输入，重新全量 expand | 简单但丢失物理含义 |
| 交叉策略 | `'crossover'` | top-K × 原始特征变体 (乘+除) | 保留物理背景 |
| **分层策略** | `'layered'` | 原始特征基础expand + top-K交叉组合 | 兼顾效率与可解释性 |

**默认使用分层策略** — 每轮同时做：(1)原始特征的基础扩展 (2)top-3 种子与原始特征变体的交叉 (乘+除)

### 模块 2: 树状结果存储 `result_storage.py`

**FeatureTree 类：**
- `FeatureNode`: 存储特征名、数据、R²、loss、系数、父节点ID、生成操作
- `FeatureTree`: 管理全量节点，支持去重（保留 R² 更高版本）
- `trace_path()`: 从最优节点追溯到原始特征 → **推导路径可解释**
- `get_derivation_tree_str()`: 生成推导树的文本表示

**汇报可展示：**
```
原始特征 → α^(1/3) → α^(1/3)×γ×Ω → α^(1/3)×γ×Ω/(R×kT)
```

### 模块 3: 终止策略 `termination.py`

```python
stop_or_not(best_r2_history, iteration,
            max_iter=20,           # 最大迭代
            r2_target=0.999,       # 目标达成
            window_size=5,         # 滑动窗口大小
            convergence_threshold=1e-5)  # 收敛阈值
```

三种条件任一满足即停止：
1. R² ≥ 0.999 → "目标公式已找到"
2. 最近 N 轮 R² 波动 < 阈值 → "已收敛"
3. 迭代轮数 ≥ max_iter → "达到最大迭代次数"

### 模块 4 (可选): 物理约束过滤

- Ω [L³] | γ [E/L²] | R [L] | α [无量纲] | kT [E]
- 目标 q 无量纲 → 只保留量纲匹配的组合
- 例：`Ω/(R³)` = [L³/L³] = 无量纲 ✅，`Ω+R` = [L³]+[L] ❌

### 运行方式

```bash
# 基础实验 (7.6.2 / 7.6.3)
python main.py

# 迭代式搜索 (7.6.4)
python main_iterative.py
```

**配置修改：** 编辑 `main_iterative.py` 中 CONFIG 字典即可切换策略。

---

## 五、汇报准备材料清单

### 必备图表 (需运行后生成)
- [ ] 算法框架示意图 → 见上方 ASCII 图，可用 draw.io/PPT 美化
- [ ] 空间膨胀对比表 → 运行 7.6.3 各 step 后填入数据
- [ ] 三种方法对比表 → 基础 vs 全空间 vs 迭代 (R²/特征数/效率)
- [ ] R² 收敛曲线 → 从 `best_r2_history` 绘制
- [ ] 推导路径树 → 从 `feature_tree.trace_path()` 获取

### 推荐汇报主线 (10-15 分钟)
```
1. 背景 (1min): 符号回归 + 纳米粒子 OR 熟化 + 目标方程
2. 7.6.2 (2min): 解析 vs NN 原理 → 关键结果 → 小/大特征空间各适用哪个
3. 7.6.3 (3min): 基础框架为什么失败 → 逐步加运算 → 空间膨胀表 → 代价
4. 7.6.4 (4-5min): ⭐ 重点
   - 四模块设计逻辑 (导向扩展/树状存储/终止策略)
   - 算法框架图
   - 推导路径展示
   - 与 7.6.3 的对比 (效率+可解释性)
5. 思考题 (2min): 物理约束/量纲分析/大模型整合
```