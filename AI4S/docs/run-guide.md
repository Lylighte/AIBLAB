# AI4S 实验运行指南

## 环境要求

```bash
# WSL conda 环境 (ai4science)
conda activate ai4science
pip list | grep -E "torch|numpy|sympy|matplotlib|pandas"
# 应包含: torch==1.12.0, numpy==1.26.4, sympy==1.13.1, matplotlib, pandas
```

## 运行前检查

```bash
cd /path/to/AI4S/
ls *.py *.csv  # 确认所有文件齐全
```

---

## 实验 7.6.2: 解析解 vs 神经网络

### Step 1: 运行 baseline (解析解 + SIS 开启)
```bash
# 编辑 main.py 确保:
#   FIT_METHOD = 'analytical'
#   USE_SIS = True
#   SIS_N = 10
python main.py
# 输出: log 文件 + fit.png
# 记录 top-1 的 R² 和耗时
```

### Step 2: 切换到神经网络求解
```bash
# 编辑 main.py:
#   FIT_METHOD = 'nn'
#   NN_EPOCHS = 100    # 调大确保收敛
#   NN_LR = 0.01       # 可能需要微调
python main.py
# 对比 NN 与解析解的 R²，直到差距 < 0.001
```

### Step 3: 注释 SIS (全量特征对比)
```bash
# 编辑 main.py:
#   USE_SIS = False
#   FIT_METHOD = 'analytical'  # 先测试解析解
python main.py
# 记录耗时 → 

# 然后切换:
#   FIT_METHOD = 'nn'
python main.py
# 记录耗时 → 
```

---

## 实验 7.6.3: 重构特征扩展模块

### 手动逐步测试空间膨胀
```bash
python3 -c "
from expansion import expand
import pandas as pd
data = pd.read_csv('data.csv')

# Step 0: baseline (幂次+二元乘)
r0 = expand(data, enable_binary_div=False, enable_ternary_mul=False, verbose=True)
print(f'Baseline: {r0.shape[1]} features')

# Step 1: +二元除
r1 = expand(data, enable_binary_div=True, enable_ternary_mul=False, verbose=True)
print(f'+二元除: {r1.shape[1]} features')

# Step 2: +三元乘 (注意: 特征数可能很大)
r2 = expand(data, enable_binary_div=True, enable_ternary_mul=True, 
            enable_ternary_mixed=False, verbose=True)
print(f'+三元乘: {r2.shape[1]} features')

# Step 3: +三元混合
r3 = expand(data, enable_binary_div=True, enable_ternary_mul=False,
            enable_ternary_mixed=True, verbose=True)
print(f'+三元混合: {r3.shape[1]} features')
"
```

### 完整运行 (所有运算全开)
```bash
# 编辑 main.py，修改 expand 调用添加参数:
#   data_expanded = expand(data, enable_binary_div=True, enable_ternary_mul=True, ...)
python main.py
```

---

## 实验 7.6.4: 迭代式搜索

### 运行
```bash
python main_iterative.py
```

### 切换策略
编辑 `main_iterative.py` 中 CONFIG:
```python
CONFIG = {
    'strategy': 'layered',    # 'layered' | 'crossover' | 'replace'
    'sis_n': 50,              # 每轮保留特征数
    'guide_k': 3,             # 种子特征数
    'max_iter': 20,           # 最大迭代
    ...
}
```

### 输出
- 控制台: 每轮 top-5 特征 + 最终最优公式
- `log_iterative`: 详细日志
- `fit.png`: 最优公式拟合图

---

## 结果记录模板

| 实验 | 配置 | 特征数 | Top-1 R² | 最优公式 | 耗时 |
|---|---|---|---|---|---|
| 7.6.2 解析+SIS | analytical, SIS=10 | - | TBD | TBD | TBD |
| 7.6.2 NN+SIS | nn, SIS=10 | - | TBD | TBD | TBD |
| 7.6.2 解析-SIS | analytical, SIS=OFF | - | TBD | TBD | TBD |
| 7.6.2 NN-SIS | nn, SIS=OFF | - | TBD | TBD | TBD |
| 7.6.3 baseline | 幂次+二元乘 | TBD | TBD | TBD | TBD |
| 7.6.3 +二元除 | +a/b, b/a | TBD | TBD | TBD | TBD |
| 7.6.3 +三元乘 | +a×b×c | TBD | TBD | TBD | TBD |
| 7.6.3 +三元混合 | +a×b/c等 | TBD | TBD | TBD | TBD |
| 7.6.4 迭代 | layered, K=3 | TBD (累计) | TBD | TBD | TBD |