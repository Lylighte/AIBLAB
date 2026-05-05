# 长短期记忆网络（LSTM）详解

## 1. 为什么需要 LSTM？

传统 RNN 在长序列训练中存在严重的**梯度消失/爆炸**问题。当序列长度增加时，梯度在反向传播过程中逐时间步连乘，若权重矩阵的奇异值 < 1，梯度指数级衰减（消失）；若 > 1，梯度指数级增长（爆炸）。这使得 RNN 无法捕捉长距离依赖关系。

LSTM（Long Short-Term Memory）通过**门控机制**和**细胞状态（Cell State）** 的引入，让信息可以在时间步之间有选择地传递和保存，有效解决了这一问题。

## 2. LSTM 的数学原理

LSTM 在每一个时间步 $t$ 的核心运算如下：

### 遗忘门（Forget Gate）
决定从上一时刻的细胞状态中丢弃哪些信息：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

### 输入门（Input Gate）
决定将哪些新信息写入细胞状态：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

### 候选记忆（Candidate Memory）
生成当前时间步的候选更新值：
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

### 细胞状态更新（Cell State Update）
融合遗忘门和输入门的结果，更新细胞状态：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

### 输出门（Output Gate）
决定从细胞状态中输出哪些信息到隐藏状态：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

### 隐藏状态输出（Hidden State Output）
$$h_t = o_t \odot \tanh(C_t)$$

其中 $\sigma$ 是 sigmoid 函数（输出范围 0~1，控制信息通过的比例），$\odot$ 表示逐元素乘法。

---

## 3. LSTM 的信息流图解

```
                    ┌──────────┐
    C_{t-1} ───────▶│  合并/更新 │───▶ C_t
                    └──────────┘
                        ▲
            ┌───────────┼──────────────────────┐
            │    ┌──────┴──────┐   ┌─────────┐ │
            │    │ 遗忘门 f_t  │   │ 输入门 i_t│ │
            │    │  σ(W·[h,x]) │   │ σ(W·[h,x])│ │
            │    └──────┬──────┘   └────┬┬────┘ │
            │           │               ││      │
            │    ┌──────┴───────────────┘│      │
            │    │ 候选记忆 ˜C_t         │      │
            │    │  tanh(W·[h,x])        │      │
            │    └───────────────────────┘      │
            │                                   │
            │    ┌──────────────────────────┐   │
            │    │    输出门 o_t             │   │
            │    │    σ(W·[h,x])             │   │
            │    └────────────┬─────────────┘   │
            │                 │                  │
    h_{t-1} ───────┘         ▼                  │
    x_t ───────┘        h_t = o_t ⊙ tanh(C_t)    │
                    └──────────────────────────────┘
```

---

## 4. LSTM 与 GRU 的对比

| 特性 | LSTM | GRU |
|------|------|-----|
| **门的数量** | 3（遗忘门 / 输入门 / 输出门） | 2（重置门 / 更新门） |
| **细胞状态** | 有独立的 $C_t$，长期记忆 | 无独立细胞状态 |
| **参数量** | 更多（$4 \times hidden^2$ per layer） | 更少（$3 \times hidden^2$ per layer） |
| **表达能力** | 更强，适合大数据和复杂依赖 | 足够，适合小数据或简单任务 |
| **训练速度** | 较慢 | 较快 |
| **过拟合风险** | 较高 | 较低 |
| **内存占用** | 较大（需维护 $h$ 和 $C$） | 较小（只需维护 $h$） |

### 选择建议

| 场景 | 推荐 |
|------|------|
| 小数据集（如 SpeechCommands 子集） | **GRU** — 更快，不易过拟合 |
| 大数据集（如 THCHS-30、LibriSpeech） | **LSTM** — 更强的长程依赖建模 |
| 需要精细时序控制 | **LSTM** — 独立细胞状态更灵活 |
| 计算资源有限 | **GRU** — 参数少，训练快 |

---

## 5. PyTorch 中的 LSTM 实现

### 5.1 使用 nn.LSTM（推荐）

```python
import torch.nn as nn

lstm = nn.LSTM(
    input_size=64,       # 输入特征维度（如 Mel 滤波器数量）
    hidden_size=256,      # 隐藏层大小
    num_layers=2,         # LSTM 堆叠层数
    batch_first=True,     # 输入形状: (batch, seq_len, input_size)
    dropout=0.3,          # 层间 Dropout（num_layers>1 时生效）
    bidirectional=False   # 是否双向 LSTM
)

# 前向传播
# x: (batch, seq_len, input_size)
# h0, c0: 初始隐藏状态和细胞状态（全零即可）
out, (hn, cn) = lstm(x, (h0, c0))
# out: (batch, seq_len, hidden_size * num_directions)
```

### 5.2 手动实现 LSTM 细胞（教学用）

```python
class LSTMCellManual(nn.Module):
    """手动实现 LSTM 单个时间步的计算"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 输入到 4 个门的联合权重
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        # 隐藏状态到 4 个门的联合权重
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        gates = self.W_ih(x) + self.W_hh(h_prev)
        f_t, i_t, g_t, o_t = gates.chunk(4, dim=-1)

        f_t = torch.sigmoid(f_t)   # 遗忘门
        i_t = torch.sigmoid(i_t)   # 输入门
        g_t = torch.tanh(g_t)      # 候选记忆
        o_t = torch.sigmoid(o_t)   # 输出门

        c_next = f_t * c_prev + i_t * g_t   # 细胞状态更新
        h_next = o_t * torch.tanh(c_next)   # 隐藏状态更新

        return h_next, c_next
```

---

## 6. 参数调优指南

### 6.1 核心超参数速查表

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `hidden_size` | 128 / 256 / 512 | 隐藏层大小，越大表达能力越强 |
| `num_layers` | 1 / 2 / 3 | 层数，2 层通常足够 |
| `dropout` | 0.2 ~ 0.5 | 正则化，大模型配大 dropout |
| `learning_rate` | 5e-4 ~ 3e-3 | Adam 推荐 0.001 起步 |
| `batch_size` | 64 ~ 512 | 受显存限制 |
| `bidirectional` | True / False | 双向 LSTM 参数量翻倍 |

### 6.2 调参五步策略

**第一步：建立基线**
- 设置 `hidden_size=128, num_layers=1, dropout=0.2`
- 跑通训练流程，记录验证集准确率

**第二步：逐一调整 hidden_size**
- 尝试 128 → 256 → 512
- 观察验证集准确率的变化趋势
- 若 512 比 256 提升 < 1%，选 256（更小更快）

**第三步：调整 num_layers**
- 1 层 → 2 层 → 3 层
- 若 2 层比 1 层提升明显但 3 层过拟合，选 2 层

**第四步：调整正则化强度**
- 若过拟合（train loss ↓ 但 val loss ↑），增大 dropout
- 若欠拟合（train 和 val 都不好），减小 dropout

**第五步：尝试双向 LSTM**
- 单向 → 双向（参数量 ×2）
- 命令词识别任务中双向通常有提升

### 6.3 必要训练技巧

| 技巧 | 代码 | 作用 |
|------|------|------|
| **梯度裁剪** | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` | 防梯度爆炸 |
| **学习率调度** | `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)` | loss 停滞时降 lr |
| **早停 (Early Stopping)** | 验证集连续 N 轮不提升则停止 | 防过拟合 |
| **权重初始化** | 正交初始化或 Xavier | 加速收敛 |

### 6.4 快速对比实验模板

```python
configs = [
    {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'bidirectional': False},
    {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3, 'bidirectional': False},
    {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.4, 'bidirectional': True},
    {'hidden_size': 512, 'num_layers': 2, 'dropout': 0.5, 'bidirectional': False},
]

for cfg in configs:
    model = LSTMClassifier(input_size=64, num_classes=35, **cfg)
    # 训练并记录验证集准确率
```

### 6.5 常见问题与解决

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss 为 NaN | 梯度爆炸 | 添加梯度裁剪 / 降低 lr |
| 验证集准确率不提升 | 学习率太小 / 模型容量不足 | 增大 lr 或 hidden_size |
| 训练 loss 很低但验证 loss 高 | 过拟合 | 增大 dropout / 减小模型 / 加数据 |
| 训练速度太慢 | 模型太大 / batch 太小 | 减小 hidden_size / 增大 batch_size |
| 双向 LSTM 反而更差 | 数据量不足 | 换回单向 / 增大 dropout |

---

## 7. 项目代码位置

- **LSTM 命令词识别完整训练脚本**：`RNN/train_lstm.py`
  - 包含手动 LSTM 细胞实现（教学用）和 nn.LSTM 封装（实际训练用）
  - 支持梯度裁剪、学习率调度、最佳模型保存
- **GRU 命令词识别脚本**：`RNN/train_sc2.py`（原始实验代码）
- **端到端 ASR 脚本**：`RNN/train_asr_new2.py`（使用 GRU + CTC Loss）

---

## 8. 参考资料

1. [Understanding LSTM Networks (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
2. [D2L - 长短期记忆网络 (LSTM)](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)
3. [D2L - 门控循环单元 (GRU)](https://zh.d2l.ai/chapter_recurrent-modern/gru.html)
4. [PyTorch nn.LSTM 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
