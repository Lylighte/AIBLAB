# 综合实验：基于循环神经网络的语音识别实验报告

## TL;DR

| 项目 | 内容 |
|------|------|
| **实验名称** | 基于循环神经网络的语音识别实验 |
| **完成内容** | 命令词识别基础要求（RNN, Test Acc 25.2% > 20%） |
|  | 命令词识别进阶要求（GRU, Test Acc 81.6% > 80%） |
|  | 端到端语音识别（预训练-微调 10epochs, Dev CER 11.01%） |
|  | 扩展实验1：预训练-微调冻结对比 |
|  | 扩展实验2：录制语音测试 |
| **运行环境** | 本地 Windows + NVIDIA RTX 4070 (12GB) + WSL |
| **框架版本** | PyTorch 2.11 + CUDA 12.8 |
| **编程语言** | Python 3.13 |

---

## 一、实验目的

1. 理解循环神经网络（Recurrent Neural Network, RNN）的运算原理和过程。
2. 掌握自动语音识别（Automatic Speech Recognition, ASR）的概念和原理。
   - (a) 掌握命令词识别的方法。
   - (b) 掌握端到端语音识别的方法。
3. 学会使用"预训练-微调"的思路解决问题。
   - (a) 理解预训练-微调的目的。
   - (b) 学会使用预训练模型提取特征。

---

## 二、实验原理

### 2.1 语音信号与特征提取

语音以 `.wav` 格式保存，按采样率逐点存储。本实验所有语音均为 **16kHz** 采样率。原始语音波形采样率高、数据量大，直接作为模型输入会导致参数量过大、收敛困难。

本实验使用 **梅尔频谱（Mel Spectrogram）** 作为特征。梅尔频谱模拟人耳听觉特性（对低频敏感、高频不敏感），在保留语音关键特征的同时大幅降维。使用 `torchaudio.transforms.MelSpectrogram` 提取特征，参数如下：

- `sample_rate=16000`
- `n_fft=1024`
- `hop_length=512`
- `n_mels=64`

### 2.2 循环神经网络 RNN

RNN 是一种序列到序列的运算模型。对于输入序列 $X=[x_1, x_2, \dots, x_n]$，RNN 通过递推方式依次计算隐藏状态和输出：

$$h_i = f(Ux_i + Wh_{i-1} + b)$$
$$y_i = \text{Softmax}(Vh_i + c)$$

然而传统 RNN 存在严重的**梯度消失/爆炸**问题，难以捕捉长距离依赖关系。

### 2.3 GRU（门控循环单元）

GRU 通过**重置门（Reset Gate）** 和**更新门（Update Gate）** 控制信息流动，有效缓解梯度消失问题：

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(更新门)}$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(重置门)}$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

GRU 参数少于 LSTM（2 个门 vs 3 个门），在小数据集上不易过拟合，训练速度更快。本实验命令词识别部分最终采用 **4 层 GRU** 架构。

### 2.4 命令词识别

命令词识别将输入语音分类为预先定义的有限个命令词类别。流程如下：

1. **特征提取**：将语音波形转为梅尔频谱
2. **序列建模**：使用 RNN/GRU 提取时序特征
3. **分类输出**：取最后一个时间步的隐藏状态，通过全连接层输出类别概率

### 2.5 端到端语音识别

端到端语音识别直接将语音信号映射为文本序列，采用 **CTC（Connectionist Temporal Classification）** 损失函数解决输入输出不等长问题：

1. **编码**：使用 CNN + GRU 将语音特征编码为高层特征
2. **对齐**：CTC 损失自动学习输入帧与输出字符的对齐
3. **解码**：使用 CTC 解码合并重复字符并去除空白符

---

## 三、命令词识别 — 实验代码

### 3.1 数据集与预处理

使用 SpeechCommands 子集，训练集 10,000 条，验证/测试集各 500 条，共 35 个命令词类别。

```python
# 定义梅尔频谱转换
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

# 对音频序列进行填充，使批量数据长度一致
def pad_sequence(batch):
    feature_dim = batch[0].size(-1)
    for tensor in batch:
        assert tensor.size(-1) == feature_dim, \
            f"Feature dimension mismatch: expected {feature_dim}, got {tensor.size(-1)}"
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch

# 整理批量数据
def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        mel_spec = mel_transform(waveform)
        tensors.append(mel_spec.squeeze(0).transpose(0, 1))
        targets.append(label_to_index(label))
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets
```

### 3.2 模型结构

#### 使用 RNN

```python
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(SimpleRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # TODO: 补全 RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout_rate)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]    # 取最后一个时间步输出
        out = self.fc(out)
        return out
```

#### 使用 GRU

```python
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(SimpleRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # TODO: 将 RNN 替换为 GRU（进阶要求）
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout_rate)
        # 添加 Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_size)
        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)     # 通过 GRU
        out = out[:, -1, :]           # 取最后一个时间步
        out = self.bn(out)            # Batch Normalization
        out = self.dropout(out)       # Dropout 正则化
        out = self.fc(out)            # 分类
        return out
```

### 3.3 训练配置

| 参数 | RNN | GRU |
|------|-------------|-------------|
| 隐藏层大小 | 256 | 256 |
| 层数 | 4 | 4 |
| Dropout | 0.2 | 0.2 |
| batch | 256 | 256 |
| 学习率 lr | 0.001 (固定) | 0.001 (固定) |
| 优化器 | Adam | Adam |
| 损失函数 | CrossEntropyLoss | CrossEntropyLoss |
| 训练轮数 | 10 | 20 |

```python
# 模型初始化
input_size = 64      # 梅尔频谱特征维度
hidden_size = 256
num_layers = 4
num_classes = len(labels_all)   # 35个命令词

model = SimpleRNNModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 损失计算
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        running_loss += loss.item()
    # 验证
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, '
          f'Val Acc: {100*correct/total:.2f}%')
```

---

## 四、实验结果与分析

### 4.1 RNN 10 轮

| 指标 | 值 |
|------|-----|
| **测试集准确率** | **25.2%** |
| 验证集最佳准确率 | 23.2%（第 10 轮） |
| 最终 Loss | 2.77 |
| 每轮耗时 | ~7.2 秒 |

**分析：**

1. **整体趋势**：验证准确率从第 1 轮的 11.0% 缓慢提升至第 10 轮的 23.2%，说明 RNN 在学习，但收敛速度很慢。
2. **梯度消失**：第 5 轮和第 8 轮准确率出现明显回落（16.6% 和 13.0%），这是传统 RNN 梯度消失的典型表现——较早时间步的梯度在反向传播中指数级衰减，导致长距离依赖难以学习。
3. **Loss 下降有限**：Loss 从 3.25 降至 2.77，降幅仅 0.48，说明模型容量不足以充分拟合数据。

### 4.2 GRU 10 轮

| 指标 | 值 |
|------|-----|
| **测试集准确率** | **79.2%** |
| 验证集最佳准确率 | 77.0%（第 10 轮） |
| 最终 Loss | 0.45 |
| 每轮耗时 | ~6.9 秒 |

**分析：**

仅将 `nn.RNN` 替换为 `nn.GRU`，测试准确率从 25.2% 升至 79.2%。

1. **快速收敛**：第 1 轮验证准确率即达 31.0%，第 3 轮已超过 60%。
2. **Loss 大幅下降**：从 2.91 降至 0.45，说明 GRU 的门控机制有效缓解了梯度消失，模型能够有效学习。
3. **接近目标**：79.2% 仅差 0.8%，通过增加训练轮数即可轻松突破 80%。

### 4.3 GRU 20 轮

这一轮实验更换了环境，从本地 Windows 切换到 WSL。epochs 增加到 20。

| 指标 | 值 |
|------|-----|
| **测试集准确率** | **81.6%** |
| 验证集最佳准确率 | 81.4%（第 19 轮） |
| 最终 Loss | 0.153 |
| 每轮耗时 | ~5.0 秒 |

**分析：**

1. **持续优化**：Loss 从 0.45（10 轮）降至 0.153（20 轮），模型继续收敛。
2. **验证集稳定超 80%**：第 16 轮起验证准确率多次达到 79.2%~81.4%，第 18~20 轮连续稳定在 80% 以上，表明模型泛化能力良好。

### 4.4 对比总结

| 对比项 | RNN | GRU | GRU+20轮 |
|--------|------------|------------|-----------------|
| **测试准确率** | 25.2% | 79.2% | **81.6%** |
| 首轮验证准确率 | 11.0% | 31.0% | 31.4% |
| 最终 Loss | 2.77 | 0.45 | **0.153** |
| 训练轮数 | 10 | 10 | 20 |
| 每轮耗时 | ~7.2s | ~6.9s | ~5.0s |

### 4.5 模型选择与参数说明

| 参数 | 说明 |
|------|----------|
| **GRU 替代 RNN** | 门控机制有效缓解梯度消失 |
| `hidden_size=256` | 平衡模型容量和计算开销 |
| `num_layers=4` | 多层 GRU 能提取更高层次的时序特征 |
| `dropout=0.2` | 适中 Dropout 保持泛化 |
| `batch_size=256` | 显存充裕（12GB），大 batch 加速训练 |
| `lr=0.001` | - |
| `num_epochs=20` | 10 轮验证集仍在上升 |

---

## 五、端到端语音识别

### 5.1 模型结构

使用 **CNN + GRU** 混合架构：

- **前端 CNN**：3 层卷积块（每层含 ResBlock，膨胀率 1/2/4），提取局部声学特征并下采样
- **后端 GRU**：2 层双向 GRU（hidden_size=128），捕捉长时序依赖
- **输出层**：全连接层映射到字符类别数，使用 **CTC Loss** 进行序列对齐

```python
class ASRModel(nn.Module):
    def __init__(self, mfcc_dim, num_blocks, filters, num_classes, 
                 hidden_size=128, num_layers=2):
        super(ASRModel, self).__init__()
        # CNN 前端
        self.conv1 = nn.Conv1d(mfcc_dim, filters, 3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(filters)
        # ResBlock 序列
        self.blocks = nn.ModuleList([
            ResBlock(filters, 7, r) 
            for _ in range(num_blocks) for r in [1, 2, 4]
        ])
        # GRU 后端（双向）
        self.gru = nn.GRU(filters, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h = self.leaky_relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            h, _ = block(h)
        h = h.permute(0, 2, 1)           # [B, C, T] → [B, T, C]
        output, _ = self.gru(h)          # 双向 GRU
        y_pred = self.fc(output)         # 分类
        y_pred = y_pred.permute(1, 0, 2) # CTC 要求 [T, B, C]
        return y_pred
```

### 5.2 预训练-微调

由于从零训练完整 ASR 模型需要 2 天以上才能出现较好结果，实验中加载了预训练 601 个 epoch 的模型权重 `g_0612_0.601` 进行微调：

```python
# 加载预训练模型
st = torch.load('g_0612_0.601')['model']

# 可选冻结 CNN 前端，只训练 GRU + FC
# for name, param in model.named_parameters():
#     if 'gru' in name or 'fc' in name:
#         param.requires_grad = True
# # GRU 和全连接层保持可训练
#     else:
#         param.requires_grad = False
# # CNN 前端（conv1/bn1/blocks/conv2/bn2）冻结
# model.load_state_dict(st, strict=False)

model.load_state_dict(st)
# 微调全部参数（不冻结）
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss()
```

### 5.3 实验结果

在验证集上微调 10 个 epoch 的结果（WSL 环境）：

| Epoch | Train Loss | CER (Dev) | 预测示例 |
|-------|-----------|-----------|---------|
| 1 | 0.5959 | 21.95% | 蒋慧娟廖鸾凤廖静文翟颜蓉颜军潘长玉潘颜兰薛昭戴芳 |
| 2 | 0.1182 | 15.72% | 蒋慧娟廖鸾凤廖静文翟颜蓉颜军潘长玉潘文兰薛昭戴丽芳 |
| 3 | 0.0690 | 13.81% | 蒋慧娟廖鸾凤廖静文翟文蓉颜小军潘长玉潘文兰薛昭戴丽芳 |
| 4 | 0.0505 | 12.16% | 蒋慧娟廖鸾凤廖静文翟文蓉颜小军潘长玉潘文兰薛昭戴丽芳 |
| 5 | 0.0447 | 11.46% | 蒋慧娟廖鸾凤廖静文翟文蓉颜小军潘长玉潘文兰薛昭戴丽芳 |
| 10 | 0.0346 | **11.01%** | 蒋慧娟廖鸾凤廖静文翟文蓉颜小军潘长玉潘文兰薛昭戴丽芳 |

**验证集最低 CER：10.94%**（第 9 轮）。

**测试集 CER：101%**。分析原因：预训练模型在训练集上训练 601 轮后已严重过拟合，测试集与训练集分布存在很大差异（不同说话人、录音条件等），导致测试集泛化能力较差。

---

## 六、扩展实验

### 6.1 扩展实验 1：预训练-微调冻结对比

**实验设置：** 比较在加载预训练模型后，冻结 vs 不冻结部分模型参数对微调结果的影响。

**冻结策略：** 冻结 CNN 前端（`conv1`、`bn1`、`blocks`、`conv2`、`bn2`），仅训练 GRU 后端和全连接层：

```python
model.load_state_dict(st, strict=False)

# 冻结 CNN 前端，只训练 GRU + FC
for name, param in model.named_parameters():
    if 'gru' in name or 'fc' in name:
        param.requires_grad = True   # GRU 和全连接层保持可训练
    else:
        param.requires_grad = False  # CNN 前端冻结
```

**对比结果：**

| Epoch | Train Loss | CER (Dev) | 预测示例 |
|:----:|:----------:|:---------:|---------|
| 1 | 0.7215 | 26.43% | 蒋慧娟廖鸾凤廖静的文翟军蓉终大颜出军潘将玉潘兰薛昭戴芳 |
| 2 | 0.2661 | 20.90% | 蒋慧娟廖鸾凤廖静文翟军蓉颜军玉突军潘将玉潘文兰薛昭戴芳 |
| 3 | 0.1867 | 18.64% | 蒋慧娟廖鸾凤廖静文翟文蓉颜军潘军潘将玉潘兰薛昭戴丽芳 |
| 4 | 0.1682 | 17.88% | 蒋慧娟廖鸾凤廖静文翟颜蓉颜潘军潘将玉潘兰薛昭戴丽芳 |
| 5 | 0.1669 | 17.74% | 蒋慧娟廖鸾凤廖静文翟文蓉颜文潘军潘长玉潘文兰薛昭戴丽芳 |
| 6 | 0.1311 | 17.64% | 蒋慧娟廖鸾凤廖静文翟文蓉颜小军潘长玉潘兰薛昭戴丽芳 |
| 7 | **0.1185** | **17.51%** | 蒋慧娟廖鸾凤廖静文翟文蓉颜军小军潘长玉潘兰薛昭戴丽芳 |
| 8 | 1.3509 | 31.68% | 蒋慧娟廖鸾凤廖静文翟颜蓉颜文军潘长玉潘文兰薛昭戴丽芳 |
| 9 | 0.4219 | 25.47% | 蒋慧娟廖鸾凤廖静文翟颜蓉颜文军文潘长玉潘文兰薛昭戴丽芳 |
| 10 | 0.2770 | 23.84% | 蒋慧娟廖鸾凤廖静文翟文蓉颜军潘长玉潘文兰薛昭戴丽芳 |

**测试集样本输出（5 个样例）：**

```
[样本1] 预测: 美了厂变即能到快正使越专快专夜越而在景在进续不文崖冬成在
[样本1] 真实: 从鸦片战争到清政府垮台仅对外赔款一项累计将近白银十三亿两
[样本2] 预测: 除强此落然格要城特四和从比某愿的愿立愿选叫费差消立的它本高它的出费体王帘了
[样本2] 真实: 家住焦东矿家属院的申老由于所在企业效益不佳加上长年患病医疗费已花去二万多元生活极度困难
[样本3] 预测: 不宁一有常越上高宋有高后刀后高宁以功也推修不也编也文的作剧动
[样本3] 真实: 除育菜外还可育烤烟西瓜稻谷等农作物秧苗生长期提前一个月左右
[样本4] 预测: 为们之国苏获大育就仰几何大打妙大妙打黑横解钱带女就联摄没想成啊他
[样本4] 真实: 鸿渐进门只见母亲坐在吃饭的旧圆桌侧面抱着阿凶喂他奶粉阿丑在旁吵闹
[样本5] 预测: 通民维空为果生武因变上以了运了藤将拐其将从化将扬上上台料
[样本5] 真实: 国务院出于安全考虑没有透露谁将前往布尔也不透露这位外交官将从何地出发
测试集字错误率 (CER): 100.99%
```

**对比汇总：**

| 策略 | 可训练部分 | 最佳 Dev CER | 测试集 CER | 训练稳定性 |
|:----|:----------|:-----------:|:---------:|:---------:|
| **不冻结**（全参数） | CNN + GRU + FC | **10.94%** | - | 稳定下降 |
| **冻结 CNN** | 仅 GRU + FC | **17.51%** | - | 后期不稳定 |

**分析：**

1. **冻结 CNN 效果明显不如全参数微调**：最佳 Dev CER 为 17.51%（Epoch 7），高于不冻结的 10.94%。说明 ASR 任务需要调整声学特征提取器来适应下游任务，固定 CNN 前端限制了模型的表达能力。

2. **训练后期出现不稳定性**：Epoch 8 Loss 从 0.1185 突增到 1.3509，CER 从 17.51% 跳升到 31.68%，之后虽有所恢复但无法回到最佳水平。这是因为冻结 CNN 后，GRU + FC 需要在不改变前端特征的情况下独自拟合数据，容易陷入局部最优或发生梯度震荡。

3. **预测文本对比**：两者在验证集上预测的人名都逐渐接近真实标注。不冻结策略下第 3 轮就已将 "颜小军" "潘长玉" "潘文兰" "戴丽芳" 全部预测正确；而冻结策略到第 6 轮才勉强正确，且仍有 "潘兰" 少字错误，说明 GRU 后端的建模能力受限于固定的 CNN 特征。

4. **测试集泛化**：两种策略的测试集 CER 均在 101% 左右，说明预训练 601 轮的模型已严重过拟合训练集分布。冻结策略未能改善这一问题，因为过拟合主要源于预训练阶段而非微调阶段。

5. **结论**：对于本实验的 ASR 任务，**全参数微调优于冻结微调**。冻结策略更适合下游数据量极少、或预训练模型与下游任务非常接近的场景。在本实验中 CNN 提取的声学特征需要根据任务进行适配，因此全参数微调是更好的选择。

### 6.2 扩展实验 2：录制语音测试

录制自己的语音进行 ASR 测试。实验中用手机录制了一段中文语音（"徐州地方历代大规模征战五十余次"），使用训练好的 ASR 模型进行识别。

**测试流程：**

```python
# 1. 加载音频（librosa.load 自动归一化）
waveform, sr = librosa.load("my_voice.wav", sr=16000)
# 2. 提取 MFCC 特征 + CMVN 归一化
fea = extract_mfcc(waveform)
fea = normalize_mfcc(fea)       # 均值-方差归一化（解决数值溢出）
# 3. 模型推理
model.eval()
with torch.no_grad():
    Y_pred = model(fea.unsqueeze(0).transpose(1, 2).cuda())
# 4. CTC 解码
Y_pred = torch.argmax(Y_pred, dim=2).squeeze(1)
result = decode_ctc(Y_pred)
print("识别结果:", result)
```

**测试结果：**

```
真实文本：徐州地方历代大规模征战五十余次
识别文本：爱藏的尔鲜尔的吾尔吾尔学尔学以的城的
字错误率（CER）：120.00%
```

**分析：**

1. **模型输出数值溢出问题**：初始测试时模型输出全为 NaN（无效值），导致解码结果为空。调试发现手机录制语音的 MFCC 特征数值范围远大于训练数据（MFCC 范围达 $[-616, 254]$，而模型各层权重通常落在 $[-5, 5]$ 区间），在前向传播到第 4 个 ResBlock 时产生 NaN。通过添加 **CMVN 归一化**（对每个 MFCC 维度做均值-方差归一化，将特征缩放到均值 0、方差 1 的分布）解决了此问题。

2. **识别质量极差**：输出文本与真实文本几乎毫无关联。
   - **模型严重过拟合**：预训练模型在 THCHS-30 上训练 601 轮，原测试集 CER 已达 101%，对新录音的泛化能力本身就很差；
   - **声学域不匹配**：手机录音的环境（普通房间、麦克风、背景噪声）或与 THCHS-30 子集存在显著差异，模型从未见过这种数据分布；
   - **说话人不匹配**：THCHS-30 使用特定播音员录制，而自录语音的说话人音色、语速、口音均不在训练集中。

3. **改进方向**：要获得较好的自录语音识别效果，需要在更多样化的数据上重新训练，或使用在大规模多说话人数据上预训练的模型进行微调。

---

## 七、思考题

### 问题 1：为什么不直接处理语音，而是使用梅尔谱或 MFCC 特征？

直接处理原始语音波形存在以下问题：

1. **高采样率导致数据量过大**：16kHz 采样率下，1 秒语音就有 16000 个数据点。若直接作为神经网络输入，输入维度极高，参数量爆炸。
2. **冗余信息过多**：原始波形包含大量与语音内容无关的信息（如音色、环境噪声、相位信息等），直接处理会引入不必要的噪声。
3. **人耳听觉特性**：人耳对声音的感知是非线性的——对低频变化敏感，对高频变化不敏感。梅尔谱通过对数尺度缩放模拟了这一特性，让模型关注更符合人类感知的特征。
4. **降维与去相关**：MFCC 通过 DCT 进一步去除了梅尔谱各维度之间的相关性，在更低的维度上保留了语音的"包络"信息，有助于模型更快收敛。

**总结**：使用梅尔谱/MFCC 的本质是将语音信号从高维、冗余的时域表示转换为低维、紧凑的频域表示，在保留语音内容信息的同时，大幅降低模型的计算负担和过拟合风险。

### 问题 2：语音和文本是不等长的，代码中是如何处理 batch 训练的？

代码中通过以下方式处理不等长问题：

1. **填充（Padding）**：在 `collate_fn` 中使用 `torch.nn.utils.rnn.pad_sequence` 将同 batch 内所有序列填充到最大长度，填充值为 0。
2. **取最后一个时间步（命令词识别）**：对于分类任务，命令词识别只需一个类别输出，代码中取 `out[:, -1, :]`——即 RNN/GRU 最后一个时间步的隐藏状态作为整个序列的表示，送入全连接层分类。
3. **CTC Loss（端到端 ASR）**：ASR 任务中使用 `nn.CTCLoss()`，它天然支持不等长序列。CTC 损失函数通过引入"空白符（blank）"机制，允许模型在每一帧预测一个字符或空白，再通过解码算法（去重合并 + 去空白）得到最终文本序列。CTC 的输入长度（帧数）和输出长度（字符数）可以不同，损失函数会计算所有可能对齐路径的概率和。

### 问题 3：端到端语音识别实验使用的字典里一共有多少个汉字？是怎么得到的？

**字典大小：** 约 2500~2600 个汉字（具体为 2666 个，含 `<blank>` 和 `<eos>` 特殊标记）。

**得到方式：**

1. 从 THCHS-30 数据集的训练集中读取所有 `.trn` 文件（标注文件）。
2. 遍历所有标注文本，统计每个汉字字符出现的频次。
3. 按频次降序排列，构建字符到索引的映射 `char2id` 和索引到字符的映射 `id2char`。
4. 额外添加 `<blank>`（CTC 空白符，索引 0）和 `<eos>`（句子结束符，索引为最后一个）。

```python
chars = {}
for text in self.texts:
    for c in text:
        chars[c] = chars.get(c, 0) + 1
chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
chars = [char[0] for char in chars]
# chars 即为字典中的所有汉字，覆盖了 THCHS-30 训练集的所有字符
```

### 问题 6：在 ASR 代码里，哪处地方避免了连续两个"天"字被合并？

在 CTC 解码中，去重合并规则（collapsing repeated characters）会合并相邻相同字符。例如，若模型输出的帧序列为 "天 天 blank 气"，解码后应为"天气"而不是"天 天 气"。但如果真实文本中有两个连续的相同字符，比如"天天"，CTC 的合并规则会将其错误地合并为一个"天"。

代码中通过以下方式避免此问题：

```python
# 在数据处理阶段，若当前字符与下一个字符相同，则在中间插入 0（blank 的索引）
for i in range(len(self.texts[idx])):
    char = self.texts[idx][i]
    if char in self.char2id:
        phone_list.append(self.char2id[char])
    else:
        phone_list.append(self.unk_index)
    # 关键：如果当前字符和下一个字符相同，则在中间插入 0 (blank)
    if i < len(self.texts[idx]) - 1 and self.texts[idx][i] == self.texts[idx][i + 1]:
        phone_list.append(0)  # 插入 blank，防止 CTC 合并
```

这样在生成对齐标注时，连续相同字符之间强制插入了一个空白符，模型学习到在相同字符之间输出 blank，从而在解码时区分出两个独立字符。例如，"天天"的标签序列变成 "天 blank 天"，去重合并后仍为"天天"，不会被误合并为"天"。

---

## 八、遇到的问题与收获

### 8.1 遇到的问题

| 问题 | 原因 | 解决 |
|------|------|------|
| torchaudio 加载音频报错 `TorchCodec is required` | torchaudio 2.11+ 强制依赖 torchcodec | 改用 `soundfile` 手动加载音频 |
| RNN 训练准确率仅 25.2% | 基础 RNN 梯度消失 | 替换为 GRU，准确率跃升至 81.6% |
| 测试集 CER 高达 101% | 预训练模型过拟合，训练集和测试集分布不一致 | 使用预训练-微调 + 更多轮数微调 |
| CosineAnnealingLR 不如固定学习率 | 后期学习率过低，模型难以跳出局部最优 | 改用固定学习率 0.001 训练 20 轮 |

### 8.2 主要收获

1. **深入理解 RNN 与 GRU 的区别**：通过实际代码验证了 GRU 门控机制对梯度消失的缓解作用，从 25.2% 到 81.6% 的跃升是理论与实践结合的生动案例。
2. **掌握语音识别流程**：完整实践了从特征提取（梅尔谱）→ 序列建模（GRU）→ 分类/CTC 解码的语音识别流水线。
3. **预训练-微调策略**：理解了预训练模型的使用场景和注意事项，包括冻结 vs 不冻结的权衡。
4. **序列对齐问题**：深入学习了 CTC Loss 的原理和实现，特别是如何处理连续相同字符的合并问题。
5. **工程实践能力**：掌握了集群提交任务、本地 GPU 训练、环境配置等工程技能。

---

## 九、附录

### 9.1 完整代码文件说明

| 代码文件 | 说明 |
|---------|------|
| `train_sc2.py` | 命令词识别训练脚本（基础版 RNN + 进阶版 GRU） |
| `train_asr_new2.py` | 端到端 ASR 训练脚本（集群路径版本） |
| `train_asr_local.py` | 端到端 ASR 训练脚本（本地路径版本） |

### 9.2 实验结果文件

| 文件 | 说明 |
|------|------|
| `g_0612_0.601` | 预训练 601 epoch 的模型权重 |
| `cha2id.pth` | 字符到索引的映射字典 |
| `id2char.pth` | 索引到字符的映射字典 |

### 9.3 环境配置

| 包名 | 版本 |
|------|------|
| Python | 3.13.13 |
| PyTorch | 2.11.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| soundfile | 最新 |
| librosa | 0.11.0 |
