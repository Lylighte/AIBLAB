"""
===============================================================================
 LSTM (Long Short-Term Memory) — 长短期记忆网络
===============================================================================

1. LSTM 解决了什么问题？
   - 传统 RNN 在长序列上存在"梯度消失/爆炸"问题，无法捕捉长距离依赖。
   - LSTM 通过"门控机制"和"细胞状态(Cell State)"，让信息可以长时间保存。

2. LSTM 的核心公式（单个时间步 t）：
   遗忘门  : f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   输入门  : i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   候选记忆: ˜C_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   细胞更新: C_t = f_t ⊙ C_{t-1} + i_t ⊙ ˜C_t
   输出门  : o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   隐藏状态: h_t = o_t ⊙ tanh(C_t)

   其中 σ 是 sigmoid 函数，⊙ 是逐元素乘法。

3. 与 GRU 的对比：
   - LSTM 有 3 个门（遗忘/输入/输出），GRU 有 2 个门（重置/更新）
   - LSTM 有独立的细胞状态 C，GRU 没有
   - LSTM 参数量更大，表达能力强，但更易过拟合
   - GRU 参数更少，训练更快，在小数据集上往往更好

4. 本项目使用 LSTM 代替 GRU 进行命令词识别。
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import soundfile as sf
import os
import time
import subprocess
from datetime import datetime

result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
output = result.stdout
print(output)
print(torch.cuda.is_available())


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = 42
set_seed(seed)


# =============================================================================
# 数据集部分（与 train_sc2.py 相同）
# =============================================================================
class SpeechCommandsDataset(Dataset):
    def __init__(self, data_dir, subset="training"):
        self.data_dir = data_dir
        all_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".wav"):
                    all_files.append(os.path.join(root, f))
        def load_list(filename):
            filepath = os.path.join(data_dir, filename)
            with open(filepath) as f:
                return {os.path.normpath(os.path.join(data_dir, line.strip())) for line in f}
        val_list = load_list("validation_list.txt") if os.path.exists(os.path.join(data_dir, "validation_list.txt")) else set()
        test_list = load_list("testing_list.txt") if os.path.exists(os.path.join(data_dir, "testing_list.txt")) else set()
        if subset == "validation":
            self.files = [f for f in all_files if os.path.normpath(f) in val_list]
        elif subset == "testing":
            self.files = [f for f in all_files if os.path.normpath(f) in test_list]
        else:
            excludes = val_list | test_list
            self.files = [f for f in all_files if os.path.normpath(f) not in excludes]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1)
        waveform = waveform.unsqueeze(0)
        label = os.path.basename(os.path.dirname(path))
        return waveform, sr, label, "", 0


SC_DIR = os.path.join(os.path.dirname(__file__), "SpeechCommands-sub", "SpeechCommands", "speech_commands_v0.02")

train_set = SpeechCommandsDataset(SC_DIR, "training")
val_set = SpeechCommandsDataset(SC_DIR, "validation")
test_set = SpeechCommandsDataset(SC_DIR, "testing")

labels_all = sorted(list(set(datapoint[2] for datapoint in train_set)))

def label_to_index(word):
    if isinstance(word, torch.Tensor):
        word = word.item()
    return torch.tensor(labels_all.index(word))

def index_to_label(index, labels):
    if isinstance(index, torch.Tensor):
        index = index.item()
    return labels[index]

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)


def pad_sequence(batch):
    feature_dim = batch[0].size(-1)
    for tensor in batch:
        assert tensor.size(-1) == feature_dim
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch


def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        mel_spec = mel_transform(waveform)
        tensors.append(mel_spec.squeeze(0).transpose(0, 1))
        targets.append(label_to_index(label))
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# =============================================================================
# 【核心】LSTM 模型定义
# =============================================================================
#
# 两种实现方式：
#   1. LSTMCell — 手动实现单个时间步的 LSTM 运算，便于理解原理
#   2. nn.LSTM  — PyTorch 高效封装，实际训练使用
#
# =============================================================================

# -----------------------------------------------------------------------------
# 方式一：手动 LSTM 单元（教学用）
# -----------------------------------------------------------------------------
class LSTMCellManual(nn.Module):
    """
    手动实现 LSTM 细胞（单个时间步）。

    公式回忆:
        f_t = σ(W_ih_f @ x_t + W_hh_f @ h_{t-1} + b_f)    # 遗忘门
        i_t = σ(W_ih_i @ x_t + W_hh_i @ h_{t-1} + b_i)    # 输入门
        g_t = tanh(W_ih_g @ x_t + W_hh_g @ h_{t-1} + b_g) # 候选记忆
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t                   # 细胞状态更新
        o_t = σ(W_ih_o @ x_t + W_hh_o @ h_{t-1} + b_o)    # 输出门
        h_t = o_t ⊙ tanh(C_t)                              # 隐藏状态
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入到 4 个门的权重 (input_size -> hidden_size)
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        # 隐藏状态到 4 个门的权重 (hidden_size -> hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        """
        x:      (batch, input_size)
        h_prev: (batch, hidden_size)
        c_prev: (batch, hidden_size)
        返回:
            h_next: (batch, hidden_size)
            c_next: (batch, hidden_size)
        """
        gates = self.W_ih(x) + self.W_hh(h_prev)  # (batch, 4*hidden_size)

        # 拆分为 4 个门
        f_gate, i_gate, g_gate, o_gate = gates.chunk(4, dim=-1)

        # 遗忘门 / 输入门 / 输出门 → sigmoid；候选记忆 → tanh
        f_t = torch.sigmoid(f_gate)   # 遗忘门
        i_t = torch.sigmoid(i_gate)   # 输入门
        g_t = torch.tanh(g_gate)      # 候选记忆
        o_t = torch.sigmoid(o_gate)   # 输出门

        # 细胞状态更新
        c_next = f_t * c_prev + i_t * g_t
        # 隐藏状态更新
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next


# -----------------------------------------------------------------------------
# 方式二：使用 PyTorch 的 nn.LSTM（实际训练用）
# -----------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """
    基于 LSTM 的命令词识别模型。

    架构:
        Mel谱(64维) → LSTM → BatchNorm → Dropout → FC(分类)

    超参数说明（也是调参重点）:
        input_size  : 输入特征维度 = Mel 滤波器数量 (默认 64)
        hidden_size : LSTM 隐藏层大小 (默认 256, 可调 128/256/512)
        num_layers  : LSTM 堆叠层数 (默认 2, 可调 1/2/3/4)
        dropout_rate: Dropout 概率 (默认 0.3, 可调 0.1~0.5)
        bidirectional: 是否双向 (默认 False, 可设为 True 增加表达能力)
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout_rate=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # ---- LSTM 层 ----
        # dropout: num_layers>1 时在层间加入 dropout，最后一层无 dropout
        # batch_first: 输入形状为 (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # ---- 后续处理层 ----
        lstm_output_dim = hidden_size * self.num_directions
        self.bn = nn.BatchNorm1d(lstm_output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        返回: (batch, num_classes)
        """
        # ---- LSTM 前向 ----
        # 初始化隐藏状态和细胞状态 (全零)
        # h0: (num_layers * num_directions, batch, hidden_size)
        # c0: 同上
        device = x.device
        h0 = torch.zeros(self.num_layers * self.num_directions,
                         x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * self.num_directions,
                         x.size(0), self.hidden_size, device=device)

        # out: (batch, seq_len, hidden_size * num_directions)
        # hn, cn 是最后时刻的 (h, c)，此处不用
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # ---- 取最后一个时间步的输出 ----
        if self.bidirectional:
            # 双向 LSTM: 拼接正向最后 + 反向最后
            # out[:, -1, :hidden_size] 正向最后
            # out[:, 0, hidden_size:]  反向最后
            last_out = torch.cat((out[:, -1, :self.hidden_size],
                                  out[:, 0, self.hidden_size:]), dim=-1)
        else:
            last_out = out[:, -1, :]  # (batch, hidden_size)

        # ---- 后续处理 ----
        out = self.bn(last_out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# =============================================================================
# 参数初始化
# =============================================================================
input_size = 64       # Mel 频谱特征维度
hidden_size = 256     # LSTM 隐藏层大小 (调参重点)
num_layers = 2        # LSTM 层数
num_classes = len(labels_all)
dropout_rate = 0.3
bidirectional = False # 是否使用双向 LSTM

model = LSTMClassifier(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout_rate=dropout_rate,
    bidirectional=bidirectional
)

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[LSTM] 总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}")


# =============================================================================
# 损失函数与优化器
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 可选: 学习率调度器 (当验证 loss 不再下降时降低 lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20

# =============================================================================
# 训练循环
# =============================================================================
best_val_acc = 0.0

for epoch in range(num_epochs):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    time_start = time.time()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪: 防止梯度爆炸 (LSTM 常见技巧)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        running_loss += loss.item()

    time_end = time.time()
    elapsed = time_end - time_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    avg_loss = running_loss / len(train_loader)

    print(f"\n第 {epoch+1} 轮花费了 {minutes} 分 {seconds:.2f} 秒")
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # --- 验证阶段 ---
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_loss_avg = val_loss / len(val_loader)
    print(f'Validation Loss: {val_loss_avg:.4f}, Accuracy: {val_acc:.2f}%')

    # 学习率调度
    scheduler.step(val_loss_avg)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(os.path.dirname(__file__), 'best_lstm_sc.pth'))
        print(f'✓ 保存最佳模型 (val_acc={val_acc:.2f}%)')

# --- 测试阶段 ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n{"="*50}')
print(f'测试集准确率: {100 * correct / total:.2f}%')
print(f'最佳验证集准确率: {best_val_acc:.2f}%')
print(f'{"="*50}')


# =============================================================================
# LSTM 参数调优指南
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LSTM 超参数调优指南 (SpeechCommands)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. hidden_size (隐藏层大小)                                                │
│     - 默认 256                                                              │
│     - 调优范围: [64, 128, 256, 512]                                         │
│     - 太小 (<64):  欠拟合，无法捕捉复杂模式                                   │
│     - 太大 (>512):  过拟合，参数量暴增，训练变慢                               │
│     - 建议: 从 128 开始，用验证集准确率判断增减                                 │
│                                                                             │
│  2. num_layers (LSTM 层数)                                                  │
│     - 默认 2                                                                │
│     - 调优范围: [1, 2, 3, 4]                                                │
│     - 1 层: 训练最快，适合简单任务                                            │
│     - 2-3 层: 更深的特征提取，适合中等复杂度任务                                │
│     - 4 层+: 容易过拟合，训练慢，需要更多数据和正则化                            │
│     - 建议: SpeechCommands 用 2 层足够                                       │
│                                                                             │
│  3. dropout_rate                                                            │
│     - 默认 0.3                                                              │
│     - 调优范围: [0.1, 0.2, 0.3, 0.4, 0.5]                                  │
│     - 过低: 过拟合                                                           │
│     - 过高: 欠拟合                                                           │
│     - 建议: 大模型(大 hidden_size/num_layers)配大 dropout                     │
│                                                                             │
│  4. bidirectional (双向 LSTM)                                                │
│     - False (单向): 只看过去 → 未来方向                                      │
│     - True  (双向): 同时看过去和未来，参数量翻倍                                │
│     - 命令词识别场景: 整个 utterance 已完整输入，双向能看到更多上下文，            │
│       通常能提升准确率，但参数量 ×2，训练时间 ×1.5~2                            │
│     - 建议: 先单向，若验证集不提升再试双向                                      │
│                                                                             │
│  5. learning_rate (学习率)                                                   │
│     - 默认 0.001 (Adam 常用)                                                 │
│     - 调优范围: [5e-4, 1e-3, 3e-3]                                          │
│     - 太大: loss 震荡不收敛                                                   │
│     - 太小: 收敛极慢                                                         │
│     - 建议: Adam 用 0.001 起步，配合 scheduler 动态调整                        │
│                                                                             │
│  6. batch_size                                                              │
│     - 默认 256                                                              │
│     - 调优范围: [64, 128, 256, 512]                                         │
│     - 受限于 GPU 显存                                                        │
│     - 太大的 batch 会降低泛化能力                                              │
│     - 建议: 在显存允许下尽量大，但不超过 512                                    │
│                                                                             │
│  7. 梯度裁剪 (Gradient Clipping)                                             │
│     - LSTM 容易出现梯度爆炸，clip_grad_norm_ 是必备技巧                        │
│     - max_norm 建议: [1.0, 5.0, 10.0]                                       │
│     - 太小: 梯度截断太狠，训练缓慢                                             │
│     - 太大: 起不到防梯度爆炸的作用                                              │
│     - 建议: 先设 5.0，观察 loss 曲线调整                                       │
│                                                                             │
│  8. 优化器选择                                                               │
│     - Adam (默认): 自适应学习率，收敛快，适合大多数情况                            │
│     - SGD + Momentum: 需要精细调 lr，但泛化性可能更好                           │
│     - AdamW: Adam + 解耦权重衰减，比 Adam 正则化效果更好                         │
│     - 建议: 先用 Adam，想进一步提升换 AdamW 或 SGD                             │
│                                                                             │
│  9. 学习率调度策略                                                            │
│     - ReduceLROnPlateau: val_loss 停滞时降 lr (已启用)                        │
│     - CosineAnnealing: 余弦退火，适合长时间训练                                │
│     - StepLR: 每隔固定步数降 lr                                               │
│     - 建议: ReduceLROnPlateau 是最省心的选择                                   │
│                                                                             │
│  10. LSTM vs GRU 在这个任务上的对比                                           │
│      - LSTM 参数量更多 (~4×hidden_size² per layer)                           │
│      - GRU 参数量更少 (~3×hidden_size² per layer)                            │
│      - 小数据集 (如本任务的子集) GRU 可能更好，不容易过拟合                       │
│      - 大数据集 LSTM 更强的表达能力更有优势                                     │
│      - 建议: 都试试，谁验证集高用谁                                            │
│                                                                             │
│  11. 快速调参实验模板                                                         │
│      params_grid = [                                                         │
│          {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3},              │
│          {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},              │
│          {'hidden_size': 256, 'num_layers': 3, 'dropout': 0.4},              │
│          {'hidden_size': 512, 'num_layers': 2, 'dropout': 0.5},              │
│      ]                                                                       │
│      每次跑一个配置，记录验证集准确率，选最优的。                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# 附加: 使用 LSTM 进行预测的示例函数
# =============================================================================
def predict_lstm(model, audio_path, mel_transform, labels, device):
    """
    对单条音频进行命令词预测。

    参数:
        model: 训练好的 LSTM 模型
        audio_path: 音频文件路径
        mel_transform: MelSpectrogram 变换
        labels: 命令词列表
        device: 设备
    返回:
        predicted_label: 预测的命令词
        probabilities: 各类别概率
    """
    model.eval()
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=1)
    waveform = waveform.unsqueeze(0)

    mel_spec = mel_transform(waveform)                    # (1, n_mels, T)
    mel_spec = mel_spec.squeeze(0).transpose(0, 1)        # (T, n_mels)
    mel_spec = mel_spec.unsqueeze(0).to(device)            # (1, T, n_mels)

    with torch.no_grad():
        output = model(mel_spec)                           # (1, num_classes)
        probabilities = torch.softmax(output, dim=1)       # (1, num_classes)
        pred_idx = torch.argmax(probabilities, dim=1).item()

    predicted_label = labels[pred_idx]
    return predicted_label, probabilities.cpu()

# 对一条真实音频做预测（训练完成后会执行）
example_audio = os.path.join(SC_DIR, "yes", "022cd682_nohash_0.wav")
if os.path.exists(example_audio):
    label, probs = predict_lstm(model, example_audio, mel_transform, labels_all, device)
    top5 = torch.topk(probs, 5)
    print(f'\n{"="*50}')
    print(f'单条预测示例: {example_audio}')
    print(f'真实标签: yes | 预测结果: {label}')
    print(f'Top-5 预测:')
    for i in range(5):
        idx = top5.indices[0][i].item()
        print(f'  {i+1}. {labels_all[idx]}: {top5.values[0][i].item():.2%}')
    print(f'{"="*50}')
