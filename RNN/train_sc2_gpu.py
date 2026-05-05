"""
train_sc2_gpu.py — GPU 利用率优化版
=====================================
优化策略：
1. 将 MelSpectrogram 移到 GPU 计算 (卸载 CPU 密集的 FFT)
2. 启用 cuDNN benchmark (自动选择最快算法)
3. 增大 batch_size (提高 GPU 计算密度)
4. 预计算 label 索引 (避免每次线性搜索)
5. 使用 pin_memory + prefecth 加速数据传输
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

# ── GPU 信息 ──
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
print(result.stdout)
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── cuDNN 优化 ──
torch.backends.cudnn.benchmark = True   # 自动选择最快算法
torch.backends.cudnn.deterministic = False  # 训练阶段无需确定性的算法

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 42
set_seed(seed)


# ====== 数据集类 ======

class SpeechCommandsDataset(Dataset):
    """自定义数据集：使用 soundfile 加载音频"""
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

        # 预计算标签索引 (优化：避免每次 .index() 线性搜索)
        self.labels = []
        for path in self.files:
            label = os.path.basename(os.path.dirname(path))
            self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1)
        waveform = waveform.unsqueeze(0)  # (1, T)
        return waveform, self.labels[idx]


# ====== 数据集初始化 ======

SC_DIR = os.path.join(os.path.dirname(__file__), "SpeechCommands-sub", "SpeechCommands", "speech_commands_v0.02")

train_set = SpeechCommandsDataset(SC_DIR, "training")
val_set = SpeechCommandsDataset(SC_DIR, "validation")
test_set = SpeechCommandsDataset(SC_DIR, "testing")

# 获取标签映射
labels_all = sorted(list(set(train_set.labels)))
label_to_idx = {label: i for i, label in enumerate(labels_all)}  # O(1) 映射


# ====== 梅尔频谱变换 (将在 GPU 上运行) ======

def create_mel_transform(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64):
    """创建 MelSpectrogram 变换"""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    return transform


# ====== 整理函数 (只处理标签, 不处理梅尔频谱) ======

def collate_fn(batch):
    """只将原始波形打包，MelSpectrogram 在 GPU 上计算"""
    waveforms, labels = zip(*batch)
    # 标签转为索引 (在 CPU 上用字典 O(1))
    targets = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
    return list(waveforms), targets


# ====== 训练参数 ======

batch_size = 512  # 增大 batch，提高 GPU 利用率

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn, pin_memory=True
)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False,
    collate_fn=collate_fn, pin_memory=True
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False,
    collate_fn=collate_fn, pin_memory=True
)


# ====== 模型 ======

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


input_size = 64
hidden_size = 256
num_layers = 4
num_classes = len(labels_all)

model = GRUModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)


# ====== GPU 批处理函数: 对一批原始波形计算梅尔频谱 ======

@torch.no_grad()
def batch_melspectrogram(waveforms, mel_transform, device):
    """批量在 GPU 上计算梅尔频谱并填充"""
    # 将所有波形转移到 GPU
    mel_specs = []
    lengths = []
    for wav in waveforms:
        wav = wav.to(device, non_blocking=True)
        mel = mel_transform(wav)                  # (1, n_mels, T)
        mel = mel.squeeze(0).transpose(0, 1)      # (T, n_mels)
        mel_specs.append(mel)
        lengths.append(mel.size(0))

    # 在 GPU 上填充序列
    padded = torch.nn.utils.rnn.pad_sequence(
        mel_specs, batch_first=True, padding_value=0.
    )
    return padded


# ====== 训练 ======

num_epochs = 10
mel_transform = create_mel_transform().to(device)  # 变换也在 GPU 上

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    time_start = time.time()

    for waveforms, targets in train_loader:
        # [GPU] 梅尔频谱计算 → 填充
        inputs = batch_melspectrogram(waveforms, mel_transform, device)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    time_end = time.time()
    elapsed = time_end - time_start
    print(f"第 {epoch+1} 轮 耗时 {elapsed:.2f}s  "
          f"Loss: {running_loss / len(train_loader):.4f}")

    # 验证
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for waveforms, targets in val_loader:
            inputs = batch_melspectrogram(waveforms, mel_transform, device)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"  Validation Accuracy: {100 * correct / total:.2f}%")

# 测试
model.eval()
correct = total = 0
with torch.no_grad():
    for waveforms, targets in test_loader:
        inputs = batch_melspectrogram(waveforms, mel_transform, device)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
print(f"\nTest Accuracy: {100 * correct / total:.2f}%")
