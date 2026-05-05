import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio  # 仅用于 MelSpectrogram 变换，不用于音频加载
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

# 自定义数据集：使用 soundfile 加载音频，避免 torchcodec 依赖
class SpeechCommandsDataset(Dataset):
    def __init__(self, data_dir, subset="training"):
        self.data_dir = data_dir
        # 扫描所有 .wav 文件
        all_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".wav"):
                    all_files.append(os.path.join(root, f))
        # 加载划分列表
        def load_list(filename):
            filepath = os.path.join(data_dir, filename)
            with open(filepath) as f:
                return {os.path.normpath(os.path.join(data_dir, line.strip())) for line in f}
        val_list = load_list("validation_list.txt") if os.path.exists(os.path.join(data_dir, "validation_list.txt")) else set()
        test_list = load_list("testing_list.txt") if os.path.exists(os.path.join(data_dir, "testing_list.txt")) else set()
        # 按子集划分
        if subset == "validation":
            self.files = [f for f in all_files if os.path.normpath(f) in val_list]
        elif subset == "testing":
            self.files = [f for f in all_files if os.path.normpath(f) in test_list]
        else:  # training
            excludes = val_list | test_list
            self.files = [f for f in all_files if os.path.normpath(f) not in excludes]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        # 用 soundfile 加载音频
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()  # (T,)
        if waveform.dim() > 1:  # 多声道 -> 单声道
            waveform = waveform.mean(dim=1)
        waveform = waveform.unsqueeze(0)  # (1, T)
        # 标签来自父目录名
        label = os.path.basename(os.path.dirname(path))
        return waveform, sr, label, "", 0

# 数据集根目录
SC_DIR = os.path.join(os.path.dirname(__file__), "SpeechCommands-sub", "SpeechCommands", "speech_commands_v0.02")

# 加载训练集、验证集和测试集
train_set = SpeechCommandsDataset(SC_DIR, "training")
val_set = SpeechCommandsDataset(SC_DIR, "validation")
test_set = SpeechCommandsDataset(SC_DIR, "testing")

# 获取所有的命令词标签
labels_all = sorted(list(set(datapoint[2] for datapoint in train_set)))
# print(labels)
# 将命令词标签转换为对应的索引
def label_to_index(word):
    if isinstance(word, torch.Tensor):
        word = word.item()  # 如果 word 是张量，转换为 Python 标量
    return torch.tensor(labels_all.index(word))

# 将索引转换为对应的命令词标签
def index_to_label(index,labels):
    if isinstance(index, torch.Tensor):
        index = index.item()  # 如果 index 是张量，转换为 Python 标量
    return labels[index]

# 定义梅尔频谱转换
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

# 对音频序列进行填充，使批量数据长度一致
def pad_sequence(batch):
    # 确保所有序列的特征维度相同
    feature_dim = batch[0].size(-1)
    for tensor in batch:
        assert tensor.size(-1) == feature_dim, f"Feature dimension mismatch: expected {feature_dim}, got {tensor.size(-1)}"
    # 填充序列
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch

# 整理批量数据，包括音频填充和标签转换
def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        # 转换为梅尔频谱
        mel_spec = mel_transform(waveform)
        # 确保梅尔频谱的维度正确
        # print(label)
        tensors.append(mel_spec.squeeze(0).transpose(0, 1))
        targets.append(label_to_index(label))
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

# 设置批量大小
batch_size = 256
# 创建训练集、验证集和测试集的数据加载器
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 定义纯RNN模型
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(SimpleRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 使用 GRU 替代简单 RNN
        # 参数说明
        # input_size: 输入特征的维度（梅尔频谱的特征维度）
        # hidden_size: RNN 隐藏状态的维度
        # num_layers: RNN 的层数
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        # 添加 Batch Normalization 层
        self.bn = nn.BatchNorm1d(hidden_size)
        # 添加 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        # 定义全连接层，用于输出分类结果
        # num_classes: 分类的类别数（命令词的数量）
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 通过 GRU 层进行前向传播
        out, _ = self.gru(x, h0)
        # out, _ = self.rnn(x, h0)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 使用 Batch Normalization
        out = self.bn(out)

        # 使用 Dropout
        out = self.dropout(out)

        # 通过全连接层得到最终输出
        out = self.fc(out)
        return out
    

# 初始化模型的参数
input_size = 64  # 根据梅尔频谱的特征维度修改
hidden_size = 256
num_layers = 4
num_classes = len(labels_all)
# 创建RNN模型实例
model = SimpleRNNModel(input_size, hidden_size, num_layers, num_classes)

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 检查是否有可用的GPU，若有则使用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device='cpu'
model.to(device)

# 设置训练的轮数
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    time_start=time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()
        running_loss += loss.item()
    time_end=time.time()
    elapsed_time=time_end-time_start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"第 {epoch+1} 轮花费了 {minutes} 分 {seconds:.2f} 秒")
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # 在验证集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

    print(f'Validation Accuracy: {100 * correct / total}%')

# 在测试集上评估模型
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

print(f'Test Accuracy: {100 * correct / total}%')