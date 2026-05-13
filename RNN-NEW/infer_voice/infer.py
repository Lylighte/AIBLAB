"""
录制语音测试脚本
================
扩展实验 2：用手机录制一段中文语音（m4a），转换为 WAV 后，
用训练好的 ASR 模型（g_0612_0.601）进行语音识别。

用法：
    1. 把手机录制的 m4a 文件放到本目录下
    2. 运行：python infer.py my_voice.m4a
    3. 或直接指定 WAV 文件：python infer.py my_voice.wav

依赖：
    pip install torch librosa soundfile pydub numpy
    需要安装 ffmpeg（用于 m4a 转 wav）
"""

import os
import sys
import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
import argparse
import subprocess
from pathlib import Path


# =============================================================================
# 模型定义（与训练时一致）
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(filters, filters, kernel_size,
                               stride=1, padding='same', dilation=dilation_rate)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, 1,
                               stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        hf = self.leaky_relu(self.bn1(self.conv1(x)))
        hg = self.leaky_relu(self.bn1(self.conv1(x)))
        h0 = hf * hg
        ha = self.leaky_relu(self.bn2(self.conv2(h0)))
        hs = self.leaky_relu(self.bn2(self.conv2(h0)))
        return ha + x, hs


class ASRModel(nn.Module):
    def __init__(self, mfcc_dim, num_blocks, filters, num_classes,
                 hidden_size=128, num_layers=2):
        super(ASRModel, self).__init__()
        self.conv1 = nn.Conv1d(mfcc_dim, filters, 3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.blocks = []
        for i in range(num_blocks):
            for r in [1, 2, 4]:
                self.blocks.append(ResBlock(filters, 7, r))
        self.blocks = nn.ModuleList(self.blocks)
        self.conv2 = nn.Conv1d(filters, filters, 3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)
        self.dropout2 = nn.Dropout(0.2)
        self.gru = nn.GRU(filters, hidden_size, num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = self.leaky_relu(self.bn1(self.conv1(x)))
        h0 = self.dropout1(h0)
        shortcut = []
        for block in self.blocks:
            h0, s = block(h0)
            shortcut.append(s)
        h1 = self.leaky_relu(sum(shortcut))
        h1 = self.leaky_relu(self.bn2(self.conv2(h1)))
        h1 = self.dropout2(h1)
        h1 = h1.permute(0, 2, 1)       # [B, C, T] -> [B, T, C]
        output, _ = self.gru(h1)
        y_pred = self.fc(output)        # [B, T, num_classes]
        y_pred = y_pred.permute(1, 0, 2)  # [T, B, C]  CTC 格式
        return y_pred


# =============================================================================
# 工具函数
# =============================================================================

def extract_mfcc(waveform, sr=16000, n_mfcc=13):
    """
    从波形提取 MFCC 特征，与 prepare_mfcc.py 保持一致。

    参数：
        waveform: numpy数组，音频波形
        sr: 采样率 (Hz)
        n_mfcc: MFCC 维度

    返回：
        torch.FloatTensor, shape (T, n_mfcc)
    """
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # (n_mfcc, T) -> (T, n_mfcc)
    return torch.FloatTensor(mfcc)


def normalize_mfcc(fea):
    """
    CMVN 归一化：对每个 MFCC 维度做均值-方差归一化。
    这是语音识别中的标准预处理，能显著提升数值稳定性。

    参数：
        fea: torch.Tensor, shape (T, n_mfcc)

    返回：
        normalized: torch.Tensor, shape (T, n_mfcc)
    """
    mean = fea.mean(dim=0, keepdim=True)  # (1, n_mfcc)
    std = fea.std(dim=0, keepdim=True)    # (1, n_mfcc)
    # 防止除零
    std = torch.clamp(std, min=1e-8)
    return (fea - mean) / std


def levenshtein_distance(ref, hyp):
    """计算 Levenshtein 编辑距离"""
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return int(dp[m][n])


def decode_ctc(pred_ids, id2char, blank_id=0, eos_id=None):
    """
    CTC 解码：去重合并 + 去除空白符，可选去除 <eos> 及其之后内容。

    参数：
        pred_ids: list[int] 或 1D torch.Tensor，模型 argmax 后的预测序列
        id2char: dict[int -> str]，索引到字符的映射
        blank_id: 空白符索引 (默认 0)
        eos_id: 句子结束符索引 (若为 None 则自动从 id2char 中查找 '<eos>')

    返回：
        result: str，解码后的文本
    """
    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.tolist()

    if eos_id is None:
        eos_id = None
        for k, v in id2char.items():
            if v == '<eos>':
                eos_id = k
                break

    # 步骤 1：去重合并 + 去 blank
    merged = []
    prev = None
    for num in pred_ids:
        if num == blank_id:
            prev = None
            continue
        if num != prev:
            merged.append(num)
            prev = num
        # 相同且非 blank → 跳过（CTC 合并规则）

    # 步骤 2：截断到 <eos> 之前
    if eos_id is not None and eos_id in merged:
        merged = merged[:merged.index(eos_id)]

    # 步骤 3：映射为文本
    result = ''.join([id2char.get(t, '') for t in merged])
    return result


def convert_m4a_to_wav(m4a_path, wav_path=None, target_sr=16000):
    """
    用 ffmpeg 将 m4a 转换为 16kHz 单声道 WAV。

    参数：
        m4a_path: str，输入的 m4a 文件路径
        wav_path: str 或 None，输出的 wav 文件路径
        target_sr: int，目标采样率

    返回：
        wav_path: str，输出的 wav 文件路径
    """
    if wav_path is None:
        wav_path = str(Path(m4a_path).with_suffix('.wav'))

    print(f"转换: {m4a_path} -> {wav_path} (16kHz, mono)")
    cmd = [
        'ffmpeg', '-y',
        '-i', m4a_path,
        '-ac', '1',           # 单声道
        '-ar', str(target_sr), # 重采样
        wav_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print("转换完成")
    return wav_path


# =============================================================================
# 主逻辑
# =============================================================================

@torch.no_grad()
def infer(audio_path, model, id2char, device):
    """
    对一段语音进行识别。

    参数：
        audio_path: str，WAV 音频路径
        model: nn.Module，加载好权重的 ASRModel
        id2char: dict[int -> str]
        device: torch.device

    返回：
        result: str，识别文本
    """
    # 1. 加载音频（使用 librosa.load，与 prepare_mfcc.py 一致，自动归一化到 [-1, 1]）
    print(f"加载音频: {audio_path}")
    waveform, sr = librosa.load(audio_path, sr=16000)

    # 2. 提取 MFCC 并做 CMVN 归一化
    print("提取 MFCC 特征...")
    fea = extract_mfcc(waveform, sr=16000)  # (T, 13)
    fea = normalize_mfcc(fea)
    print(f"   MFCC 形状: {fea.shape}")

    # 3. 模型推理
    print("模型推理...")
    model.eval()
    # fea: (T, 13) -> (1, 13, T)
    fea_input = fea.unsqueeze(0).transpose(1, 2).to(device)
    Y_pred = model(fea_input)  # (T, 1, num_classes)

    # 4. CTC 解码
    Y_pred = torch.argmax(Y_pred, dim=2).squeeze(1)  # (T,)
    result = decode_ctc(Y_pred, id2char)

    return result


def calc_cer(reference, hypothesis):
    """计算字错误率 CER"""
    if len(reference) == 0:
        return float('inf')
    return levenshtein_distance(reference, hypothesis) / len(reference)


def main():
    parser = argparse.ArgumentParser(
        description='用训练好的 ASR 模型识别录制的语音'
    )
    parser.add_argument('audio', type=str,
                        help='输入的音频文件路径（m4a 或 wav）')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join('..', 'g_0612_0.601'),
                        help='预训练模型权重路径')
    parser.add_argument('--cha2id', type=str,
                        default=os.path.join('..', 'cha2id.pth'),
                        help='char2id 字典路径')
    parser.add_argument('--id2char', type=str,
                        default=os.path.join('..', 'id2char.pth'),
                        help='id2char 字典路径')
    parser.add_argument('--true_text', type=str, default=None,
                        help='录音的真实文本（中文），用于计算字错误率 CER')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用 CPU（默认自动选择 GPU）')
    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"使用设备: {device}")

    # 处理输入文件
    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"错误: 文件不存在 -> {audio_path}")
        sys.exit(1)

    ext = Path(audio_path).suffix.lower()
    if ext == '.m4a':
        print("检测到 m4a 文件，正在转换为 WAV...")
        try:
            audio_path = convert_m4a_to_wav(audio_path)
        except Exception as e:
            print(f"ffmpeg 转换失败: {e}")
            print("请确保已安装 ffmpeg，或手动将 m4a 转换为 16kHz 单声道 WAV。")
            sys.exit(1)
    elif ext != '.wav':
        print(f"警告: 未知格式 '{ext}'，将尝试作为 WAV 加载...")

    # 加载字典
    print("加载字典...")
    char2id = torch.load(args.cha2id, map_location='cpu')
    id2char = torch.load(args.id2char, map_location='cpu')

    # 调整为训练时的索引偏移（blank=0, 原 id+1）
    # 与 train_asr_local.py / train_asr_new2.py 的 __main__ 逻辑保持一致
    new_char2id = {char: id + 1 for char, id in char2id.items()}
    new_id2char = {id + 1: char for id, char in id2char.items()}
    new_id2char[0] = '<blank>'
    new_char2id['<blank>'] = 0
    index = len(new_id2char)
    new_id2char[index] = '<eos>'
    new_char2id['<eos>'] = index
    char2id, id2char = new_char2id, new_id2char
    num_classes = len(char2id)
    print(f"字典大小: {num_classes}")

    # 加载模型
    print("加载模型...")
    mfcc_dim = 13
    num_blocks = 3
    filters = 128
    model = ASRModel(mfcc_dim, num_blocks, filters, num_classes).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    print("模型加载成功")

    # 推理
    result = infer(audio_path, model, id2char, device)

    # 输出结果
    print("\n" + "=" * 50)
    print("识别结果:", result)
    if args.true_text:
        print("真实文本:", args.true_text)
        cer = calc_cer(args.true_text, result)
        print(f"字错误率 (CER): {cer * 100:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()
