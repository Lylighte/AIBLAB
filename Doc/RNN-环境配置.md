# RNN 实验环境配置说明

## 1. 创建 Conda 环境

```powershell
# 使用 Python 3.13 创建环境（PyTorch 官方最新已支持）
conda create -n rnn python=3.13 -y
```

## 2. 安装 PyTorch（CUDA 版本）

```powershell
# 用 pip 安装 PyTorch 2.11 + CUDA 12.8
conda run -n rnn pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

> **注意**：`torchaudio` 2.11 及以上版本**强制依赖 `torchcodec`** 进行音频解码，不再支持 `soundfile`、`sox` 等传统后端，因此需要额外安装 `soundfile` 用于手动加载音频。

## 3. 安装音频处理库

```powershell
# soundfile：用于加载 .wav 音频文件（绕过 torchcodec）
conda run -n rnn pip install soundfile

# 可选：安装 librosa（包含更多音频处理功能，但依赖较多）
conda run -n rnn pip install librosa
```

## 4. 代码中的关键适配

### 4.1 数据集类（绕过 torchcodec）

由于 `torchaudio` 2.11 的 `load()` 函数强制使用 `torchcodec` 解码，在 Windows 下需要安装 FFmpeg 才能使用，较为麻烦。因此自定义数据集类 `SpeechCommandsDataset`，使用 **`soundfile`** 手动加载音频：

```python
import soundfile as sf

class SpeechCommandsDataset(Dataset):
    def __getitem__(self, idx):
        path = self.files[idx]
        # 用 soundfile 加载音频
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()  # (T,)
        if waveform.dim() > 1:  # 多声道 -> 单声道
            waveform = waveform.mean(dim=1)
        waveform = waveform.unsqueeze(0)  # (1, T)
        label = os.path.basename(os.path.dirname(path))
        return waveform, sr, label, "", 0
```

### 4.2 数据集路径

数据集放在 `RNN/SpeechCommands-sub/` 目录下，代码通过脚本所在路径动态获取：

```python
SC_DIR = os.path.join(os.path.dirname(__file__), "SpeechCommands-sub", "SpeechCommands", "speech_commands_v0.02")
```

## 5. 运行实验

```powershell
# 方法一：激活环境后运行
conda activate rnn
python RNN/train_sc2.py

# 方法二：直接使用环境 Python 路径
& "C:\Users\USTCRAFT\.conda\envs\rnn\python.exe" RNN/train_sc2.py
```

## 6. 排查记录

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'torch'` | 新环境未安装 PyTorch | `pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `TorchCodec is required` | torchaudio 2.11+ 强制依赖 torchcodec | 改用 `soundfile` 手动加载音频，绕过 torchaudio 的 `load()` |
| `DLL load failed while importing _lzma` | librosa 依赖的 `pooch` 需要 `lzma`，但 conda 环境元数据损坏 | 改用 `soundfile` 替代 `librosa` |
| `DirectoryNotACondaEnvironmentError` | conda 环境元数据问题 | 直接用环境 Python 路径 `C:\Users\USTCRAFT\.conda\envs\rnn\python.exe` |
| FFmpeg 安装失败 (`UnicodeDecodeError`) | `gdk-pixbuf` 的 post-link 脚本在中文 Windows 下编码报错 | 不依赖 torchcodec，改用 soundfile 加载音频 |

## 7. 当前环境包版本

| 包名 | 版本 |
|------|------|
| Python | 3.13.13 |
| PyTorch | 2.11.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| torchvision | (由 PyTorch 配套安装) |
| soundfile | 最新 |
| librosa（备选） | 0.11.0 |

## 8. GPU 信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4070 (12GB) |
| Driver | 596.36 |
| CUDA Driver | 13.2（向下兼容低版本 CUDA） |
