# 录制语音测试 — 扩展实验 2

## 快速开始

```bash
# 1. 把手机录的 m4a 放到 RNN-NEW/infer_voice/ 目录下
# 2. 运行
python infer.py my_voice.m4a
# 3. 或者如果已经是 WAV
python infer.py my_voice.wav
```

## 流程

```
手机录音 (.m4a)
    │  ffmpeg 转换 (自动)
    ▼
16kHz 单声道 WAV
    │  librosa 提取 MFCC
    ▼
MFCC 特征 (13维)
    │  ASRModel (CNN + GRU) 推理
    ▼
CTC 解码 → 识别文本
```

## 依赖

```bash
pip install torch librosa soundfile numpy
```

另外需要安装 **ffmpeg**：
- **Windows**: `winget install ffmpeg` 或从 https://ffmpeg.org/download.html 下载
- **WSL Ubuntu**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

## 参数说明

```bash
python infer.py <音频文件> [选项]

选项:
  --model_path  预训练权重路径 (默认 ../g_0612_0.601)
  --cha2id      char2id 字典路径 (默认 ../cha2id.pth)
  --id2char     id2char 字典路径 (默认 ../id2char.pth)
  --true_text   录音的真实文本，用于计算 CER（字符错误率）
  --cpu         强制使用 CPU (默认自动使用 GPU)
```

### 示例（带真实文本对比）

```bash
python infer.py my_voice.wav --true_text "今天天气真不错"
```

输出：

```
==================================================
识别结果: 今天天气真不错
真实文本: 今天天气真不错
字错误率 (CER): 0.00%
==================================================
```

## 注意事项

1. 录音时尽量安静，语速适中、吐字清晰
2. 模型在 THCHS-30 数据集上预训练，对标准普通话识别效果较好
3. 测试集 CER 约 101% 说明模型过拟合较严重，自己的语音识别效果可能一般
4. 如果 ffmpeg 不在 PATH 中，可手动将 m4a 转为 16kHz 单声道 WAV 后再运行
