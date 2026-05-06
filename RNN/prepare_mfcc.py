"""
MFCC 特征预处理脚本
为 THCHS-30 数据集中的每个 .wav 文件提取 13 维 MFCC 特征，
保存为 .npy 文件，存放在 data_thchs30/mfcc/ 目录下。
"""

import os
import librosa
import numpy as np
from tqdm import tqdm

# 配置路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_thchs30")
SPLITS = ["train", "dev", "test"]
SR = 16000
N_MFCC = 13

for split in SPLITS:
    wav_dir = os.path.join(DATA_DIR, split)
    mfcc_dir = os.path.join(DATA_DIR, "mfcc", split)
    os.makedirs(mfcc_dir, exist_ok=True)

    # 获取所有 .wav 文件
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    if not wav_files:
        print(f"[{split}] 未找到 .wav 文件，跳过。")
        continue

    print(f"[{split}] 共 {len(wav_files)} 个文件，开始提取 MFCC...")
    for wav_file in tqdm(wav_files, desc=f"Processing {split}"):
        wav_path = os.path.join(wav_dir, wav_file)
        try:
            # 加载音频，重采样至 16kHz
            y, sr = librosa.load(wav_path, sr=SR)
            # 提取 13 维 MFCC，形状为 (13, T)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            # 转置为 (T, 13) 以适应模型输入
            mfcc = mfcc.T  # shape: (T, 13)
            # 保存
            npy_path = os.path.join(mfcc_dir, wav_file.replace(".wav", ".npy"))
            np.save(npy_path, mfcc)
        except Exception as e:
            print(f"  处理 {wav_file} 时出错: {e}")

    print(f"[{split}] 完成！")

print("所有 MFCC 特征提取完毕！")
