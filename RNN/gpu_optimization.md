# GPU 利用率优化方案

## 问题分析

运行 `train_sc2.py` 时，CPU 占用高而 GPU 利用率低，原因是 **数据预处理管线在 CPU 上串行执行**。

### 当前瓶颈链路

```
DataLoader (CPU)
  ├── sf.read()         → 加载 .wav 文件 (CPU密集: I/O)
  ├── mel_transform()   → 梅尔频谱 FFT (CPU密集: 计算)
  ├── pad_sequence()    → 序列填充 (CPU)
  └── .to(device)       → 数据传输至 GPU
         ↓
GPU 前向/反向传播 (GPU, 很快)
         ↓
    等待 CPU 准备下一批数据 ← 瓶颈在这里！
```

`MelSpectrogram` 包含 **FFT (快速傅里叶变换)**，是 CPU 上最密集的计算操作。GPU 算完一个 batch 后，CPU 还没准备好下一批，导致 GPU 空转。

---

## 优化方案对比

### 方案 1: 将 MelSpectrogram 移到 GPU 上计算 ✅（推荐）

**做法：** 在训练循环内，先将原始波形传到 GPU，再调用 `mel_transform`。

```
DataLoader (CPU)
  ├── sf.read()         → 加载 .wav (仅 I/O)
  └── 原始波形 → GPU
         ↓
GPU
  ├── mel_transform()   → 梅尔频谱 (GPU并行FFT, 极快)
  ├── pad_sequence()    → 序列填充
  ├── GRU 前向传播
  └── 反向传播
```

**优点：** 梅尔频谱计算从 CPU 卸载到 GPU，并行加速明显
**缺点：** 需要改写 collate_fn，传原始波形到 GPU

### 方案 2: DataLoader 多进程 + 预取

```python
DataLoader(..., num_workers=4, pin_memory=True, prefetch_factor=4)
```

**优点：** 多进程并行处理 CPU 任务
**缺点：** 梅尔频谱计算仍在 CPU，只能缓解不能根除；Windows 下 `num_workers>0` 有兼容性问题

### 方案 3: 启用 cuDNN Benchmark + 关闭 Deterministic

```python
torch.backends.cudnn.benchmark = True
# 移除 deterministic = True (训练阶段不需要)
```

**优点：** 零成本提速，cuDNN 自动选择最快卷积/循环算法
**缺点：** 单独使用效果有限，需配合方案 1

### 方案 4: 编译模型 (torch.compile)

```python
model = torch.compile(model)
```

**优点：** 融合 GPU 算子，减少 kernel launch 开销
**缺点：** 首次运行需要编译时间；对 GRU 提升幅度较小

### 方案 5: 增大 batch_size

增大 `batch_size` 可提高 GPU 计算密度，减少 CPU-GPU 同步次数

---

## 综合推荐方案

```
✅ 方案 1 (MelSpec→GPU) + ✅ 方案 3 (cuDNN benchmark) + ✅ 方案 5 (更大 batch)
```

---

## 预期效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| CPU 占用 | ~80-100% | ~20-40% |
| GPU 利用率 | ~10-30% | ~60-90% |
| 每轮耗时 | ~6-8s | ~2-4s |
| 瓶颈 | CPU 梅尔频谱 | GPU 计算 |
