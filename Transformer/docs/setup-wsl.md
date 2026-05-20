# WSL 环境配置手册

> 个人使用 WSL2 和 Miniconda 进行实验，请自行了解 WSL 和 Miniconda 的安装和使用方法。以下步骤假设你已经安装并配置好了 WSL 和 Miniconda。
> 假设 Windows 用户名为 `Lighte`，WSL 发行版为 Ubuntu 26.04，代码存放在 Windows 文件系统中的路径为 `C:\Users\Lighte\GitHub\AIBLAB\Transformer`。

如未特别说明，所有命令均在 WSL 终端中执行。

## 1. 打开终端并进入实验目录

WSL 中 Windows 文件系统挂载在 `/mnt/` 下，切换到代码所在目录：

```bash
cd AIBLAB/Transformer
```

## 2. 创建 Miniconda 虚拟环境

```bash
conda create -n transformer python=3.12 -y
```

## 3. 激活虚拟环境

```bash
conda activate transformer
```

此时，命令行提示符会显示 `(transformer)`，表示当前已激活该虚拟环境。

## 4. 升级 pip

```bash
python -m pip install --upgrade pip
```

## 5. 安装依赖

如果需要使用 WSL2 的 NVIDIA CUDA，请先确定 `nvidia-smi` 可用，然后根据 PyTorch 官方安装页面选择对应命令安装 `torch` 后再执行以下的安装命令。

```bash
python -m pip install -r requirements.txt
```

## 6. 检查环境

```bash
python - <<'PY'
import torch
import transformers
import datasets
import numpy
import pandas
import sklearn
import pyarrow

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("imports: ok")
PY
```

## 7. 运行代码

运行之前确保 WSL 家目录（`/home/<your_wsl_username>/`，即`~/`）下的 `model/` 存在 `tokenizer-bert-base-uncased/` 和 `IMDB_datasets/` 及其内容（从实验代码包复制过去）。

再看一眼 conda 环境是否正确激活，确认后运行：

```bash
python nlp_code.py
```

## 8. 查看结果

程序运行结束后，结果文件位于 Windows 文件系统中的路径为 `C:\Users\Lighte\GitHub\AIBLAB\Transformer\outputs\score.txt`。

## 9. 退出虚拟环境

```bash
conda deactivate
```