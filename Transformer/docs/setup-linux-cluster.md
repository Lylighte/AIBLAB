# Linux 集群环境配置手册

## 1. 上传或解压实验包

在集群工作目录中放置实验包，然后进入实验包根目录：

```bash
cd student_package_transformer_sentiment
```

## 2. 检查 Python

```bash
python --version
```

如果 `python` 不可用，请执行：

```bash
python3 --version
```

后续步骤中使用当前可用的 Python 命令。

## 3. 创建虚拟环境

如果集群使用 `python`：

```bash
python -m venv .venv
```

如果集群使用 `python3`：

```bash
python3 -m venv .venv
```

## 4. 激活虚拟环境

```bash
source .venv/bin/activate
```

## 5. 升级 pip

```bash
python -m pip install --upgrade pip
```

## 6. 安装依赖

```bash
python -m pip install -r requirements.txt
```

如果需要在 Linux 集群上使用 NVIDIA CUDA，请根据 PyTorch 官方安装页面选择对应命令安装 `torch`，然后再执行上一条依赖安装命令。

## 7. 检查环境

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

## 8. 运行模板代码

补全 `nlp_code.py` 中的 `...` 后，执行：

```bash
python nlp_code.py
```

## 9. 查看结果

程序运行结束后，结果文件位于：

```text
outputs/score.txt
```

## 10. 退出虚拟环境

```bash
deactivate
```
