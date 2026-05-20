# macOS 环境配置手册

## 1. 打开终端并进入实验包目录

在终端中进入实验包根目录：

```bash
cd student_package_transformer_sentiment
```

## 2. 创建虚拟环境

```bash
python3 -m venv .venv
```

## 3. 激活虚拟环境

```bash
source .venv/bin/activate
```

## 4. 升级 pip

```bash
python -m pip install --upgrade pip
```

## 5. 安装依赖

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
print("mps:", torch.backends.mps.is_available())
print("imports: ok")
PY
```

## 7. 运行模板代码

补全 `nlp_code.py` 中的 `...` 后，执行：

```bash
python nlp_code.py
```

## 8. 查看结果

程序运行结束后，结果文件位于：

```text
outputs/score.txt
```

## 9. 退出虚拟环境

```bash
deactivate
```
