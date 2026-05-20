# Windows 环境配置手册

## 1. 打开 PowerShell 并进入实验包目录

```powershell
cd student_package_transformer_sentiment
```

## 2. 检查 Python

先执行：

```powershell
py --version
```

如果 `py` 不可用，再执行：

```powershell
python --version
```

后续步骤中使用当前可用的命令。

## 3. 创建虚拟环境

如果可用命令是 `py`：

```powershell
py -3 -m venv .venv
```

如果可用命令是 `python`：

```powershell
python -m venv .venv
```

## 4. 激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，请先在当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

然后再次执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 5. 升级 pip

```powershell
python -m pip install --upgrade pip
```

## 6. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

如果需要在 Windows 上使用 NVIDIA CUDA，请根据 PyTorch 官方安装页面选择对应命令安装 `torch`，然后再执行上一条依赖安装命令。

## 7. 检查环境

```powershell
python -c "import torch, transformers, datasets, numpy, pandas, sklearn, pyarrow; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('imports: ok')"
```

## 8. 运行模板代码

补全 `nlp_code.py` 中的 `...` 后，执行：

```powershell
python nlp_code.py
```

## 9. 查看结果

程序运行结束后，结果文件位于：

```text
outputs\score.txt
```

## 10. 退出虚拟环境

```powershell
deactivate
```
