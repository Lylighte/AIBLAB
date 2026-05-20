# 基于 Transformer 的情感分析实验包

## 1. 目录结构

```text
student_package_transformer_sentiment/
├── nlp_code.py
├── requirements.txt
├── docs/
│   ├── setup-macos.md
│   ├── setup-windows.md
│   └── setup-linux-cluster.md
├── IMDB_datasets/hf_imdb/
│   ├── train-00000-of-00001.parquet
│   ├── test-00000-of-00001.parquet
│   └── unsupervised-00000-of-00001.parquet
├── tokenizer-bert-base-uncased/
└── outputs/
```

## 2. 使用顺序

1. 解压实验包，或将实验包上传到集群工作目录。
2. 进入实验包根目录。
3. 根据操作系统阅读对应手册：
macOS：`docs/setup-macos.md`
Windows：`docs/setup-windows.md`
Linux 集群：`docs/setup-linux-cluster.md`

## 3. 说明

1. `nlp_code.py` 是实验模板，代码中的 `...` 需要自行补全。
2. 数据集和 tokenizer 已包含在实验包中，无需单独下载。
3. 模板代码使用相对路径，请保持 `nlp_code.py`、`IMDB_datasets/`、`tokenizer-bert-base-uncased/` 位于同一实验包根目录下。
4. 运行结果默认写入 `outputs/score.txt`。
