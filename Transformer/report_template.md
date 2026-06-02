# 基于 Transformer 的 IMDB 情感分析实验报告

**组号：** 第X组 &nbsp;&nbsp;&nbsp;&nbsp; **姓名：** XXX &nbsp;&nbsp;&nbsp;&nbsp; **学号：** PB24XXXXXX

---

## 1 实验基本信息

### 1.1 实验目的

1. 掌握深度学习的基本步骤，包括定义网络的流程、损失函数的作用、优化算法的作用。
2. 理解模型评估与改进，包括模型超参数调节、模型微调的基本过程。
3. 掌握基于 Transformer 的情感分析设计流程，包括基于 Transformer 的句子向量生成方法、文本数据预处理方法、词嵌入维度对结果的影响。

### 1.2 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | WSL2 (Ubuntu) / Linux 集群 |
| GPU | NVIDIA GeForce RTX 4070 (12GB) |
| Python | 3.13 |
| 框架 | PyTorch, HuggingFace Transformers, scikit-learn |
| 数据集 | IMDB Large Movie Review Dataset (25,000 训练 + 25,000 测试) |

### 1.3 实验流程

```text
raw text
→ BERT tokenizer
→ input_ids + attention_mask
→ token embedding + position encoding
→ TransformerEncoder
→ 第一个 token 位置的句向量
→ 辅助分类头训练 encoder
→ 抽取 train/test 句向量
→ StandardScaler + LogisticRegression
→ accuracy
```

---

## 2 关键代码说明

### 2.1 模型结构

```python
class TransformerSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=N_HEAD, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.projection = nn.Linear(embed_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.classifier = nn.Linear(output_dim, 2)
```

> **说明：** `embed_dim` 为词嵌入向量维度 / Transformer d_model，即实验对比变量（100 vs 200）；`output_dim` 为投影层输出维度，固定为 100 不参与对比。

### 2.2 前向传播

```python
def forward(self, input_ids, attention_mask):
    x = self.embedding(input_ids)                    # (batch, seq_len, embed_dim)
    x = self.pos_encoder(x)
    attn_mask = (attention_mask == 0)
    x = self.transformer(x, src_key_padding_mask=attn_mask)
    sentence_vector = x[:, 0, :]                     # 取 [CLS] token
    projected = self.projection(sentence_vector)
    sentence_vector = torch.tanh(self.layer_norm(projected))
    logits = self.classifier(sentence_vector)
    return sentence_vector, logits
```

### 2.3 评估流程

训练完成后，利用 encoder 抽取句向量，使用逻辑回归进行分类评估，以排除分类器差异的干扰。

---

## 3 实验结果与分析

### 3.1 学习率调优

固定 `batch_size=32, epochs=10, embed_dim=100`

| 学习率 | Loss（最后 epoch） | 准确率 | 备注 |
|--------|-------------------|--------|------|
| 5e-5   |                   |        |      |
| **1e-4** |                 | **最佳** |      |
| 2e-4   |                   |        |      |

### 3.2 批次大小对比

固定 `lr=1e-4, epochs=10, embed_dim=100`

| 批次大小 | Loss（最后 epoch） | 准确率 | 耗时（秒） |
|----------|-------------------|--------|-----------|
| 32       |                   |        |           |
| 64       |                   |        |           |

### 3.3 训练轮数对比

固定 `lr=1e-4, batch_size=32, embed_dim=100`

| 训练轮数 | Loss（最后 epoch） | 准确率 | 耗时（秒） |
|----------|-------------------|--------|-----------|
| 5        |                   |        |           |
| 10       |                   |        |           |

### 3.4 词嵌入维度对比（核心实验）

固定 `lr=1e-4, batch_size=32, epochs=10`

| 嵌入维度 | 句向量维度 | Loss（最后 epoch） | 准确率 | 耗时（秒） |
|----------|-----------|-------------------|--------|-----------|
| 100      | 100       |                   |        |           |
| 200      | 100       |                   |        |           |

> **注：** 句向量维度统一为 `output_dim=100`，仅改变词嵌入层和 Transformer 隐藏层维度。

### 3.5 结果可视化

<!-- 可以放柱状图：不同学习率/批次大小/嵌入维度的准确率对比，建议用 matplotlib 或 Excel 生成 -->

### 3.6 结果分析

1. **学习率影响：** ...
2. **批次大小影响：** ...
3. **训练轮数影响：** ...
4. **嵌入维度影响：** 100 维 vs 200 维的准确率差异及原因分析。

---

## 4 总结与思考

### 4.1 实验结论

（围绕以下问题展开）
- 向量维度增加为何可能提升模型性能？是否存在性能上限？
- 保持其他参数一致的意义是什么？如何验证参数设置的合理性？
- 使用逻辑回归作为分类器有哪些优势与局限？

### 4.2 遇到的问题与解决方案

| 问题 | 原因 | 解决方法 |
|------|------|---------|
|      |      |         |

### 4.3 改进方向

- ...

---

## 参考文献

1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
2. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
3. Maas, A., et al. "Learning Word Vectors for Sentiment Analysis." ACL 2011.