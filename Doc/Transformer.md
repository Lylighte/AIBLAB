# 基于 Transformer 的情感分析实验

---

## 1 背景知识

### 1.1 文本分类任务
文本分类是自然语言处理中的基础任务，其目标是使计算机自动理解文本内容并将其划分到预定义的类别中。

文本分类任务通常包含以下步骤。
第一步，文本预处理。对输入文本进行清洗和规范化处理，包括分词、去除停用词等。针对英文文本，还可能涉及词干提取。同时，通过词嵌入技术将文本转换为数值向量表示，以便模型处理。
第二步，特征提取。传统方法依赖人工设计的特征，如 TF-IDF、词袋模型。深度学习方法则通常采用 RNN、Transformer 或预训练语言模型自动捕获文本的语义和上下文特征。
第三步，分类决策。特征提取完成后，传统机器学习使用分类器进行类别预测；深度学习中则常通过全连接层结合 softmax 函数输出类别概率分布，最终选择概率最高的类别作为分类结果。

需要明确的核心思路是：文本分类的本质不是让模型直接处理字符串，而是先将自然语言文本转换为模型可计算的张量，再由模型学习文本中的语义特征，最终完成分类。

### 1.2 情感分类任务
情感分类是文本分类的典型应用之一，旨在识别文本中表达的情感倾向，如正面、负面、中性，或进一步细化至喜悦、愤怒、悲伤等具体情感类别。

其应用场景广泛。在电商评论分析中，可用于判断用户对商品的情感倾向，辅助商家改进产品或优化推荐。在社交媒体监测中，可追踪公众对品牌或事件的情感态度，服务于舆情分析或危机预警。在影视评价挖掘中，可分析影评的情感极性，预测影片口碑或观众偏好。

情感分类任务面临的技术挑战主要有三方面。一是语义复杂性，文本中可能包含反讽、隐喻等复杂表达，需依赖上下文才能准确理解。二是细粒度分析，部分任务需区分情感指向的具体对象，例如“相机画质好，但电池续航差”中对不同方面的情感判定不同。三是领域适配性，不同领域的情感表达差异显著，需借助迁移学习或领域自适应技术提升泛化能力。

本实验采用最典型的情感二分类设定，数据来自 IMDB 英文电影评论，标签 1 表示正面评论，标签 0 表示负面评论。

### 1.3 文本预处理
文本预处理的作用在于清洗噪声、统一格式规范，从而消除文本歧义、提升语义表征质量。对缺失文本、异常符号及冗余信息的处理有助于增强数据可靠性，保障模型鲁棒性。

面向文本分类任务的预处理一般流程如下。首先进行文本清洗，去除 HTML 标签、特殊符号、冗余空格等噪声。其次进行分词或子词切分，将句子切分为词或子词单元，如 BERT 使用的 WordPiece 分词。随后进行词向量映射，将文本转换为数值向量，如 Word2Vec、GloVe 或 BERT 嵌入。之后进行长度标准化，将文本截断或填充至统一长度。此外还有若干可选步骤，包括数据增强、数据归一化、停用词过滤等。最后是数据打乱、批量化和并行化处理，以确保训练过程高效稳定。

在本实验中，预处理重点集中在三个环节：
(1) 使用 `pandas.read_parquet()` 读取本地 IMDB 数据；
(2) 使用 `AutoTokenizer` 完成分词、截断和 padding；
(3) 使用 `TensorDataset` 和 `DataLoader` 将文本张量、mask 与标签组织为批次。

### 1.4 用于文本分类的 Transformer 模型
适用于自然语言处理的神经网络架构需满足两项要求。一是序列建模能力，即无论目标词出现在文本何处，模型均应能捕捉其与上下文的关系，具备长距离依赖建模能力。二是全局语义感知，即模型需从整体文本中提取层次化语义特征，而非局限于局部片段，最终通过聚合全局信息完成分类预测。

早期基于 RNN 或 CNN 的模型受限于局部窗口或单向信息流动，难以有效建模长文本依赖。自注意力机制的提出使 Transformer 架构得以通过并行化计算和位置编码同时解决长程依赖与计算效率问题。BERT、GPT 等预训练模型则进一步通过自监督学习在大规模语料上获取通用语义表征，显著提升了文本分类任务的性能。

本实验从基础 Transformer 入手，学习构建文本分类模型的核心流程，涵盖嵌入层、注意力头、全连接分类器等组件。在掌握其原理后，可进一步迁移至预训练模型，理解微调策略与迁移学习对性能的提升作用。

需要特别说明的是：本实验代码使用了 BERT 的 tokenizer 进行分词，但并未直接调用 BERT 模型本身。实际训练的是自行定义的 `nn.TransformerEncoder` 文本编码器。

### 1.5 情感分类评价指标
情感分类任务包括二分类和多分类。二分类将情感极性划分为两个类别，如影评中的好评与差评；多分类则可进一步划分为喜悦、悲伤、愤怒、惊讶等类别，或按情感强度分级。

评价指标是衡量模型性能的核心依据。情感二分类任务中的主流指标包括准确率、精确率、召回率和 F1 值。本实验为简化流程，主要采用准确率作为核心指标，计算方式为：

准确率 = (正确预测的文本数量 / 待预测文本总数) × 100%

本实验最终比较的是 100 维和 200 维句向量在 IMDB 测试集上的 accuracy。其余评价指标不在本次实验范围内展开，但需明确 accuracy 是本实验的主指标。

### 1.6 神经网络超参数调整
超参数是在模型训练前需手动设置的参数，直接影响模型的训练动态、收敛效果与泛化性能。情感分类任务中的关键超参数包括学习率、训练轮次、正则化系数、词嵌入维度、注意力头数等。

本实验需关注的超参数如下。学习率 `lr` 的典型取值范围为 0.1 至 1e-5，过大可能导致优化震荡甚至发散，过小则收敛缓慢。`batch_size` 常取 32、64、128 或 256，大批次训练速度快且梯度稳定，但泛化性可能下降。词嵌入维度常取 100、200 或 300，维度越高语义表征能力越强，但计算成本增加且易过拟合。注意力头数常取 2、4 或 8，多头机制可增强语义多样性建模，但同时增加参数和计算开销。训练轮次 `epoch` 常取 10 至 100，过少导致欠拟合，过多则可能引发过拟合。

本实验固定大多数参数不变，重点比较词嵌入维度分别为 100 和 200 时的结果。控制变量的意义在于单独观察 Embedding Size 对准确率的影响。

### 1.7 IMDB 数据集
本实验使用 IMDB Large Movie Review Dataset，这是自然语言处理领域经典的二元情感分类数据集，也是评估文本分类模型性能的基准数据集之一。

该数据集的主要特性如下。数据规模为 50,000 条英文电影评论。类别标注方面，`label=1` 表示正面评论，`label=0` 表示负面评论，另有 `label=-1` 的无监督数据。数据划分为训练集 25,000 条（正面和负面各 12,500 条）、测试集 25,000 条（正面和负面各 12,500 条）、无监督数据 50,000 条。数据来源为 Internet Movie Database。文件格式在本实验包中以 parquet 格式存储，可由 pandas 直接读取。

本实验无需现场下载数据。实验包已包含 `IMDB_datasets/hf_imdb/` 目录和本地 tokenizer，重点在于理解数据如何被读入、分词、批量化并送入模型。


## 2 实验目的与代码讲解

### 2.1 实验目的
本实验的目的分为三个层次。
第一，掌握深度学习的基本步骤，包括定义网络的流程、损失函数的作用、优化算法的作用。
第二，理解模型评估与改进，包括模型超参数调节、模型微调的基本过程、可视化分析方法。
第三，掌握基于 Transformer 的情感分析设计流程，包括基于 Transformer 的句子向量生成方法、文本数据预处理方法、词嵌入维度对结果的影响。

概括而言，本实验的目标是完成一条从 IMDB 文本、tokenizer、Transformer Encoder、句向量抽取到逻辑回归分类评估的完整情感分析流水线，并比较 100 维和 200 维句向量的效果。

### 2.2 实验主线
本实验的完整数据处理流程如下：
```text
raw text
-> BERT tokenizer
-> input_ids + attention_mask
-> token embedding + position encoding
-> TransformerEncoder
-> 第一个 token 位置的句向量
-> 辅助分类头训练 encoder
-> 抽取 train/test 句向量
-> StandardScaler + LogisticRegression
-> accuracy
```

需要注意以下三点。
(1) `nlp_code.py` 是填空式模板，代码中的 `...` 为需要补全的实验内容。
(2) 本实验仅使用 BERT 的分词器，并非微调 BertModel。
(3) Transformer 编码器、训练逻辑、句向量抽取及评估流程均需自行补全。

### 2.3 代码结构速览
`nlp_code.py` 由以下六个模块组成。

(1) 导入依赖与超参数：导入 PyTorch、Transformers、sklearn、pandas 等库，需明确各库的职责划分。
(2) `prepare_data()`：负责加载 parquet 数据、执行分词、构造 DataLoader。核心内容是文本如何转换为 `input_ids`、`attention_mask` 和 `label`。
(3) `TransformerSentenceEncoder`：定义 Transformer 句向量模型，包含嵌入层、位置编码、TransformerEncoder 及分类头。
(4) `easy_test_model()`：训练前执行一次前向传播检查，验证张量形状与 forward 过程是否正常。
(5) `train_model()`：训练 Transformer 编码器，涉及 loss 计算、optimizer 参数更新、scheduler 学习率调整及反向传播。
(6) `eval_model()`：抽取句向量并使用逻辑回归进行评估，以句向量为特征，accuracy 为评估结果。

最终在 `__main__` 中分别运行 100 维和 200 维实验，对比 Embedding Size 对性能的影响。

### 2.4 数据加载与预处理：prepare_data()
本节对应文本预处理与模型输入格式转换，按以下顺序展开。

首先，读取本地数据集：
```python
train_data = pd.read_parquet(DATA_DIR / "train-00000-of-00001.parquet")
test_data = pd.read_parquet(DATA_DIR / "test-00000-of-00001.parquet")
```

其次，从 DataFrame 中提取文本列与标签列：
```python
train_list = train_data["text"].to_list()
train_labels = torch.tensor(train_data["label"].tolist())
```

然后，加载本地 BERT tokenizer：
```python
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
```

对文本进行标记化并转换为模型输入格式：
```python
train_encodings = tokenizer(
    train_list,
    padding=True,
    max_length=512,
    truncation=True,
    return_tensors="pt",
)
```

最后，构建 TensorDataset 和 DataLoader：
```python
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_labels,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
```

此处有一个重要的实现细节。tokenizer 输出的 `input_ids` 是 token 的编号，`attention_mask` 用于区分真实 token 与 padding。在 Hugging Face 中，`attention_mask=1` 表示真实 token，`0` 表示 padding；而 PyTorch 的 `src_key_padding_mask` 语义相反，True 表示需要屏蔽。因此代码中需执行 `attention_mask == 0` 的转换。

本节中需补全的填空点主要涉及测试集的对称处理，包括 `test_data`、`test_list`、`test_encodings`、`test_labels`、`test_dataset` 和 `test_loader`。

### 2.5 Transformer 模型构建
本模板中的模型类定义如下：
```python
class TransformerSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, output_dim=100):
        ...
    def forward(self, input_ids, attention_mask):
        ...
```

`__init__()` 负责初始化网络结构，核心组件包括：`nn.Embedding(vocab_size, output_dim)` 将 token id 映射为向量；位置编码为每个 token 注入位置信息；`nn.TransformerEncoderLayer` 定义单层 Transformer Encoder；`nn.TransformerEncoder` 堆叠多层 Encoder；`nn.Linear` 和 `nn.LayerNorm` 对句向量执行映射与归一化；`classifier` 为辅助分类头，利用情感标签监督 encoder 学习句向量。

`forward()` 定义数据的前向传播过程，其流程为：
```text
input_ids
-> embedding
-> 加位置编码
-> Transformer 编码
-> 取第一个 token 的输出作为句向量
-> projection + LayerNorm + tanh
-> classifier 输出 logits
```

关于位置编码的必要性：Transformer 的自注意力机制本身不具备感知 token 顺序的能力，因此必须通过位置编码显式注入位置信息。模型输出每个 token 的上下文表示后，取第一个 token 的输出作为整句话的句向量，与 BERT 中 `[CLS]` 的用法一致。

实现时有三个易错点需注意。
(1) `output_dim` 必须能被 `N_HEAD` 整除。本实验 `N_HEAD=4`，100 和 200 均满足该条件。
(2) `nn.TransformerEncoderLayer` 默认输入形状为 `(seq_len, batch_size, hidden_dim)`，而 embedding 输出通常为 `(batch_size, seq_len, hidden_dim)`，中间需执行转置。
(3) `src_key_padding_mask` 的形状应为 `(batch_size, seq_len)`。

### 2.6 训练前检查：easy_test_model()
正式训练前需验证模型代码能否正常执行前向传播。模板中的 `easy_test_model()` 即用于此目的。

具体步骤为：从 `train_loader` 中取出一个 batch，将 `input_ids`、`attention_mask`、`labels` 迁移至同一 device，调用模型获取 `sentence_vector` 和 `logits`，打印输出以检查前向传播是否正常运行及张量形状是否符合预期。

此步骤的意义在于：模型训练耗时较长，在启动完整训练前先用单个 batch 检查 forward，是排查 shape 错误最为高效的方式。

### 2.7 模型训练：train_model(vector_dim)
训练部分的核心代码如下：
```python
_, logits = model(input_ids, attention_mask)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
scheduler.step()
```

执行流程如下。每次调用 `train_model(vector_dim)` 均会创建并训练一个新的 Transformer 编码器。`criterion` 采用交叉熵损失函数，比较模型输出的 `logits` 与真实标签 `labels`。`optimizer` 负责更新模型参数。`scheduler` 动态调整学习率。每个 epoch 结束后打印平均 loss，通过观察 loss 是否正常下降来初步判断训练过程是否正常。

需要强调的是：训练阶段的辅助分类头并非最终评估的唯一依据。其作用是利用情感标签监督 encoder 学习生成更具判别力的句向量。后续评估阶段将把 encoder 作为特征提取器使用。

### 2.8 模型评估：eval_model(vector_dim)
评估流程依次包括以下步骤。
(1) 调用 `train_model(vector_dim)` 完成 encoder 训练。
(2) 使用训练好的 encoder 抽取训练集句向量。
(3) 使用同一 encoder 抽取测试集句向量。
(4) 将句向量拼接为 NumPy 数组。
(5) 使用 `StandardScaler` 对特征执行标准化。
(6) 使用 `LogisticRegression(max_iter=1000)` 在训练集句向量上训练线性分类器。
(7) 在测试集句向量上进行预测并计算 `accuracy_score`。

采用逻辑回归的原因在于：将 Transformer encoder 生成的句向量视为特征，再使用统一的线性分类器评估句向量的线性可分性，使 100 维与 200 维的比较更为公平，排除了分类器差异的干扰。

实现时需注意以下几点。评估阶段应调用 `model.eval()` 切换至评估模式。特征抽取应在 `torch.no_grad()` 上下文中执行，避免不必要的梯度计算。句向量需先执行 `.cpu()` 再拼接并转换为 NumPy 数组。`scaler` 应在训练集上调用 `fit_transform`，在测试集上仅调用 `transform`，不可在测试集上重新 fit。

### 2.9 执行实验与结果对比
主函数中分别执行两组实验：
```python
acc_100 = eval_model(100)
acc_200 = eval_model(200)
```
结果写入 `outputs/score.txt`。

获取结果后，应围绕以下问题进行分析。
(1) 向量维度增加为何可能提升模型性能？是否存在性能上限？
(2) 保持其他参数一致的意义是什么？如何验证参数设置的合理性？
(3) 使用逻辑回归作为分类器有哪些优势与局限？
(4) 实验中哪些步骤对最终结果影响最大？原因是什么？

### 2.10 作业提交要求
提交内容包括以下几项。
(1) 修改后的完整代码，应包含网络架构定义、数据加载与预处理、模型训练与测试过程。
(2) 超参数对比实验结果表格，至少包括 100 维和 200 维句向量的 accuracy。
(3) 对数据处理与模型实现过程的简要描述，以及所使用的数据结构说明。
(4) 遇到的报错信息与解决方案记录；若未遇到报错，需说明所采用的调试策略。
(5) 实验报告应重点突出实验目标、方法、结果及不同配置下的性能对比分析。