# 环境配置与依赖导入
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import math
from pathlib import Path
# 按需自行添加


# 超参数
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-4
N_HEAD = 4
# embed_dim must be divisible by num_heads
NUM_LAYERS = 6

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "IMDB_datasets" / "hf_imdb"
TOKENIZER_DIR = ROOT / "tokenizer-bert-base-uncased"
OUTPUT_DIR = ROOT / "outputs"


# 数据加载与预处理
def prepare_data():
    # 加载数据集
    train_data = pd.read_parquet(DATA_DIR / "train-00000-of-00001.parquet")
    test_data = pd.read_parquet(DATA_DIR / "test-00000-of-00001.parquet")

    # 将DataFrame格式数据中text列提取为文字列表
    train_list = train_data["text"].to_list()
    test_list = test_data["text"].to_list()

    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    # 标记化文本并将其转换为输入格式
    train_encodings = tokenizer(train_list, padding=True, max_length=512,
                                truncation=True, return_tensors='pt')
    test_encodings = tokenizer(test_list, padding=True, max_length=512,
                               truncation=True, return_tensors='pt')

    # 标签
    train_labels_list = train_data["label"].tolist()
    train_labels = torch.tensor(train_labels_list)
    test_labels_list = test_data["label"].tolist()
    test_labels = torch.tensor(test_labels_list)

    # 构建 DataLoader
    train_dataset = TensorDataset(train_encodings['input_ids'],
                                  train_encodings['attention_mask'], train_labels)
    test_dataset = TensorDataset(test_encodings['input_ids'],
                                 test_encodings['attention_mask'], test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 分类数（num_classes）：标签的唯一值数量
    num_classes = len(train_data["label"].unique())

    # 词汇表大小（vocab_size）：BERT分词器的词汇表大小
    vocab_size = tokenizer.vocab_size

    return train_loader, test_loader, num_classes, vocab_size


# 简单位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Transformer模型定义
class TransformerSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim=100):
        """
        Args:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入向量维度 / Transformer隐藏层维度（d_model）
                       这就是实验对比的变量：100维 vs 200维
            output_dim: 最后一个线性投影层的输出维度（投影到句子向量空间）
                        注意：这不是实验对比的变量，保持不变即可
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=N_HEAD, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.projection = nn.Linear(embed_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.classifier = nn.Linear(output_dim, 2)  # 用于训练的辅助分类头

    def forward(self, input_ids, attention_mask):
        # 嵌入层
        x = self.embedding(input_ids)                    # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)                           # (batch, seq_len, embed_dim)

        # Transformer编码
        # 已设置 batch_first=True，输入形状为 (batch, seq_len, embed_dim)
        attn_mask = (attention_mask == 0)                 # 转换为Transformer需要的mask格式（True=遮罩位置）
        x = self.transformer(x, src_key_padding_mask=attn_mask)
                                                          # (batch, seq_len, embed_dim)

        # 句子向量提取（取第一个token [CLS]）
        sentence_vector = x[:, 0, :]                      # (batch, embed_dim)
        projected = self.projection(sentence_vector)      # (batch, output_dim)
        sentence_vector = torch.tanh(self.layer_norm(projected))  # (batch, output_dim)

        # 分类输出
        logits = self.classifier(sentence_vector)         # (batch, 2)
        return sentence_vector, logits


# 模型参数
nhead = N_HEAD
num_layers = NUM_LAYERS

# 数据准备
train_loader, test_loader, num_classes, vocab_size = prepare_data()
print(f"num_classes: {num_classes}")
print(f"vocab_size: {vocab_size}")

# 验证数据加载器是否正确
for batch in train_loader:
    input_ids, attention_mask, labels = batch
    print(f"input_ids shape: {input_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"labels[:10]: {labels[:10]}")
    break


def easy_test_model():
    # 在训练前初步验证模型实现代码
    model = TransformerSentenceEncoder(embed_dim=100, output_dim=100, vocab_size=vocab_size).to(device)
    print("test model before training")
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        sentence_vector, logits = model(input_ids, attention_mask)
        print(f"sentence_vector shape: {sentence_vector.shape}")
        print(f"logits shape: {logits.shape}")
        print(f"sentence_vector[0]: {sentence_vector[0]}")
        print(f"logits[0]: {logits[0]}")
        break


easy_test_model()


# 模型训练
def train_model(embed_dim=100):
    model = TransformerSentenceEncoder(vocab_size=vocab_size, embed_dim=embed_dim, output_dim=100).to(device)

    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 预训练阶段
    print(f"Training embed_dim={embed_dim}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            _, logits = model(input_ids, attention_mask)

            # 损失函数
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")
    return model


# 模型评估
def eval_model(embed_dim=100):
    train_features = []
    test_features = []
    train_labels_list = []
    test_labels_list = []
    model = train_model(embed_dim)
    print(f"Eval embed_dim={embed_dim}...")
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            vectors, _ = model(input_ids, attention_mask)
            train_features.append(vectors.cpu())
            train_labels_list.append(labels.cpu())
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            vectors, _ = model(input_ids, attention_mask)
            test_features.append(vectors.cpu())
            test_labels_list.append(labels.cpu())
    train_features = torch.cat(train_features, dim=0).numpy()
    train_labels = torch.cat(train_labels_list, dim=0).numpy()
    test_features = torch.cat(test_features, dim=0).numpy()
    test_labels = torch.cat(test_labels_list, dim=0).numpy()
    # 归一化特征
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    # 逻辑回归分类器
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(train_features, train_labels)
    test_preds = lr_clf.predict(test_features)
    # 评估结果
    test_accuracy = accuracy_score(test_labels, test_preds)
    return test_accuracy


# 执行实验
if __name__ == "__main__":
    start_time = time.time()

    # 对比实验：变化的是 embed_dim（词嵌入向量维度），而不是 output_dim
    print("=" * 60)
    print("实验：对比 100维 vs 200维 词嵌入向量")
    print("=" * 60)

    acc_100 = eval_model(embed_dim=100)
    print(f"\n100-dim Embedding Accuracy: {acc_100:.4f}")

    acc_200 = eval_model(embed_dim=200)
    print(f"200-dim Embedding Accuracy: {acc_200:.4f}")

    end_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = OUTPUT_DIR / "score.txt"
    with open(score_path, "a+", encoding='utf-8') as f:
        f.write("\nResult Comparison:\n")
        f.write(f"100-dim Embedding Model Accuracy: {acc_100:.4f}\n")
        f.write(f"200-dim Embedding Model Accuracy: {acc_200:.4f}\n")
        f.write(f"cost time:{end_time - start_time :.4f} seconds\n")

    print("\nResult Comparison:\n")
    print(f"100-dim Embedding Model Accuracy: {acc_100:.4f}")
    print(f"200-dim Embedding Model Accuracy: {acc_200:.4f}")
    print(f"cost time:{end_time - start_time :.4f} seconds")