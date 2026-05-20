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
from pathlib import Path
# 按需自行添加
# ...

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
# DATA_DIR = ROOT / "IMDB_datasets" / "hf_imdb"
# TOKENIZER_DIR = ROOT / "tokenizer-bert-base-uncased"
DATA_DIR = Path.home() / "model/IMDB_datasets/hf_imdb"
TOKENIZER_DIR = Path.home() / "model/tokenizer-bert-base-uncased"
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


# Transformer模型定义
class TransformerSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, output_dim=100):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, output_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, output_dim))  # 简单位置编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.classifier = nn.Linear(output_dim, 2)  # 用于训练的辅助分类头

    def forward(self, input_ids, attention_mask):
        # 嵌入层
        x = self.embedding(input_ids) * (self.output_dim ** 0.5)
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformer编码
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)
        attn_mask = (attention_mask == 0)  # 转换为Transformer需要的mask格式
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)

        # 句子向量提取（取第一个token）
        sentence_vector = x[:, 0, :]
        projected = self.projection(sentence_vector)
        sentence_vector = torch.tanh(self.layer_norm(projected))

        # 分类输出
        logits = self.classifier(sentence_vector)
        return sentence_vector, logits


# 模型参数
nhead = N_HEAD
num_layers = NUM_LAYERS

# 数据准备
train_loader, test_loader, num_classes, vocab_size = prepare_data()
print(num_classes)

# 验证数据加载器是否正确
print(next(batch for batch in train_loader))


def easy_test_model():
    # 在训练前初步验证模型实现代码
    model = TransformerSentenceEncoder(output_dim=100, vocab_size=vocab_size).to(device)
    print("test model before training")
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        sentence_vector, logits = model(input_ids, attention_mask)
        print(sentence_vector)
        print(logits)
        break


easy_test_model()


# 模型训练
def train_model(vector_dim=100):
    model = TransformerSentenceEncoder(vocab_size=vocab_size, output_dim=vector_dim).to(device)

    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 预训练阶段
    print(f"Training vector_dim={vector_dim}...")

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
def eval_model(vector_dim=100):
    train_features = []
    test_features = []
    train_labels_list = []
    test_labels_list = []
    model = train_model(vector_dim)
    print(f"Eval vector_dim={vector_dim}...")
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
    acc_100 = eval_model(100)
    print(acc_100)
    acc_200 = eval_model(200)
    print(acc_200)
    end_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = OUTPUT_DIR / "score.txt"
    with open(score_path, "a+", encoding='utf-8') as f:
        f.write("\nResult Comparison:\n")
        f.write(f"100-dim Model Accuracy: {acc_100:.4f}\n")
        f.write(f"200-dim Model Accuracy: {acc_200:.4f}\n")
        f.write(f"cost time:{end_time - start_time :.4f} seconds\n")

    print("\nResult Comparison:\n")
    print(f"100-dim Model Accuracy: {acc_100:.4f}")
    print(f"200-dim Model Accuracy: {acc_200:.4f}")
    print(f"cost time:{end_time - start_time :.4f} seconds")
