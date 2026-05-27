#!/bin/bash
# IMDB Transformer 调参实验脚本 —— 在 WSL 中运行
# 用法: bash run_experiments.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS="outputs/score.txt"
echo "==========================================" | tee -a "$RESULTS"
echo "Experiment Start: $(date)" | tee -a "$RESULTS"
echo "==========================================" | tee -a "$RESULTS"

# ====== 第一轮实验（已归档，lr>=5e-4 未收敛） ======
# 学习率对比: lr=1e-4(0.8240) lr=5e-4(0.6406) lr=1e-3(0.6261)
# 批次大小对比: bs=32(0.6306) bs=64(0.6576) bs=128(内存交换)
# 训练轮数对比: ep=5(0.6198) ep=10(0.6338) ep=20(未完成)

echo ""
echo "========== 学习率调优（第二轮） ==========" | tee -a "$RESULTS"
for lr in 5e-5 1e-4 2e-4; do
    echo ">>> lr=$lr batch=64 epochs=10 emb_dim=100" | tee -a "$RESULTS"
    python nlp_code.py --lr "$lr" --batch_size 64 --epochs 10 --emb_dim 100
done

echo ""
echo "==========================================" | tee -a "$RESULTS"
echo "Experiment End: $(date)" | tee -a "$RESULTS"
echo "==========================================" | tee -a "$RESULTS"