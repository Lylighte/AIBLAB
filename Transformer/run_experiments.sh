#!/bin/bash
# IMDB Transformer 调参实验脚本 —— 在 WSL 中运行
# 用法: bash run_experiments.sh

set -e
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS="outputs/score.txt"
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

echo "==========================================" | tee -a "$RESULTS"
echo "Experiment Start: $(date)" | tee -a "$RESULTS"
echo "==========================================" | tee -a "$RESULTS"

# ====== 第一轮实验（已归档，lr>=5e-4 未收敛） ======
# 学习率对比: lr=1e-4(0.8240) lr=5e-4(0.6406) lr=1e-3(0.6261)
# 批次大小对比: bs=32(0.6306) bs=64(0.6576) bs=128(内存交换)
# 训练轮数对比: ep=5(0.6198) ep=10(0.6338) ep=20(未完成)

# ====== 已完成的实验 ======
# 学习率调优: lr=5e-5(0.8165) lr=1e-4(0.8224) lr=2e-4(0.8079)
# for lr in 5e-5 1e-4 2e-4; do
#     LOGFILE="$LOGS_DIR/lr_${lr}_bs64_ep10.log"
#     echo ">>> lr=$lr batch=64 epochs=10 emb_dim=100 | log: $LOGFILE"
#     python nlp_code.py --lr "$lr" --batch_size 64 --epochs 10 --emb_dim 100 2>&1 | tee "$LOGFILE"
#     tail -8 "$LOGFILE"
# done

# ====== 第二轮实验（已完成） ======
# 嵌入维度对比: emb=100(0.8178) emb=200(0.8270)
# 批次大小对比: bs=32(0.8241) bs=64(0.8224)
# 训练轮数对比(bs=64): ep=5(0.8015) ep=10(0.8224) ep=20(0.8154) ep=30(0.8232)
# for dim in 100 200; do
#     LOGFILE="$LOGS_DIR/emb${dim}_lr1e-4_bs64_ep10.log"
#     echo ">>> emb_dim=$dim lr=1e-4 batch=64 epochs=10 | log: $LOGFILE"
#     python -u nlp_code.py --lr 1e-4 --batch_size 64 --epochs 10 --emb_dim "$dim" 2>&1 | tee "$LOGFILE"
#     tail -8 "$LOGFILE"
# done
# for bs in 32; do
#     LOGFILE="$LOGS_DIR/bs${bs}_lr1e-4_ep10.log"
#     echo ">>> batch=$bs lr=1e-4 epochs=10 emb_dim=100 | log: $LOGFILE"
#     python -u nlp_code.py --lr 1e-4 --batch_size "$bs" --epochs 10 --emb_dim 100 2>&1 | tee "$LOGFILE"
#     tail -8 "$LOGFILE"
# done
# for ep in 5 20 30; do
#     LOGFILE="$LOGS_DIR/ep${ep}_lr1e-4_bs64.log"
#     echo ">>> epochs=$ep lr=1e-4 batch=64 emb_dim=100 | log: $LOGFILE"
#     python -u nlp_code.py --lr 1e-4 --batch_size 64 --epochs "$ep" --emb_dim 100 2>&1 | tee "$LOGFILE"
#     tail -8 "$LOGFILE"
# done

echo ""
echo "========== 5. 嵌入维度对比（修复后重跑） ==========" | tee -a "$RESULTS"
for dim in 100 200; do
    LOGFILE="$LOGS_DIR/emb${dim}_lr1e-4_bs64_ep10_v2.log"
    echo ">>> emb_dim=$dim lr=1e-4 batch=64 epochs=10 | log: $LOGFILE" | tee -a "$RESULTS"
    python -u nlp_code.py --lr 1e-4 --batch_size 64 --epochs 10 --emb_dim "$dim" 2>&1 | tee "$LOGFILE"
    echo ""
    tail -8 "$LOGFILE" | tee -a "$RESULTS"
    echo ""
done

echo ""
echo "==========================================" | tee -a "$RESULTS"
echo "Experiment End: $(date)" | tee -a "$RESULTS"
echo "==========================================" | tee -a "$RESULTS"