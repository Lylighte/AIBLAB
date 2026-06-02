#!/bin/bash
# WSL 本地超参数调优脚本
# 用法：bash run_wsl.sh
set -euo pipefail

# ===== 切换到 nlp_code_new.py 所在目录（支持从任意位置运行） =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ===== 实验配置区 =====
LR_LIST=(5e-5 1e-4 2e-4)
BATCH_SIZE_LIST=(32 64)
EPOCHS_LIST=(5 10)

echo "============================================"
echo "WSL 实验开始: $(date)"
echo "运行环境: $(python nlp_code_new.py --help > /dev/null 2>&1 && echo 'OK')"
echo "GPU: $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)"
echo "============================================"
echo ""

# ===== 实验组 1：学习率 × 批次大小 =====
# 固定 epochs=10，自动跑 embed_dim=100 & 200
echo "========== Group 1: lr × batch_size (epochs=10) =========="
for LR in "${LR_LIST[@]}"; do
    for BS in "${BATCH_SIZE_LIST[@]}"; do
        echo ">>> lr=${LR}  bs=${BS}  ep=10  [dim=100 & 200]"
        python nlp_code_new.py --lr "${LR}" --batch_size "${BS}" --epochs 10
        echo ""
    done
done

# ===== 实验组 2：训练轮数对比 =====
# 固定 lr=1e-4, bs=32
echo "========== Group 2: epochs comparison (lr=1e-4, bs=32) =========="
for EPOCHS in "${EPOCHS_LIST[@]}"; do
    echo ">>> lr=1e-4  bs=32  ep=${EPOCHS}  [dim=100 & 200]"
    python nlp_code_new.py --lr 1e-4 --batch_size 32 --epochs "${EPOCHS}"
    echo ""
done

# ===== 汇总结果 =====
echo "============================================"
echo "实验完成: $(date)"
echo ""
echo "结果汇总 (outputs/score.txt):"
echo "--------------------------------------------"
grep -hE "Accuracy|cost time|Result Comparison" outputs/score.txt 2>/dev/null \
    || echo "(暂无)"
echo "============================================"