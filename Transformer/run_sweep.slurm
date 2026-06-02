#!/bin/bash
#SBATCH -c 4
#SBATCH --time=239
#SBATCH --chdir=/home/scc/pb24231833/aib/transformer
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --gres=gpu:1

set -euo pipefail
cd /home/scc/pb24231833/aib/transformer

# ===== 环境激活 =====
set +u
source /home/scc/pb24231833/miniconda3/etc/profile.d/conda.sh
conda activate py313
set -u

nvidia-smi
echo ""
python nlp_code_cluster.py

# ===== 实验配置区 =====
# 要测试的参数组合
LR_LIST=(5e-5 1e-4 2e-4)
BATCH_SIZE_LIST=(32 64)
EPOCHS_LIST=(5 10)

PY_FILE="nlp_code_cluster.py"
TEMP_FILE="run_temp.py"

# ===== 辅助函数 =====
# 用法：run_exp lr batch_size epochs
# nlp_code_cluster.py 已修复 embed_dim 参数
# 每次运行自动输出 100 维和 200 维两组结果
run_exp() {
    local lr=$1 bs=$2 epochs=$3
    echo ">>> lr=${lr}  bs=${bs}  epochs=${epochs}  [embed_dim=100 & 200]"

    cp "${PY_FILE}" "${TEMP_FILE}"
    sed -i "s/^BATCH_SIZE = .*/BATCH_SIZE = ${bs}/"          "${TEMP_FILE}"
    sed -i "s/^EPOCHS = .*/EPOCHS = ${epochs}/"              "${TEMP_FILE}"
    sed -i "s/^LEARNING_RATE = .*/LEARNING_RATE = ${lr}/"    "${TEMP_FILE}"

    python "${TEMP_FILE}"
    rm -f "${TEMP_FILE}"
    echo ""
}

# ===== 实验组 1：学习率 × 批次大小联合验证 =====
# 固定 epochs=10，自动跑 embed_dim=100 & 200
echo "========== Group 1: lr × batch_size (epochs=10) =========="
for LR in "${LR_LIST[@]}"; do
    for BS in "${BATCH_SIZE_LIST[@]}"; do
        run_exp "${LR}" "${BS}" 10
    done
done

# ===== 实验组 2：训练轮数对比 =====
# 固定 lr=1e-4, bs=32，自动跑 embed_dim=100 & 200
echo "========== Group 2: epochs comparison (lr=1e-4, bs=32) =========="
for EPOCHS in "${EPOCHS_LIST[@]}"; do
    run_exp 1e-4 32 "${EPOCHS}"
done

# ===== 汇总结果 =====
echo "============================================================"
echo "All experiments done at $(date)"
echo ""
echo "Results summary:"
echo "------------------------------------------------------------"
grep -hE "Accuracy|cost time|Result Comparison" outputs/score.txt 2>/dev/null \
    || echo "(outputs/score.txt not found)"
echo "============================================================"