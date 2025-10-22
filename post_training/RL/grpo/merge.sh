#!/bin/bash
# set -e  # 遇到错误立即退出
# set -x  # 打印执行的命令（可注释掉）

# === 路径配置 ===
BASE_PATH="/root/All_in_llm/post_training/RL/grpo"
MODEL_NAME="qwen2.5_3b_grpo_lora"
STEP="100"

# === 自动生成路径 ===
CHECKPOINT_DIR="${BASE_PATH}/checkpoints/verl_grpo_example_gsm8k/${MODEL_NAME}/global_step_${STEP}/actor"
MODEL_ACTOR_DIR="${BASE_PATH}/model_actor"
LORA_DIR="${MODEL_ACTOR_DIR}/lora_adapter"
OUTPUT_DIR="${BASE_PATH}/model_actor_merged"

# === 1. FSDP 权重合并 ===
echo "=== [1/3] FSDP merging checkpoint ==="
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CHECKPOINT_DIR}" \
    --target_dir "${MODEL_ACTOR_DIR}"

# === 2. LoRA 权重合并 ===
echo "=== [2/3] Merging LoRA adapter ==="
python merge_lora.py \
    --base_dir "${MODEL_ACTOR_DIR}" \
    --lora_dir "${LORA_DIR}" \
    --output_dir "${OUTPUT_DIR}"

# === 3. 清理临时目录 ===
echo "=== [3/3] Cleaning up temporary directories ==="
rm -rf "${MODEL_ACTOR_DIR}"

echo "✅ Merge completed successfully!"
echo "Merged model saved at: ${OUTPUT_DIR}"
