#!/bin/bash
# chmod +x eval_math7b.sh
# ./eval_math7b.sh

model_paths=(
    # "/root/All_in_llm/models/Qwen2.5-Math-7B"
    # "/root/All_in_llm/post_training/sft/merge_models/Qwen2.5-7B_Math_gsm8k_lora"
    # /root/All_in_llm/models/Qwen2.5-Math-1.5B
    # /root/All_in_llm/post_training/sft/merge_models/Qwen2.5-1.5B_Math_gsm8k_lora_212
    /root/All_in_llm/post_training/sft/merge_models/Qwen2.5-1.5B_Instruct_gsm8k_lora
    /root/All_in_llm/post_training/sft/merge_models/Qwen2.5-1.5B_gsm8k_lora
)

# 你要评测的 few-shot 数量列表
n_shots=(0 1 2 3 4 5)

HOST=0.0.0.0
PORT=8000

for model_path in "${model_paths[@]}"; do
    echo "Starting vLLM server for $model_path..."
    
    # 后台启动 vLLM serve
    vllm serve "$model_path" --host $HOST --port $PORT &
    VLLM_PID=$!

    # 给模型加载时间（根据显卡性能调整）
    sleep 120

    # 遍历不同 few-shot
    for n in "${n_shots[@]}"; do
        echo "Evaluating $model_path with ${n}-shot..."

        lm_eval \
            --model local-completions \
            --model_args "model=$model_path,base_url=http://$HOST:$PORT/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False" \
            --tasks gsm8k \
            --device cpu \
            --batch_size 64 \
            --output_path ./eval_results \
            --num_fewshot $n
    done

    # 停掉 vLLM 服务
    echo "Stopping vLLM server for $model_path..."
    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null
done

echo "✅ All evaluations completed."
