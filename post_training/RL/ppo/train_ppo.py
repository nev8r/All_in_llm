import json
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from peft import LoraConfig, get_peft_model, PeftModel  # 新增：PEFT LoRA 支持

# 1. 配置（硬编码，基于你的设置）
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Policy/SFT model
reward_model_path = "/root/All_in_llm/post_training/RL/ppo/qwen2.5-1.5b-reward-model-lora"  # Reward model
output_dir = "/root/All_in_llm/post_training/RL/ppo/qwen2.5-1.5b-ppo-lora"  # 更新路径，包含 LoRA

# PPO 配置（简化）
training_args = PPOConfig(
    output_dir=output_dir,
    learning_rate=1.41e-5,
    num_train_epochs=1,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=100,
    save_strategy="steps",
    bf16=torch.cuda.is_available(),
    optim="adamw_torch",
    report_to=None,
    remove_unused_columns=True,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    # PPO 特定
    num_mini_batches=1,
    total_episodes=None,
    response_length=100,
    temperature=0.7,
    num_ppo_epochs=1,
    kl_coef=0.05,
    cliprange=0.2,
    vf_coef=0.1,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    whiten_rewards=False,
    kl_estimator="k1",
    reward_model_path=reward_model_path,
    exp_name="ppo_training",
    missing_eos_penalty=1.0,
    stop_token="eos",
)

# 2. 加载 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # 或 "[PAD]"
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

# Value model 和 Reward model（使用相同的 reward model）
value_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, num_labels=1, torch_dtype=torch.float32
)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path, num_labels=1, torch_dtype=torch.float32
)

# Policy model（你的 SFT model，这里用原模型）
base_policy = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
base_policy.config.pad_token_id = tokenizer.pad_token_id

# 新增：LoRA 配置（针对 Qwen 模型的注意力层）
peft_config = LoraConfig(
    r=16,  # LoRA 秩（可调，16 平衡效果/内存）
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],  # Qwen 注意力模块
    lora_dropout=0.05,  # dropout
    bias="none",
    task_type="CAUSAL_LM",  # 因果语言模型任务
)

# 应用 LoRA 到 policy
policy = get_peft_model(base_policy, peft_config)
policy.print_trainable_parameters()  # 显示可训练参数数量（应 <1%）

# Ref policy（无 PEFT，用 None 或复制）
ref_policy = None  # LoRA 下 ref 通常 None

# 3. 加载数据集（你的本地 JSON，只用 question 作为 prompt）
def load_prompt_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    prompts = [{"messages": [{"role": "user", "content": item["question"]}]} for item in data]
    return Dataset.from_list(prompts)

dataset = load_prompt_data("/root/All_in_llm/post_training/RL/ppo/data/preference.json")
train_dataset = dataset  # 无 split

def prepare_dataset(dataset, tokenizer):
    def tokenize(element):
        input_ids = tokenizer.apply_chat_template(
            element["messages"][:1],
            padding=False,
            add_generation_prompt=True,
        )
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=4,  # 调整 proc 数
    )

# 准备数据集
train_dataset = prepare_dataset(train_dataset, tokenizer)
# 过滤长度
train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512)

# 确保最后不是 EOS
assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

# 4. 创建 PPO Trainer 并训练（传入 peft_config）
trainer = PPOTrainer(
    args=training_args,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=train_dataset[0], 
    peft_config=peft_config,  # 新增：启用 LoRA
)

# print("开始 PPO 训练（LoRA）...")
trainer.train()

# 保存模型（LoRA adapter 自动保存）
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"PPO LoRA 模型已保存到 {output_dir}")

# 测试生成（加载 PEFT 模型）
def test_ppo_generation(prompt, model_path=output_dir):
    # 从 base + adapter 加载 PEFT 模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)  # 加载 LoRA adapter
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated.split("assistant\n")[-1].strip()

# 示例测试
test_prompt = "Python中的字典是什么？"
generated = test_ppo_generation(test_prompt)
print(f"PPO 生成: {generated}")