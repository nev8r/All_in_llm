import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model, PeftModel  # 新增：PEFT LoRA 支持
import torch

# 1. 数据准备（直接用你提供的数据，无需文件）
train_data = [
    {"question": "Python中的字典是什么？", "chosen": "Python中的字典是一种无序的可变容器，允许使用键-值对来存储数据。", "rejected": "Python中的字典用于存储数据。"},
    {"question": "什么是回归分析？", "chosen": "回归分析是一种统计方法，用于确定变量之间的关系，通常用于预测和模型拟合。", "rejected": "回归分析是一种数据分析方法。"},
    {"question": "如何优化机器学习模型？", "chosen": "优化机器学习模型可以通过调节超参数、选择合适的模型和特征工程来实现。", "rejected": "优化模型可以提高性能。"},
    {"question": "什么是机器学习？", "chosen": "机器学习是一种人工智能方法，通过算法和统计模型使计算机系统能够执行特定任务，而无需明确编程指令。", "rejected": "机器学习是计算机科学的一个领域。"},
    {"question": "Python的主要特点是什么？", "chosen": "Python的主要特点包括简洁的语法、强大的库支持以及跨平台的兼容性。", "rejected": "Python是一种编程语言。"},
    {"question": "如何进行数据清洗？", "chosen": "数据清洗通常包括处理缺失值、去除重复数据、标准化数据格式等步骤。", "rejected": "数据清洗是数据处理的一部分。"},
    {"question": "什么是数据库？", "chosen": "数据库是一个有组织的数据集合，允许高效的数据存储、检索和管理。", "rejected": "数据库用于存储数据。"},
    {"question": "深度学习和机器学习的区别是什么？", "chosen": "深度学习是机器学习的一个子集，主要通过神经网络处理数据，而传统机器学习算法不依赖于神经网络。", "rejected": "深度学习是机器学习的一部分。"},
    {"question": "如何使用Pandas加载CSV文件？", "chosen": "你可以使用Pandas的`read_csv()`函数来加载CSV文件，并将其转换为DataFrame。", "rejected": "使用Pandas加载CSV文件。"},
    {"question": "什么是自然语言处理？", "chosen": "自然语言处理是计算机科学与人工智能的一个子领域，专注于机器与人类语言的交互。", "rejected": "自然语言处理是计算机科学的一部分。"}
]
formatted_data = [{"prompt": item["question"], "chosen": item["chosen"], "rejected": item["rejected"]} for item in train_data]
dataset = Dataset.from_list(formatted_data)

# 2. 加载模型和 tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Policy/SFT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 基础模型加载
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # bf16 更稳定
    device_map="auto"
)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.problem_type = "regression"  # 明确为回归任务

# 新增：LoRA 配置（针对 Qwen 模型的注意力层）
peft_config = LoraConfig(
    r=32,  
    lora_alpha=64,  # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],  # Qwen 注意力模块
    lora_dropout=0.05,  # dropout
    bias="none",
    task_type="SEQ_CLS",  # 序列分类任务
)

# 应用 LoRA 到模型
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()  # 显示可训练参数数量（应 <1%）

# 3. 数据预处理（使用聊天模板）
def preprocess_function(examples):
    chosen_inputs = []
    rejected_inputs = []
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        # Qwen 聊天模板
        chosen_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}]
        rejected_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
        chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False)
        rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
        chosen_inputs.append(chosen_text)
        rejected_inputs.append(rejected_text)
    
    
    return {
        "chosen": chosen_inputs,
        "rejected": rejected_inputs,
    }

# 预处理数据集
tokenized_dataset = dataset.map(preprocess_function, batched=True)
print(tokenized_dataset)

# 4. 训练配置（优化参数）
training_args = RewardConfig(
    output_dir="./qwen2.5-1.5b-reward-model-lora",  # 更新路径，包含 LoRA
    per_device_train_batch_size=2,  # 增大批次（内存不足调1）
    gradient_accumulation_steps=4,  # 有效批次 8
    learning_rate=1e-6,  # 更小学习率
    num_train_epochs=3,  # 少 epoch 防过拟合
    logging_steps=1,  # 频繁日志监控
    save_steps=5,  # 频繁保存
    eval_strategy="no",
    remove_unused_columns=False,
    optim="adamw_torch",
    bf16=torch.cuda.is_available(),  # 启用 bf16
    max_grad_norm=1.0,  # 梯度剪裁防爆炸
    gradient_checkpointing=True,  # 省内存
    report_to=[],  # 移除 swanlab
    dataloader_pin_memory=False,  # 稳定数据加载
)

# 5. 创建 Trainer 并训练（传入 peft_config）
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,  # 新增：启用 LoRA
)

print("开始训练奖励模型（LoRA）...")
trainer.train()

# 6. 保存模型（LoRA adapter 自动保存）
trainer.save_model()
tokenizer.save_pretrained("./qwen2.5-1.5b-reward-model-lora")
print("LoRA 模型已保存到 ./qwen2.5-1.5b-reward-model-lora")

# 7. 测试训练好的奖励模型（加载 PEFT 模型，确保 fp32 稳定）
def test_reward_model(question, answer, model_path="./qwen2.5-1.5b-reward-model-lora"):
    # 从 base model + adapter 加载 PEFT 模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,  # base model
        num_labels=1,
        torch_dtype=torch.float32,  # 测试用 fp32 防 NaN
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)  # 加载 LoRA adapter
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从 base 加载 tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"输入: {tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)}")
        reward_score = outputs.logits[0].item()
    return reward_score

# 测试示例
test_question = "Python中的字典是什么？"
good_answer = "Python中的字典是一种无序的可变容器，允许使用键-值对来存储数据。"
bad_answer = "Python中的字典用于存储数据。"

good_score = test_reward_model(test_question, good_answer)
bad_score = test_reward_model(test_question, bad_answer)

print(f"\n测试结果:")
print(f"优质回答得分: {good_score:.4f}")
print(f"劣质回答得分: {bad_score:.4f}")
print(f"分数差异: {good_score - bad_score:.4f}")