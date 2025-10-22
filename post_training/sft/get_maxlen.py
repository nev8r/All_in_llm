from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import warnings

# -------------------
# 配置区
# -------------------
model_name = "/root/All_in_llm/models/Qwen2.5-Math-7B"  # 本地或HF路径
dataset_path = "/root/All_in_llm/post_training/sft/data/gsm8k"
subset = "main"           # Hugging Face 加载子集名称
split = "train"           # "train" 或 "test"
sample_limit = None       # None 表示全量
# -------------------

warnings.filterwarnings("ignore")

print(f"🔹 Loading dataset: {dataset_path}/{subset} ({split}) ...")
dataset = load_dataset(dataset_path, subset, split=split)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# -------- 统计函数 --------
def analyze_lengths(lengths, title):
    lengths = np.array(lengths)
    print(f"\n===== {title} =====")
    print(f"Samples analyzed: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Median length: {np.median(lengths):.1f}")
    print(f"Max length: {np.max(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95):.1f}")
    print(f"99th percentile: {np.percentile(lengths, 99):.1f}")
    recommended = int(np.percentile(lengths, 99)) + 128
    print(f"✅ Recommended max_length: {recommended}")
    return recommended

# -------- 无模板统计 --------
plain_lengths = []
for i, item in enumerate(tqdm(dataset, desc="Tokenizing (plain Q/A)")):
    if sample_limit and i >= sample_limit:
        break
    text = f"Question: {item['question']}\nAnswer: {item['answer']}"
    tokens = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    plain_lengths.append(len(tokens))

plain_maxlen = analyze_lengths(plain_lengths, "No Template (plain Q/A)")

# -------- 带 Qwen Chat 模板统计 --------
chat_lengths = []
for i, item in enumerate(tqdm(dataset, desc="Tokenizing (Qwen chat template)")):
    if sample_limit and i >= sample_limit:
        break

    # 构造符合 Qwen2.5 的 ChatML 输入
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": f"<think>{item['answer']}</think>"}
    ]

    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    chat_lengths.append(len(tokens))

chat_maxlen = analyze_lengths(chat_lengths, "With Qwen2.5 Chat Template")

print("\n📊 Summary:")
print(f"  → Plain Q/A 99% length + buffer: {plain_maxlen}")
print(f"  → Chat template 99% length + buffer: {chat_maxlen}")
