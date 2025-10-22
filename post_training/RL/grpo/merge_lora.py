import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge_lora(base_dir, lora_dir, output_dir, use_cpu=False):
    """
    合并已合并的 FSDP base 模型 和 LoRA adapter。
    Args:
        base_dir (str): 已合并的 actor/base 模型目录
        lora_dir (str): LoRA adapter 权重目录
        output_dir (str): 输出的合并后模型保存目录
        use_cpu (bool): 是否在 CPU 上加载（适合显存紧张）
    """
    dtype = torch.float32 if use_cpu else torch.float16
    device_map = "cpu" if use_cpu else "auto"

    print(f"🔹 Loading base model from: {base_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    print(f"🔹 Loading LoRA adapter from: {lora_dir}")
    model = PeftModel.from_pretrained(model, lora_dir)

    print("🔹 Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # === 保存模型 ===
    print(f"💾 Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)

    # === 保存 tokenizer ===
    tokenizer_src = lora_dir if os.path.exists(os.path.join(lora_dir, "tokenizer_config.json")) else base_dir
    print(f"💾 Saving tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)
    tokenizer.save_pretrained(output_dir)

    print("✅ Merge and tokenizer save completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Merge base model + LoRA adapter")
    parser.add_argument("--base_dir", required=True, help="Path to merged base model (e.g. FSDP merged)")
    parser.add_argument("--lora_dir", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_dir", required=True, help="Output path for merged full model")
    parser.add_argument("--cpu", action="store_true", help="Use CPU to merge (for low VRAM environment)")
    args = parser.parse_args()

    merge_lora(args.base_dir, args.lora_dir, args.output_dir, args.cpu)

if __name__ == "__main__":
    main()
